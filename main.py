from functools import partial
import os

import logging

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from dib import DIBWrapper, DIBLoss
from utils import prod, MLP, CrossEntropyLossGeneralize, get_exponential_decay_gamma
from data import MyCIFAR10DataModule

logger = logging.getLogger(__name__)


@hydra.main(config_name="config")
def main(cfg):

    pl.seed_everything(cfg.seed)
    cfg.paths.base_dir = hydra.utils.get_original_cwd()

    # 1 PLAYER GAME SETTING (BOB)
    cfg.current_mode = "1player"

    # dataset arguments will be used by model
    datamodule = MyCIFAR10DataModule(**cfg.data.kwargs)
    cfg.data.x_shape = prod(datamodule.dims)  # will flatten input for MLP encoder
    cfg.data.n_classes = datamodule.num_classes
    cfg.data.n_train = datamodule.num_samples

    module_bob = DIBBob(hparams=cfg)
    trainer = get_trainer(cfg)

    logger.info("TRAIN / EVALUATE representation and 1 player game scenario (Bob).")
    trainer.fit(module_bob, datamodule=datamodule)
    evaluate(trainer, datamodule, cfg, "1player")

    # 2 PLAYERS GAME SETTING (ALICE)
    for mode in cfg.alice_modes:
        cfg.current_mode = mode
        logger.info(f"TRAIN / EVALUATE 2nd player (Alice) in {mode} case.")

        gamma = get_gamma(cfg)  # weight of evaluation set
        module_alice = DIBAlice(hparams=cfg, gamma=gamma, model=module_bob.model)
        trainer = get_trainer(cfg)
        datamodule = MyCIFAR10DataModule(mode=mode, **cfg.data.kwargs)

        trainer.fit(module_alice, datamodule=datamodule)
        evaluate(trainer, datamodule, cfg, f"2player_{mode}")


class DIBBob(pl.LightningModule):
    """Main network for Bob/1player game."""

    def __init__(self, hparams):
        super().__init__()
        self.accuracy = pl.metrics.Accuracy()
        self.hparams = hparams

        V = partial(MLP, **self.hparams.V)

        self.model = DIBWrapper(
            Encoder=partial(MLP), V=V, **self.hparams.encoder  # uses a MLP encoder
        )

        self.loss = DIBLoss(V=V, **self.hparams.loss)

    def forward(self, X):
        # MLP uses a flatten input
        X = torch.flatten(X, start_dim=1)
        return self.model(X)

    def training_step(self, batch, _):
        X, targets = batch
        out = self(X)

        loss = self.loss(out, targets)
        # DEV
        self.log("H_V_yCz", self.loss.H_V_yCz)
        self.log("H_V_nCz", self.loss.H_V_nCz)

        if self.model.is_stochastic:
            self.log("z_mean_norm", self.model.z_mean_norm)
            self.log("z_std", self.model.z_std)

        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(out[0], targets[0]))
        return loss

    def evaluate(self, batch, _):
        X, targets = batch
        y, *_ = targets
        y_pred, _ = self(X)
        loss = F.cross_entropy(y_pred, y)
        acc = self.accuracy(y_pred, y)
        return loss, acc

    def validation_step(self, batch, _):
        loss, acc = self.evaluate(batch, _)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, _):
        loss, acc = self.evaluate(batch, _)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        cfgo = self.hparams.optimizer
        params = list(self.named_parameters())

        # use a smaller learning rate for the encoder
        param_groups = [
            {
                "params": [p for n, p in params if "heads_min" in n],
                "lr": cfgo.lr * cfgo.lr_factor_Vmin,
            },
            {"params": [p for n, p in params if "heads_min" not in n], "lr": cfgo.lr},
        ]

        optimizer = torch.optim.Adam(param_groups)

        max_epochs = self.hparams.trainer.max_epochs
        gamma = get_exponential_decay_gamma(cfgo.scheduling_factor, max_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        return [optimizer], [scheduler]


class DIBAlice(DIBBob):
    """Main network for Bob/2player game."""

    def __init__(self, hparams, gamma, model):
        super().__init__(hparams)
        self.model = model
        self.model.set_2nd_player_()

        self.loss = CrossEntropyLossGeneralize(gamma=gamma)

    def training_step(self, batch, _):
        X, targets = batch
        y_pred, _ = self(X)
        loss = self.loss(y_pred, targets)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(y_pred, targets[0]))
        return loss


def get_trainer(cfg):
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=cfg.paths.pretrained,
        monitor="train_loss",  # monitor train to understand effect of loss on generalization (not early stopping)
        mode="min",
        verbose=True,
    )

    loggers = []

    if "tensorboard" in cfg.logger.loggers:
        loggers.append(TensorBoardLogger(**cfg.logger.tensorboard))

    if "csv" in cfg.logger.loggers:
        loggers.append(CSVLogger(**cfg.logger.csv))

    # Dev !!
    if "wandb" in cfg.logger.loggers:
        loggers.append(WandbLogger(**cfg.logger.wandb))

    trainer = pl.Trainer(
        logger=loggers,
        checkpoint_callback=model_checkpoint,
        **cfg.trainer,
    )

    return trainer


def evaluate(trainer, datamodule, cfg, mode):
    """Evaluate the model and save to file."""
    metrics = trainer.test(datamodule=datamodule)

    with open(cfg.paths.eval, "a") as f:
        if os.stat(cfg.paths.eval).st_size == 0:
            f.write("mode,beta,seed,acc,loss" + "\n")  # header

        beta, seed = cfg.loss.beta, cfg.seed
        acc, loss = metrics[0]["test_acc"], metrics[0]["test_loss"]
        f.write(f"{mode}, {beta}, {seed}, {acc}, {loss}" + "\n")


def get_gamma(cfg):
    """Return the weight that should give to test set in 2 player game scenario."""
    if cfg.current_mode == "avg":
        gamma = 0  # no test set
    elif cfg.current_mode == "worst":
        n_not_train = 70000 - cfg.data.n_train
        data_weight = cfg.data.n_train / n_not_train
        gamma = -0.1 * data_weight  # gamma=0.1, but rescale as more train than test

    return gamma


if __name__ == "__main__":
    main()
