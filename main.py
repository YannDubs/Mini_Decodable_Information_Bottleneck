from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dib import *
from pl_bolts.datamodules import CIFAR10DataModule


class MLP(nn.Module):
    """Multi Layer Perceptron."""

    def __init__(self, dim_in, dim_out, n_hid_layers=2, dim_hid=128):
        layers = [nn.Linear(dim_in, dim_hid), nn.ReLU()]
        for _ in range(n_hid_layers):
            layers += [nn.Linear(dim_hid, dim_hid), nn.ReLU()]
        layers += [nn.Linear(dim_hid, dim_out)]
        self.module = nn.Sequential(layers)

    def forward(self, x):
        return module(x)


class Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder = partial(
            MLP,
            dim_in=(32 ** 2) * 3,
            dim_out=self.hparams.z_dim * (1 + int(self.hparams.encoder.is_stochastic)),
            n_hid_layers=self.hparams.encoder.n_hid_layers,
            dim_hid=self.hparams.encoder.dim_hid,
        )
        self.V = partial(
            MLP, dim_in=self.hparams.z_dim, dim_out=self.hparams.z_dim, **self.hparams.V
        )
        self.head_suff = self.V()

        if self.hparams.encoder.dim_hid.is_contrain_norm:
            self.batch_norm = torch.nn.BatchNorm1d(
                num_features=self.hparams.z_dim, affine=False
            )

    ### FORWARD ###
    def forward(self, X):
        batch_size = X.size(0)

        # z_sample, shape=[n_samples, batch_size, z_dim]
        if self.self.hparams.encoder.is_stochastic:
            n_samples = 1 if self.training else self.n_test_samples
            z_suff_stat = self.encoder(X)
            z_mean, z_std = z_suff_stat.view(batch_size, -1, 2).unbind(-1)
            z_std = F.softplus(z_std)
            p_zCx = MultivariateNormalDiag(z_mean, z_std)
            z_sample = p_zCx.rsample([n_samples])
        else:
            n_samples = 1
            z_sample = self.encoder(X).unsqueeze(0)  # unsqueeze as if 1 sample

        # batch norm without hyperparameters to ensure norm cannot diverge
        if self.training and self.hparams.encoder.dim_hid.is_contrain_norm:
            z_sample = self.batch_norm(z_sample.squeeze(0)).unsqueeze(0)

        y_pred = mean_p_logits(self.Q_zy(z_sample))

        return self.head_sufficiency(Z), Z

    ### TRAINING ###
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)

        loss = F.cross_entropy(pred, y)
        acc = accuracy(F.log_softmax(pred, dim=1), y)

        result = pl.TrainResult(loss)
        result.log(
            "loss/train", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        result.log(
            "accuracy/train",
            acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        return result

    ### EVALUATION ###
    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)

        loss = F.cross_entropy(pred, y)
        acc = accuracy(F.log_softmax(pred, dim=1), y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log("loss/val", loss, prog_bar=True, logger=True)
        result.log("accuracy/val", acc, prog_bar=True, logger=True)

        return result

    def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)

        loss = F.cross_entropy(pred, y)
        acc = accuracy(F.log_softmax(pred, dim=1), y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log("loss/test", loss)
        result.log("accuracy/test", acc)

        return result

    ### OPTIMIZERS + SCHEDULERS ###
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )

        if (
            "scheduler" in self.hparams.lr_scheduler
            and "steps_per_epoch" in self.hparams.lr_scheduler.scheduler.params.keys()
        ):
            steps_per_epoch = (
                self.hparams.module.train_size // self.hparams.data.batch_size
            )
            self.hparams.lr_scheduler.scheduler.params.steps_per_epoch = steps_per_epoch

        if self.hparams.lr_scheduler.name == "lr_rapiid2":
            schedulers = self.get_rapiid2_schedulers(optimizer)
        elif self.hparams.lr_scheduler.name == "lr_rapiid":
            schedulers = self.get_rapiid_schedulers(optimizer)
        else:
            scheduler = {
                "scheduler": hydra.utils.instantiate(
                    self.hparams.lr_scheduler.scheduler, optimizer=optimizer,
                ),
                "name": "learning_rate",
                **self.hparams.lr_scheduler.kwargs,
            }
            schedulers = [scheduler]

        return [optimizer], schedulers


@hydra.main(config_name="config")
def main(cfg):
    """Function only used from CLI => to keep main() usable in jupyter"""
    pass


if __name__ == "__main__":
    main()
