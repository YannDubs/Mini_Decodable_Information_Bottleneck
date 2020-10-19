import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.datasets import CIFAR10

from dib import get_DIB_data


class CIFAR10IsTrain(CIFAR10):
    """CIFAR10 but target contains `is_train`. This is needed for the 2player game worst case scenario"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = np.expand_dims(np.array(self.targets), 1)
        is_train = self.targets * 0 + int(self.train == "train")
        self.targets = np.append(self.targets, is_train, axis=1)

    def __getitem__(self, index):
        multi_targets = self.targets
        # often datasets have code that can only deal with a single target
        self.targets = multi_targets[:, 0]
        img, target = super().__getitem__(index)
        self.targets = multi_targets  # set back multi targets
        multi_target = (target,) + tuple(self.targets[index, 1:])
        return img, multi_target

    def append_(self, other):
        """Append a dataset to the current one."""
        self.data = np.append(self.data, other.data, axis=0)
        self.targets = np.append(self.targets, other.targets, axis=0)


class MyCIFAR10DataModule(CIFAR10DataModule):
    """CIFAR10DataModule module which also returns the index of the image and can append test to train.

    Parameters
    ----------
    mode : {"worst", "avg", "1player"}, optional
        What mode you are training with. In `1player` the target contain the indices of the images,
        in `worst` and `avg` the target contain a binary flag `is_train`. In `worst` the test and
        validation set is also appended to the training set.
    """

    def __init__(self, mode="1player", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.DATASET = get_DIB_data(CIFAR10) if mode == "1player" else CIFAR10IsTrain

    def train_dataloader(self):
        """
        CIFAR train set removes a subset to use for validation
        """
        transforms = (
            self.default_transforms()
            if self.train_transforms is None
            else self.train_transforms
        )
        kwargs = dict(
            download=False,
            transform=transforms,
            **self.extra_args,
        )

        dataset = self.DATASET(self.data_dir, train=True, **kwargs)

        train_length = len(dataset)
        dataset_train, dataset_valid = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # need to add evaluation set for training the worst case scenario
        if self.mode == "worst":

            # append test
            dataset_test = self.DATASET(self.data_dir, train=False, **kwargs)
            dataset.append_(dataset_test)

            # set validation as not is_train (it's a subset of train so by default is_train=1)
            dataset.targets[dataset_valid.indices, 1] = 0

        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
