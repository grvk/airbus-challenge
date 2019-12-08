import unittest

import sys
import builtins as __builtin__

import numpy as np
import shutil
import torch
import torchvision
import torchvision.transforms as transforms

from random import randrange
from os import path, listdir

from .simple_model import ConvNet
from src.trainer import Trainer


class StateAndBackupTests(unittest.TestCase):


    DATA_PATH = path.join(path.dirname(__file__), '../data/')
    LEARNING_RATE = 0.01
    DATALOADER_OPTS = {
        "shuffle": True,
        "batch_size": 64
    }

    def _get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _get_val_train_sets(self, transform):
        mnist = torchvision.datasets.MNIST( \
            root = StateAndBackupTests.DATA_PATH, train = True, \
            download = True, transform=transform)

        bndry = 10000
        return torch.utils.data.random_split(mnist, (bndry, len(mnist) - bndry))

    def train_dataloader_creator(self):
        return torch.utils.data.DataLoader( \
            dataset = self.train_set, **StateAndBackupTests.DATALOADER_OPTS)

    def val_dataloader_creator(self):
        return torch.utils.data.DataLoader( \
            dataset = self.val_set, **StateAndBackupTests.DATALOADER_OPTS)

    def setUp(self):
        transform = self._get_transforms()
        self.val_set, self.train_set = self._get_val_train_sets(transform)

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device('cpu')
        net = ConvNet()
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=StateAndBackupTests.LEARNING_RATE)

        random_int = randrange(1000000000, 9999999999)
        self.custom_back_up_path = path.join( \
            path.dirname(__file__), '../backups-{}/'.format(random_int))
        print("Find backups in: {}".format(self.custom_back_up_path))

        backup_interval = 5
        self.epochs = 5

        self.trainer = Trainer(net, optimizer, net.criterion, \
            self.train_dataloader_creator, self.val_dataloader_creator,
            backup_interval, device=device, \
            custom_back_up_path = self.custom_back_up_path)

        self.trainer.train(self.epochs)

    def tearDown(self):
        try:
            shutil.rmtree(self.custom_back_up_path)
        except:
            print("Can't remove backup folder: {}. Most likely " \
            "it doesn't exist".format(self.custom_back_up_path))

    def test_smoke(self):

        print("Final epoch idx = {}".format(self.trainer.cur_epoch_idx))
        assert self.trainer.cur_epoch_idx == self.epochs, \
            "Epoch index not updated properly. Expected {}".format(self.epochs - 1)

        print("Final sma difference on train dataset: {}".format(self.trainer.train_loss_diff))
        assert self.trainer.train_loss_diff != -1, "Bug: It's highly unlikely that diff = -1"

        print("Final sma difference on val dataset: {}".format(self.trainer.val_loss_diff))
        assert self.trainer.val_loss_diff != -1, "Bug: It's highly unlikely that diff = -1"

        print("Train losses: {}".format(self.trainer.train_loss_history))
        assert len(self.trainer.train_loss_history) == self.epochs, \
            "Train loss history doesn't contain all losses"
        assert all(loss >=0 for loss in self.trainer.train_loss_history), \
            "Can't have negative train loss"
        assert any(loss > 0 for loss in self.trainer.train_loss_history), \
            "None of the train losses > 0"

        print("Validation losses: {}".format(self.trainer.validation_loss_history))
        assert len(self.trainer.validation_loss_history) == self.epochs, \
            "Validation loss history doesn't contain all losses"
        assert all(loss >=0 for loss in self.trainer.validation_loss_history), \
            "Can't have negative validation loss"
        assert any(loss > 0 for loss in self.trainer.validation_loss_history), \
            "None of the validation losses > 0"

        files = listdir(self.custom_back_up_path)

        # leave only names of backup files
        files = [f for f in files if "pth" in f]

        print("Backup files: {}".format(files))
        l = len(files)

        assert l == 2, "Expected 2 backed up files. Recieved={}".format(l)
        for file_name in files:
            bckp = torch.load(path.join(self.custom_back_up_path, file_name))

            keys = ['states', 'fn_strings', 'trainer_state']
            assert all(key in bckp for key in keys), \
                "Some of the keys not present: {}".format(keys)

            keys = ['model', 'optimizer']
            for key in keys:
                assert isinstance(bckp['states'][key], dict)

            keys = ['loss_fn', 'train_dataloader_creator', 'val_dataloader_creator']
            assert all(key in bckp['fn_strings'] for key in keys), \
                "Some of the keys not present: {}".format(keys)

            keys = ['train_loss_history', 'validation_loss_history', \
                'backup_interval', 'cur_epoch_idx', 'cur_train_loss', \
                'cur_val_loss', 'train_loss_diff', 'val_loss_diff', \
                'back_up_path']

            for key in keys:
                one = getattr(self.trainer, key)
                two = bckp['trainer_state'][key]
                assert one == two, "Expected equal values for key={k}. " \
                    "{o} and {t}".format(k=key, o=one, t=two)
