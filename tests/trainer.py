import numpy as np
import shutil
import torch
import torchvision
import torchvision.transforms as transforms

from random import randrange
from os import path, listdir

from .simple_model import ConvNet
from src.trainer import Trainer

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train_set = torchvision.datasets.MNIST(root = "./data", train = True, \
        download = True, transform=transform)
test_set = torchvision.datasets.MNIST(root = "./data", train = False, \
        download = True, transform=transform)

val_set, train_set = torch.utils.data.random_split( \
    full_train_set, (10000, len(full_train_set) - 10000))

# running on cpu
device = torch.device('cpu')
net = ConvNet()
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=.01)

opts = {
    "shuffle": True,
    "batch_size": 16,
    "num_workers": 4
}

def train_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = train_set, **opts)

def val_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = val_set, **opts)

# test 1
epoch_num = 5

custom_back_up_path = path.join(path.dirname(__file__), \
    '../backups-{}/'.format(randrange(1000000000, 9999999999)))

trainer = Trainer(net, optimizer, net.criterion, \
    train_dataloader_creator, val_dataloader_creator, 10, device=device, \
    custom_back_up_path = custom_back_up_path)

trainer.train(epoch_num)

print("Final epoch idx = ".format(trainer.cur_epoch_idx))
assert trainer.cur_epoch_idx == epoch_num - 1, \
    "Epoch index not updated properly. Expected {}".format(epoch_num - 1)

print("Final sma difference on train dataset: {}".format(trainer.train_loss_diff))
assert trainer.train_loss_diff != -1, "Bug: It's highly unlikely that diff = -1"

print("Final sma difference on val dataset: {}".format(trainer.val_loss_diff))
assert trainer.val_loss_diff != -1, "Bug: It's highly unlikely that diff = -1"

print("Train losses: {}".format(trainer.train_loss_history))
assert len(trainer.train_loss_history) == epoch_num, \
    "Train loss history doesn't contain all losses"
assert all(loss >=0 for loss in trainer.train_loss_history), \
    "Can't have negative train loss"
assert any(loss > 0 for loss in trainer.train_loss_history), \
    "None of the train losses > 0"

print("Validation losses: {}".format(trainer.validation_loss_history))
assert len(trainer.validation_loss_history) == epoch_num, \
    "Validation loss history doesn't contain all losses"
assert all(loss >=0 for loss in trainer.validation_loss_history), \
    "Can't have negative validation loss"
assert any(loss > 0 for loss in trainer.validation_loss_history), \
    "None of the validation losses > 0"

print("Final accuracy on validation set: {}".format(trainer.final_val_acc))
assert trainer.final_val_acc > .9, "Final validation accuracy <= 90%"

print("Final accuracy on train set: {}".format(trainer.final_train_acc))
assert trainer.final_train_acc > .9, "Final train accuracy <= 90%"

files = listdir(custom_back_up_path)
files = [f for f in files if "pth" in f]

print("Backup files: {}".format(files))
l = len(files)

assert l == 1, "Expected 1 backed up files. Recieved={}".format(l)
for file_name in files:
    bckp = torch.load(path.join(custom_back_up_path, file_name))

    keys = ['states', 'fn_strings', 'trainer_state']
    assert all(key in bckp for key in keys), "Some of the keys not present: {}".format(keys)

    keys = ['model', 'optimizer']
    for key in keys:
        assert isinstance(bckp['states'][key], dict)

    keys = ['loss_fn', 'train_dataloader_creator', 'val_dataloader_creator']
    assert all(key in bckp['fn_strings'] for key in keys), "Some of the keys not present: {}".format(keys)

    keys = ['train_loss_history', 'validation_loss_history', \
        'backup_interval', 'cur_epoch_idx', 'cur_train_loss', 'cur_val_loss', \
        'train_loss_diff', 'val_loss_diff', 'final_train_acc', 'final_val_acc', \
        'back_up_path']

    for key in keys:
        one = getattr(trainer, key)
        two = bckp['trainer_state'][key]
        assert one == two, "Expected equal values for key={k}. {o} and {t}".format(k=key, o=one, t=two)

shutil.rmtree(custom_back_up_path)

# test 2
epoch_num = 2
custom_back_up_path = path.join(path.dirname(__file__), \
    '../backups-{}/'.format(randrange(1000000000, 9999999999)))

trainer = Trainer(net, optimizer, net.criterion, \
    train_dataloader_creator, val_dataloader_creator, 1, device=device, \
    custom_back_up_path = custom_back_up_path)
trainer.train(epoch_num)

files = listdir(custom_back_up_path)
files = [f for f in files if "pth" in f]

print("Backup files: {}".format(files))
l = len(files)

assert l == 3, "Expected 3 backed up files. Recieved={}".format(l)
shutil.rmtree(custom_back_up_path)

print("Done testing!!!")
