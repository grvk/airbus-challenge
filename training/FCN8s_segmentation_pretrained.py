import torch
import sys

from os import path
from torchvision import transforms
from torch import optim, nn
from src.trainer import Trainer
from src.models.fcn8s import FCN8s
from src.airbus_dataset import AirbusDataset
from src.misc import CrossEntropyLossModified, BasicIoU,\
    build_images_dict, process_images_dict, split_images_dict

ROOT_DIR  = path.join(path.dirname(__file__), '..')
DATA_FOLDER = path.join(ROOT_DIR, 'data')

CSV_TRAIN_FILE = path.join(DATA_FOLDER, "train_ship_segmentations_v2.csv")
TRAIN_IMAGES_PATH = path.join(DATA_FOLDER, "train_v2")

CLASSES_NUM = 2

LEARNING_RATE = 10**(-4)
MOMENTUM = .9
WEIGHT_DECAY = 5**(-4)

BACKUP_INTERVAL = 20
NUM_OF_EPOCHS = 250
DATA_LOADER_OPTS = {
    "batch_size": 1,
    "num_workers": 4,
    "shuffle": False
}

device = torch.device('cuda')

net = FCN8s(CLASSES_NUM, 'imagenet')
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


image_ids = build_images_dict(CSV_TRAIN_FILE)
image_ids = process_images_dict(image_ids)
train_image_ids, val_image_ids = split_images_dict(image_ids, .9)
print(len(train_image_ids["ships"]))
print(len(train_image_ids["w/ships"]))
print(len(val_image_ids['ships']))
print(len(val_image_ids['w/ships']))

train_set = AirbusDataset(train_image_ids, TRAIN_IMAGES_PATH, "segmentation", False, transformations)
val_set = AirbusDataset(val_image_ids, TRAIN_IMAGES_PATH, "segmentation", False, transforms.ToTensor())

loss_fn = CrossEntropyLossModified(reduction='sum').cuda()


def train_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = train_set, **DATA_LOADER_OPTS)

def val_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = val_set, **DATA_LOADER_OPTS)


trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator, val_dataloader_creator, \
    BACKUP_INTERVAL, device=device, final_eval_fn=BasicIoU)
trainer.train(1)
