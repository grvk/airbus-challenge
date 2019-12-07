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

BACKUP_INTERVAL = 1
NUM_OF_EPOCHS = 5
DATA_LOADER_OPTS = {
    "batch_size": 2,
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
subset_ids, _ = split_images_dict(image_ids, .05)
train_val_image_ids, test_image_ids = split_images_dict(subset_ids, .85)

train_image_ids, val_image_ids = split_images_dict(train_val_image_ids, .85)

ids = {
    "test": test_image_ids, # 600 images
    "train": train_image_ids, #~3k images
    "val": val_image_ids # 500 images
}


print(len(train_image_ids["w/ships"]))
print(len(train_image_ids["ships"].keys()))
print(len(val_image_ids["w/ships"]))
print(len(val_image_ids["ships"].keys()))
print(len(test_image_ids["w/ships"]))
print(len(test_image_ids["ships"].keys()))


train_set = AirbusDataset(train_image_ids, TRAIN_IMAGES_PATH, "segmentation", \
                                False, src_transform=transformations, target_transform=transforms.ToTensor())

val_set = AirbusDataset(val_image_ids, TRAIN_IMAGES_PATH, "segmentation", \
                                False, src_transform=transformations, target_transform=transforms.ToTensor())

loss_fn = CrossEntropyLossModified(reduction='sum').cuda()


def train_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = train_set, **DATA_LOADER_OPTS)

def val_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = val_set, **DATA_LOADER_OPTS)

trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator, val_dataloader_creator, \
    BACKUP_INTERVAL, device=device, final_eval_fn=BasicIoU, extra_backup_info=ids)
trainer.train(NUM_OF_EPOCHS)
