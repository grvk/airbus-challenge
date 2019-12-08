import torch
import sys

from time import time
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
BACKUPS_PATH = path.join(ROOT_DIR, "backups")

device = torch.device('cuda')

# image_ids = build_images_dict(CSV_TRAIN_FILE)
# image_ids = process_images_dict(image_ids)
# subset_ids, _ = split_images_dict(image_ids, .1)
# train_val_image_ids, test_image_ids = split_images_dict(subset_ids, .85)

# train_image_ids, val_image_ids = split_images_dict(train_val_image_ids, .85)

# train_set = AirbusDataset(train_image_ids, TRAIN_IMAGES_PATH, "classification", False, src_transform=transforms.ToTensor())
# val_set = AirbusDataset(val_image_ids, TRAIN_IMAGES_PATH, "classification", False, src_transform=transforms.ToTensor())

# def find_per_channel_std_mean(dtloader, device):
#     means, means_of_sq, batches_count = [], [], []
#     for i, (imgs, _) in enumerate(dtloader):
#         print(i)
#         imgs = imgs.to(device)
#         imgs_sq = imgs.pow(2)
#         mean = imgs.mean(dim=(0,2,3))
#         mean_of_sq = imgs_sq.mean(dim=(0,2,3))
#         means.append(mean)
#         means_of_sq.append(mean_of_sq)
#         batches = imgs.size()[0]
#         batches_count.append(torch.Tensor([batches, batches, batches]).to(device))

#     means = torch.stack(means, dim=0)
#     means_of_sq = torch.stack(means_of_sq, dim = 0)
#     batches_count = torch.stack(batches_count, dim = 0)
#     total_batches_count = torch.sum(batches_count, dim = 0)

#     combined = torch.stack([means, means_of_sq, batches_count], dim=0)
#     avg_means_of_sq = torch.mul(combined[1, :, :], combined[2, :, :]).sum(dim=0).div(total_batches_count)
#     avg_means = torch.mul(combined[0, :, :], combined[2, :, :]).sum(dim=0).div(total_batches_count)
#     avg_means_squared = avg_means.pow(2)

#     final_mean = avg_means
#     final_std = torch.sqrt(avg_means_of_sq - avg_means_squared)
#     return final_std.cpu().numpy().tolist(), final_mean.cpu().numpy().tolist()

# train_dataloader = torch.utils.data.DataLoader(dataset = train_set, num_workers=4, batch_size=128)
# val_dataloader = torch.utils.data.DataLoader(dataset = val_set, num_workers=4, batch_size=128)

# print("Finding Normalization settings. Train set size = {}".format(len(train_set)))

# train_normalization_stds, train_normalization_means = find_per_channel_std_mean(train_dataloader, device)
# val_normalization_stds, val_normalization_means = find_per_channel_std_mean(val_dataloader, device)


# datasets = {
#     "test": test_image_ids, # 600 images
#     "train": train_image_ids, #~3k images
#     "val": val_image_ids, # 500 images
#     "train_stds": train_normalization_stds,
#     "train_means": train_normalization_means,
#     "val_stds": val_normalization_stds,
#     "val_means": val_normalization_means
# }

PATH_TO_DATASETS_BACKUP = path.join(BACKUPS_PATH, "datasets_info.pth")

# torch.save(datasets, PATH_TO_DATASETS_BACKUP)
datasets = torch.load(PATH_TO_DATASETS_BACKUP)


print("Train - Number of images without ships: {}".format(len(datasets["train"]["w/ships"])))
print("Train - Number of images with ships: {}".format(len(datasets["train"]["ships"].keys())))
print("Train - Normalization - STDs: {s} - MEANs: {m}".format(s=datasets["train_stds"], m=datasets["train_means"]))

print("Val - Number of images without ships: {}".format(len(datasets["val"]["w/ships"])))
print("Val - Number of images with ships: {}".format(len(datasets["val"]["ships"].keys())))
print("Val - Normalization - STDs: {s} - MEANs: {m}".format(s=datasets["val_stds"], m=datasets["val_means"]))

print("Test - Number of images without ships: {}".format(len(datasets["test"]["w/ships"])))
print("Test - Number of images with ships: {}".format(len(datasets["test"]["ships"].keys())))


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

import torchfcn
import fcn
import torch.nn.functional as F

state_dict = torch.load(fcn.data.cached_download(url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms', \
    path=path.join(DATA_FOLDER, 'fcn16s_from_caffe.pth'), md5='991ea45d30d632a01e5ec48002cac617',))

model = torchfcn.models.FCN8s(n_class=2)
fcn16s = torchfcn.models.FCN16s()
fcn16s.load_state_dict(state_dict)

def copy_params_from_fcn16s(fcn8s, fcn16s):
    for name, l1 in fcn16s.named_children():
        try:
            if name == "score_fr":
                nn.init.xavier_uniform_(fcn8s.score_fn.weight)
                if fcn8s.score_fn.bias is not None:
                    fcn8s.score_fn.bias.data.fill_(.01)
                continue
            elif name == "score_pool4":
                nn.init.xavier_uniform_(fcn8s.score_pool4.weight)
                if fcn8s.score_pool4.bias is not None:
                    fcn8s.score_pool4.bias.data.fill_(.01)
                continue
            elif name == "upscore2":
                nn.init.xavier_uniform_(fcn8s.upscore2.weight)
                if fcn8s.upscore2.bias is not None:
                    fcn8s.upscore2.bias.data.fill_(.01)
                continue
                
            l2 = getattr(fcn8s, name)
            l2.weight  # skip ReLU / Dropout
        except Exception:
            continue
        assert l1.weight.size() == l2.weight.size()
        l2.weight.data.copy_(l1.weight.data)
        if l1.bias is not None:
            assert l1.bias.size() == l2.bias.size()
            l2.bias.data.copy_(l1.bias.data)
            
copy_params_from_fcn16s(model, fcn16s)

model = model.cuda()
# net = model


net = FCN8s(CLASSES_NUM, 'imagenet')
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

train_means = datasets["train_means"]
train_stds = datasets["train_stds"]
val_means = datasets["val_means"]
val_stds = datasets["val_stds"]

train_src_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_means, train_stds)
])

val_src_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(val_means, val_stds)
])


train_set = AirbusDataset(datasets["train"], TRAIN_IMAGES_PATH, "segmentation", \
                                False, src_transform=train_src_transformations, target_transform=transforms.ToTensor())

val_set = AirbusDataset(datasets["val"], TRAIN_IMAGES_PATH, "segmentation", \
                                False, src_transform=val_src_transformations, target_transform=transforms.ToTensor())

# loss_fn = CrossEntropyLossModified(reduction='sum').to(device)

def loss_fn(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = F.log_softmax(input, dim=1)
    
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def train_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = train_set, **DATA_LOADER_OPTS)

def val_dataloader_creator():
    return torch.utils.data.DataLoader(dataset = val_set, **DATA_LOADER_OPTS)

import numpy as np
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def final_fn(out, labels):
    return 0

trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator, val_dataloader_creator, \
    BACKUP_INTERVAL, device=device, final_eval_fn=None, extra_backup_info=datasets)
trainer.train(10)
