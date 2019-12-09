import torch
from torchvision import transforms
from os import path
from src.models.fcn8s import FCN8s
from src import *

ROOT_DIR  = path.join(path.dirname(__file__), '..')
DATA_FOLDER = path.join(ROOT_DIR, 'data')
BACKUPS_FOLDER = path.join(ROOT_DIR, "backups")

CSV_TRAIN_FILE = path.join(DATA_FOLDER, "train_ship_segmentations_v2.csv")
TRAIN_IMAGES_PATH = path.join(DATA_FOLDER, "train_v2")
ALL_IMAGES_BACKUP_PATH = path.join(BACKUPS_FOLDER, "all_images.pth")
TRAIN_VAL_TEST_IMAGES_BACKUP_PATH = \
    path.join(BACKUPS_FOLDER, "train_val_test_images.pth")


# split_find_metrics_and_backup_images(\
#     CSV_TRAIN_FILE, TRAIN_VAL_TEST_IMAGES_BACKUP_PATH, ALL_IMAGES_BACKUP_PATH,
#     TRAIN_IMAGES_PATH, TRAIN_IMAGES_PATH, TRAIN_IMAGES_PATH)
images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)

CLASSES_NUM = 2
LEARNING_RATE = 1e-7
MOMENTUM = .9
WEIGHT_DECAY = 5**(-4)

DATA_LOADER_OPTS = {
    "batch_size": 2,
    "num_workers": 4,
    "shuffle": True
}
NUM_OF_EPOCHS = 1 # TO UPDATE
BACKUP_INTERVAL = 1 # TO UPDATE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FCN8s(CLASSES_NUM, 'imagenet')
net.to(device)

optimizer = torch.optim.SGD(net.parameters(), \
    lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# trn_src_tf, val_src_tf, test_src_tf =
source_transforms = [transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(images[tp]["mean"], images[tp]["std"])
]) for tp in ["train", "val", "test"]]

# trn_target_tf, val_target_tf, test_target_tf =
target_transforms = [transforms.ToTensor() for tp in ["train", "val", "test"]]

imgs_opts = [images[tp]["images"] for tp in ["train", "val", "test"]]

trn_dtset, val_dtset, test_dtst = \
    [AirbusDataset(imgs_opt, TRAIN_IMAGES_PATH, "segmentation",
        src_tf, target_tf) for imgs_opt, src_tf, target_tf in \
        zip(imgs_opts, source_transforms, target_transforms)]

loss_fn = CrossEntropyLoss2D(reduction='sum').to(device)

def train_dataloader_creator():
    return DataLoader(dataset = trn_dtset, **DATA_LOADER_OPTS)

def val_dataloader_creator():
    return DataLoader(dataset = val_dtset, **DATA_LOADER_OPTS)

# remove is_debug_mode=True once you actually start training
# TODO: add final_eval_fn
trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator, val_dataloader_creator, \
    backup_interval=BACKUP_INTERVAL, device=device, \
    final_eval_fn = None, additional_backup_info=images, is_debug_mode=True)

#trainer.train(NUM_OF_EPOCHS)
