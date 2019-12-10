import torch
from torchvision import transforms, models as M
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

FILTERED_ALL_IMAGES_PATH = path.join(BACKUPS_FOLDER, "filtered_all_images.pth")
FILTERED_IMAGES_DATA_PATH = path.join(BACKUPS_FOLDER, "filtered_images_data.pth")
FILTERED_IMAGES_SUBSETS_DATA_PATH = path.join(BACKUPS_FOLDER, \
                                            "filtered_images_subsets_data.pth")

# o_train_val_test_images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)
#
# # set of images without those in train and val sets. Number of input images with
# # and without ships is equal
# filtered_all_images = torch.load(FILTERED_ALL_IMAGES_PATH)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtset = AirbusDataset(filtered_all_images, TRAIN_IMAGES_PATH,
#     "classification", transforms.ToTensor())
#
# dtldr = DataLoader(dataset = dtset, num_workers = 2, batch_size = 256)
# filtered_images_std, filtered_images_mean = find_per_channel_std_mean(dtldr, device)
#
# filtered_images_data = {
#     "images": filtered_all_images,
#     "mean": filtered_images_mean,
#     "std": filtered_images_std
# }
# torch.save(filtered_images_data, FILTERED_IMAGES_DATA_PATH)
# filtered_images_data = torch.load(FILTERED_IMAGES_DATA_PATH)
# filtered_all_images = filtered_images_data["images"]
#
# num_of_filtered_images = len(filtered_all_images)
# # number of images in the original train set
# iterations_in_epoch = len(o_train_val_test_images["train"]["images"]['ships']) + \
#     len(o_train_val_test_images["train"]["images"]['w/ships'])
#
# filtered_images_subsets = []
# filtered_images_with_ships = list(filtered_all_images["ships"].items())
#
# i = 0
# while i < num_of_filtered_images:
#     filtered_images_subsets.append({
#         "w/ships": filtered_all_images["w/ships"][i: (i + iterations_in_epoch)],
#         "ships": dict(filtered_images_with_ships[i: (i + iterations_in_epoch)])
#     })
#     i += iterations_in_epoch
#
# filtered_images_subsets_data = {
#     "subsets_images": filtered_images_subsets,
#     "global_std": filtered_images_data["std"],
#     "global_mean": filtered_images_data["mean"]
# }
#
# torch.save(filtered_images_subsets_data, FILTERED_IMAGES_SUBSETS_DATA_PATH)
filtered_images_subsets_data = torch.load(FILTERED_IMAGES_SUBSETS_DATA_PATH)

filtered_images_mean = filtered_images_subsets_data["global_mean"]
filtered_images_std = filtered_images_subsets_data["global_std"]
filtered_images_subsets = filtered_images_subsets_data["subsets_images"]


# SET UP THE MODEL
CLASSES_NUM = 2
LEARNING_RATE = 1e-7
MOMENTUM = .9
WEIGHT_DECAY = 5**(-4)

DATA_LOADER_OPTS = {
    "batch_size": 2,
    "num_workers": 4,
    "shuffle": True
}
NUM_OF_EPOCHS = 100
BACKUP_INTERVAL = 1

source_transforms = [transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize(filtered_images_mean, filtered_images_std)]) \
    for _ in filtered_images_subsets]

train_datasets = [AirbusDataset(imgs, TRAIN_IMAGES_PATH, "segmentation",
    src_trnsfrm, transforms.ToTensor()) for imgs, src_trnsfrm \
    in zip(filtered_images_subsets, source_transforms)]

def train_dataloader_creator():
    cur_idx = -1

    def creator():
        global train_datasets
        nonlocal cur_idx

        num_of_datasets = len(train_datasets)
        cur_idx = cur_idx + 1 if cur_idx != num_of_datasets - 1 else 0

        return DataLoader(dataset = train_datasets[cur_idx], **DATA_LOADER_OPTS)

    return creator

images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)

val_src_transforms = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(images["val"]["mean"], images["val"]["std"])])

val_dtset = AirbusDataset(images["val"]["images"], TRAIN_IMAGES_PATH,
    "segmentation", val_src_transforms, transforms.ToTensor())

def val_dataloader_creator():
    return DataLoader(dataset = val_dtset, **DATA_LOADER_OPTS)

cuda_count = torch.cuda.device_count()
if cuda_count == 0:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

net = M.resnet101(True)
net.fc = torch.nn.Linear(2048, CLASSES_NUM)
torch.nn.init.xavier_uniform_(net.fc.weight)
net.fc.bias.data.fill_(0.01)

if cuda_count > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(cuda_count)))
net.to(device)

loss_fn = CrossEntropyLoss2D(reduction='sum').to(device)

optimizer = torch.optim.SGD(net.parameters(), \
    lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# remove is_debug_mode=True once you actually start training
# TODO: add final_eval_fn
trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator, val_dataloader_creator, \
    backup_interval=BACKUP_INTERVAL, device=device, \
    final_eval_fn = None, additional_backup_info=filtered_images_subsets_data, is_debug_mode=True)

#trainer.train(NUM_OF_EPOCHS)
