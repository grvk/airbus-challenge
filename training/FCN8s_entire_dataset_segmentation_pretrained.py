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
NEW_TRAIN_VAL_TEST_IMAGES_BACKUP_PATH = \
    path.join(BACKUPS_FOLDER, "train_val_test_images_trash.pth")
TRAIN_VAL_TEST_IMAGES_BACKUP_PATH = \
    path.join(BACKUPS_FOLDER, "train_val_test_images.pth")

FILTERED_ALL_IMAGES_PATH = path.join(BACKUPS_FOLDER, "filtered_all_images.pth")
SHIPS_ONLY_PATH = path.join(BACKUPS_FOLDER, "ships_only_images.pth")
SHIPS_ONLY_SUBSETS_PATH = path.join(BACKUPS_FOLDER, "ships_only_subsets_images.pth")


# # ------------------------------------------------------------------------------
# # STEP 1: PARSE  TRAIN_VAL_TEST SET AND FIND ALL OTHER IMAGES THAT
# # ARE NOT A PART OF ANY OF THESE THREE DATASETS
# # ------------------------------------------------------------------------------

# all_images = build_images_dict(CSV_TRAIN_FILE)
# torch.save(all_images, ALL_IMAGES_BACKUP_PATH)
# all_images = torch.load(ALL_IMAGES_BACKUP_PATH)
# images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)

# filtered_all_images = all_images

# for set_type in ["val", "test"]:
#     for key in images[set_type]["images"]["ships"]:
#         del filtered_all_images["ships"][key]

#     for elem in images[set_type]["images"]["w/ships"]:
#         if elem in filtered_all_images["w/ships"]:
#             filtered_all_images["w/ships"].remove(elem)

# # randomly sample the w/ships so that the number of images with ships and
# # without was equal
# filtered_all_images = process_images_dict(filtered_all_images)
# torch.save(filtered_all_images, FILTERED_ALL_IMAGES_PATH)

# # ------------------------------------------------------------------------------
# # STEP 2: GET A SUBSET OF ALL IMAGES IN WHICH YOU CAN ALWAYS FIND A SHIP
# # FOR THIS DATASET FIND CHANNEL-WISE STD AND MEAN VALUES
# # ------------------------------------------------------------------------------

# filtered_all_images = torch.load(FILTERED_ALL_IMAGES_PATH)
# ships_only_images = filtered_all_images
# ships_only_images["w/ships"] = []
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dtset = AirbusDataset(ships_only_images, TRAIN_IMAGES_PATH,
#     "classification", transforms.ToTensor())

# dtldr = DataLoader(dataset = dtset, num_workers = 2, batch_size = 256)
# ships_only_std, ships_only_mean = find_per_channel_std_mean(dtldr, device)

# ships_only_data = {
#     "images": ships_only_images,
#     "mean": ships_only_mean,
#     "std": ships_only_std
# }
# torch.save(ships_only_data, SHIPS_ONLY_PATH)

# # ------------------------------------------------------------------------------
# # STEP 3: GET SUBSETS OF SIZE = ORIGINAL TRAIN DATA SET IN TRAIN_VAL_TEST
# #
# # ------------------------------------------------------------------------------
# ships_only_data = torch.load(SHIPS_ONLY_PATH)
# # ships_only_images = ships_only_data["images"]
# # images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)
# # ships_only_std = ships_only_data["std"]
# # ships_only_mean = ships_only_data["mean"]

# ships_only_num_of_images = len(ships_only_images["ships"])
# # number of images in the original train set
# iterations_in_epoch = len(images["train"]["images"]['ships']) + \
#     len(images["train"]["images"]['w/ships'])

# ships_only_subsets = []
# ships_only_items = list(ships_only_images["ships"].items())

# i = 0
# while i < ships_only_num_of_images:
#     ships_only_subsets.append({
#         "w/ships": [],
#         "ships": dict(ships_only_items[i: (i + iterations_in_epoch)])
#     })
#     i += iterations_in_epoch

# ships_only_subsets_data = {
#     "subsets_images": ships_only_subsets,
#     "global_std": ships_only_std,
#     "global_mean": ships_only_mean
# }

# torch.save(ships_only_subsets_data, SHIPS_ONLY_SUBSETS_PATH)
ships_only_subsets_data = torch.load(SHIPS_ONLY_SUBSETS_PATH)

ships_only_mean = ships_only_subsets_data["global_mean"]
ships_only_std = ships_only_subsets_data["global_std"]
ships_only_subsets = ships_only_subsets_data["subsets_images"]


# ------------------------------------------------------------------------------
# STEP 4: CREATE ACTUAL DATASETS FOR EACH OF THE SUBSETS AND DEFINE DATALOADER
# CREATOR FUNCTIONS
# ------------------------------------------------------------------------------
DATA_LOADER_OPTS = {
    "batch_size": 4,
    "num_workers": 4,
    "shuffle": True
}

source_transforms = [transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize(ships_only_mean, ships_only_std)]) \
    for _ in ships_only_subsets]

train_datasets = [AirbusDataset(imgs, TRAIN_IMAGES_PATH, "segmentation",
    src_trnsfrm, transforms.ToTensor()) for imgs, src_trnsfrm \
    in zip(ships_only_subsets, source_transforms)]

def train_dataloader_creator():
    cur_idx = -1

    def creator():
        global train_datasets
        nonlocal cur_idx

        num_of_datasets = len(train_datasets)
        cur_idx = cur_idx + 1 if cur_idx != num_of_datasets - 1 else 0

        print("~~CURIDX: {}".format(cur_idx))
        return DataLoader(dataset = train_datasets[cur_idx], **DATA_LOADER_OPTS)

    return creator


images = torch.load(TRAIN_VAL_TEST_IMAGES_BACKUP_PATH)

val_src_transforms = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(images["val"]["mean"], images["val"]["std"])])

val_dtset = AirbusDataset(images["val"]["images"], TRAIN_IMAGES_PATH,
    "segmentation", val_src_transforms, transforms.ToTensor())

def val_dataloader_creator():
    return DataLoader(dataset = val_dtset, **DATA_LOADER_OPTS)

# ------------------------------------------------------------------------------
# STEP 5: SET UP DEVICE, MODEL, LOSS FUNCTION, AND TEST
# ------------------------------------------------------------------------------
CLASSES_NUM = 2
LEARNING_RATE = 1e-7
MOMENTUM = .9
WEIGHT_DECAY = 5**(-4)

DATA_LOADER_OPTS = {
    "batch_size": 16,
    "num_workers": 8,
    "shuffle": True
}
NUM_OF_EPOCHS = 100
BACKUP_INTERVAL = 1

cuda_count = torch.cuda.device_count()
if cuda_count == 0:
    cuda_count = torch.device('cpu')
else:
    device = torch.device('cuda')

loss_fn = CrossEntropyLoss2D(reduction='sum').to(device)

net = FCN8s(CLASSES_NUM, 'imagenet')
if cuda_count > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(cuda_count)))
net.to(device)

optimizer = torch.optim.SGD(net.parameters(), \
    lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

trainer = Trainer(net, optimizer, loss_fn, \
    train_dataloader_creator(), val_dataloader_creator, \
    backup_interval=BACKUP_INTERVAL, device=device, final_eval_fn = None, \
    additional_backup_info=ships_only_subsets_data, is_debug_mode=False)

trainer.train(NUM_OF_EPOCHS)
