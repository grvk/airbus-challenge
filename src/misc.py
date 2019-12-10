import collections
import random
import pandas as pd
import numpy as np
import torch
from .airbus_dataset import AirbusDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def ClassificationAccuracy(output, target):
    """
    ClassificationAccuracy on a given batch

    Args:
        output(:obj:`torch.Tensor`) - predicted segmentation mask
            of shape BATCHES x SCORES FOR DIFFERENT CLASSES
        target(:obj:`torch.Tensor`) - expected segmentation mask
            of shape BATCHES x SCORES FOR DIFFERENT CLASSES

    Returns:
        Classification Accuracy averaged over the batch of images
    """
    predictions = torch.argmax(output.data, 1) # indices of the predicted clases
    correct = (predictions == target).sum().item()
    total = output.size(0)
    return correct / total


class CrossEntropyLoss2D(torch.nn.CrossEntropyLoss):
    """
    See official documentation of torch.nn.CrossEntropyLoss

    The only difference is that this loss accepts optional normalization
    coefficient to deal with vanishing/exploding gradients
    """

    def __init__(self, normalization_coef=1, **kwargs):
        self.norm_coef = normalization_coef
        super(CrossEntropyLoss2D, self).__init__(**kwargs)

    def forward(self, prediction, target):
        """
        Args:
            prediction(:obj:`torch.Tensor`) - predicted values of the form:
                (number of images, number of classes, height, width)
            target(:obj:`torch.Tensor`) - expected dense classification of
                one of the two forms:
                (number of images, 1, height, width) or
                (number of images, height, width)
        return:
            Total loss for all of the images in the batch
        """
        if len(target.size()) == 4:
            target = target.squeeze(1)

        res = super(CrossEntropyLoss2D, self).forward(prediction, target)
        return res if self.norm_coef == 1 else res * self.norm_coef

def IntersectionOverUnion(prediction, mask, exclude_classes=[]):
    """Vectorized Computation of Intersections over Unions
        
    For each class, calculate IoU =
    true positive / (true positives + false positives + false negatives)

    Example:
    If there are two images with two classes (0 - background, 1 - object),
    exclude_classes = [].
    
    iou_class_0 = [(IoU(image = 1, background) + IoU(image = 2, background)] / 2
    iou_class_1 = [(IoU(image = 1, object) + IoU(image = 2, object)] / 2

    Note, if an image has no objects of a given class:

    IoU(image, class) = 
        0 - if some objects were detected 
        1 - if no objects were detected

    Args:
        prediction(:obj:`array-like`): i.e. tensor, list, or 
            numpy array. Expected Shape: (IMAGES_NUM x Classes x H x W)
        mask(:ob:`torch.Tensor`): Expected classes for each of 
            the elements in `prediction`. Only allowed values 
            are matching indices of the Classes of `prediction`: 
            mask values in [0; Classes - 1].
            Expected shape: (IMAGES_NUM x H x W).
        exclude_bg_idx(list, optional): index of Classes to 
            exclude in calculation (i.e. of background)
    
    Returns:
        (mean IoU - over classes, [iou_class_1, ..., iou_class_n]) -
            with None values for the excluded classes

    """
    
    # shape (IMAGES_NUM x CLASSES x HEIGHT x WIDTH)
    num_images, n_classes, h, w = prediction.shape

    # return indices of a class with the max value
    # assumption the mask is encoded with indices of layers
    # output: (N x HEIGHT x WIDTH)
    mask = mask.astype(float)
    prediction = np.argmax(prediction, axis = 1).astype(float)

    ious = []

    for cls in range(n_classes):
        if cls in exclude_classes:
            ious.append(None)
            continue

        a = prediction == cls
        b = mask == cls
        # get intersections and unions per each image
        # output: (N,)
        intersections = np.sum(a & b, dtype=np.float, axis=(1,2))
        unions = np.sum(a | b, dtype=np.float, axis=(1,2))

        # indices of images expecting objects of this class
        indcs_expctng_cls = np.sum(b, axis=(1,2), dtype=bool) # (N,)
        # indices of images with detected objects of this class
        indcs_dtctd_cls = np.sum(a, axis=(1,2), dtype=bool) # (N,)

        ious_per_img = np.full((num_images,), None)  

        # 0 - if some objects were detected though none expected
        ious_per_img[indcs_dtctd_cls != indcs_expctng_cls] = 0

        # 1 - if some objects were neither detected nor expected
        ious_per_img[indcs_dtctd_cls == indcs_expctng_cls] = 1

        # union is never 0
        ious_per_img[indcs_expctng_cls] = \
            intersections[indcs_expctng_cls] / unions[indcs_expctng_cls]

        iou = np.sum(ious_per_img, dtype=float) / num_images
        ious.append(iou)

    calculated_ious_indices = ious != None
    calculated_ious_num = np.sum(calculated_ious_indices)
    calculated_ious = ious[calculated_ious_indices]

    mean_iou = np.sum(calculated_ious) / calculated_ious_num

    return (mean_iou, ious)


# output: a dictionary {"ships": {"imageId": list of segmentation pixels}， “w/ships”: ["imageId"]}
def build_images_dict(csv_fn):
    """
    build dictionaries of images with/without ships

    Args:
    csv_fn: file name of the segmentation csv file
    """

    data = pd.read_csv(csv_fn)
    no_ships_list = []
    ship_dict = collections.defaultdict(list)  # {"imageId": "segmentation pixels"}
    for index, row in data.iterrows():
        image_id = row["ImageId"]
        encoded_pixels = row["EncodedPixels"]
        if pd.isnull(data.loc[index, "EncodedPixels"]) :
            no_ships_list.append(image_id)
        else:
            ship_dict[image_id].append(encoded_pixels)
    image_dict = dict({"ships": ship_dict, "w/ships": no_ships_list})
    return(image_dict)

# preprocess the image dict to have same images with/without ships
def process_images_dict(image_dict):
    n = len(image_dict["ships"])
    image_dict["w/ships"] = random.choices(image_dict["w/ships"], k=n)
    return image_dict

def split_images_dict(image_dict, train_portion):
    train_img_dict = {"ships": [], "w/ships": []}
    val_img_dict = {"ships": [], "w/ships": []}

    no_ships_len = len(image_dict["w/ships"])
    ships_items = list(image_dict["ships"].items())
    with_ships_len = len(ships_items)

    no_ships_split_boundary_idx = int((no_ships_len - 1) * train_portion)
    with_ships_split_boundary_idx = int((with_ships_len - 1) * train_portion)

    train_img_dict["w/ships"] = image_dict["w/ships"][:no_ships_split_boundary_idx]
    val_img_dict["w/ships"] = image_dict["w/ships"][no_ships_split_boundary_idx:]

    train_img_dict["ships"] = dict(ships_items[:with_ships_split_boundary_idx])
    val_img_dict["ships"] = dict(ships_items[with_ships_split_boundary_idx:])

    return train_img_dict, val_img_dict

def find_per_channel_std_mean(dtloader, device):
    means, means_of_sq, batches_count = [], [], []
    print("Find each channel's std and mean")
    total_iterations = len(dtloader)
    for i, (imgs, _) in enumerate(dtloader):
        print("{i}/{t}".format(i=i + 1, t=total_iterations))
        imgs = imgs.to(device)
        imgs_sq = imgs.pow(2)
        mean = imgs.mean(dim=(0,2,3))
        mean_of_sq = imgs_sq.mean(dim=(0,2,3))
        means.append(mean)
        means_of_sq.append(mean_of_sq)
        batches = imgs.size()[0]
        batches_count.append(torch.Tensor([batches, batches, batches]).to(device))

    means = torch.stack(means, dim=0)
    means_of_sq = torch.stack(means_of_sq, dim = 0)
    batches_count = torch.stack(batches_count, dim = 0)
    total_batches_count = torch.sum(batches_count, dim = 0)

    combined = torch.stack([means, means_of_sq, batches_count], dim=0)
    avg_means_of_sq = torch.mul(combined[1, :, :], combined[2, :, :]).sum(dim=0).div(total_batches_count)
    avg_means = torch.mul(combined[0, :, :], combined[2, :, :]).sum(dim=0).div(total_batches_count)
    avg_means_squared = avg_means.pow(2)

    final_mean = avg_means
    final_std = torch.sqrt(avg_means_of_sq - avg_means_squared)
    return final_std.cpu().numpy().tolist(), final_mean.cpu().numpy().tolist()

def split_find_metrics_and_backup_images(\
    csv_path, train_val_test_path, all_path,
    train_images_path, val_images_path, test_images_path,
    first_split_ratio = .05, device=None, num_workers = 2, batch_size = 64):
    all_images = build_images_dict(csv_path)
    all_images = process_images_dict(all_images) # 42556*2 images

    # for first_split_ratio = 0.5 train_val_test_images = 2127 * 2 images
    train_val_test_images, _ = split_images_dict(all_images, first_split_ratio)
    # train_val_image_ids = 1807*2 images, test_image_ids = 320*2 images
    train_val_images, test_images = split_images_dict(train_val_test_images, .85)
    # train images = 1535*2 images; val images = 272*2 images
    train_images, val_images = split_images_dict(train_val_images, .85)

    train_val_test_images = {}

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts = {}
    if num_workers:
        opts["num_workers"] = num_workers
    opts["batch_size"] = batch_size

    trn_dtset, val_dtset, test_dtst = \
        [AirbusDataset(imgs_opt, path, "classification", transforms.ToTensor()) \
            for imgs_opt, path in \
            zip([train_images, val_images, test_images],
            [train_images_path, val_images_path, test_images_path])]

    ldr = DataLoader(dataset = trn_dtset, **opts)
    std, mean = find_per_channel_std_mean(ldr, device)

    train_val_test_images["train"] = {
        "images": train_images,
        "mean": mean,
        "std": std
    }

    ldr = DataLoader(dataset = val_dtset, **opts)
    std, mean = find_per_channel_std_mean(ldr, device)

    train_val_test_images["val"] = {
        "images": val_images,
        "mean": mean,
        "std": std
    }

    ldr = DataLoader(dataset = test_dtst, **opts)
    std, mean = find_per_channel_std_mean(ldr, device)

    train_val_test_images["test"] = {
        "images": test_images,
        "mean": mean,
        "std": std
    }

    torch.save(train_val_test_images, train_val_test_path)
    torch.save(all_images, all_path)
