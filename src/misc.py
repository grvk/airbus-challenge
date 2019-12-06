import collections
import random
import pandas as pd
import numpy as np
from torch import Tensor, nn, argmax

class CrossEntropyLossModified(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyLossModified, self).__init__(**kwargs)

    def forward(self, input, target):
        return super(CrossEntropyLossModified, self).forward( \
            input, target.squeeze(1))

def BasicIoU(output, target):
    """
    Intersection of Unions (works on binary masks only)

    Args:
        output(:obj:`torch.Tensor`) - predicted segmentation mask
            of shape BATCHES x CHANNELS(always=1) x HEIGHT x WIDTH
        target(:obj:`torch.Tensor`) - expected segmentation mask
            of shape BATCHES x CHANNELS(always=1) x HEIGHT x WIDTH

    Returns:
        IoU = Area of Intersection /
            (output area + input area - area of interscection)
            averaged over batches
    """

    batches_num, _, _, _ = output.size()

    output = output == 1 # convert to bool
    target = target == 1 # convert to bool

    intersctn = torch.sum(output == target, (1,2,3))
    output_area = torch.sum(output, (1,2,3))
    target_area = torch.sum(target, (1,2,3))

    iou_sum = 0
    iou_count = 0
    for i in list(range(batches_num)):
        if (intersctn[0] == 0) and (target_area != 0) or (intersctn[i] != 0):
            ios_sum = intersctn / (output_area + target_area - intersctn)
            iou_count += 1

    return iou_sum / iou_count if iou_count != 0 else 0

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
    predictions = argmax(output.data, 1) # indices of the predicted clases
    correct = (predictions == target).sum().item()
    total = output.size(0)
    return correct / total

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

    train_img_dict["w/ships"], val_img_dict["w/ships"] = \
        np.split(image_dict["w/ships"], (no_ships_split_boundary_idx,))

    train_img_dict["ships"] = dict(ships_items[:with_ships_split_boundary_idx])
    val_img_dict["ships"] = dict(ships_items[with_ships_split_boundary_idx:])

    return train_img_dict, val_img_dict
