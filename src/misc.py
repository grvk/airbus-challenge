import collections
import random
import pandas as pd
from torch import argmax, nn

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


class CrossEntropyLoss2D(nn.CrossEntropyLoss):
    """
    See official documentation of nn.CrossEntropyLoss

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
