import os
import PIL
import torch
import random
import collections
import numpy as np
import pandas as pd
from torch.utils import data as D
from .rle_handler import RLEHandler


class AirbusDataset(D.Dataset):
    """Airbus dataset."""

    def __init__(self, image_dict, path_to_images, loader_type,
                src_transform=None, target_transform=None, replacement=False):
        """
        Args:
            image_dict(dict): a dict of the following format:
                {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}
            path_to_images(string): the path to images
            loader_type(string): "classification" or "segmentation"
            replacement(bool, optional): sample with/without replacement
            src_transform(callable, optional): transforms to apply to input images
            target_transform(callable, optional): works only with segmentation
                dataset. Transforms to apply to segmentation images

        Attributes:
            image_filenames: list containing all the names of the images
        """
        self.image_dict = image_dict
        self.path_to_images = path_to_images
        self.loader_type = loader_type
        self.replacement = replacement
        self.image_filenames = np.sort(list(image_dict["ships"].keys()) + \
            image_dict["w/ships"]).tolist()

        self.images_num = len(self.image_filenames)
        self.src_transform = src_transform
        self.target_transform = target_transform
        self.resize_transform = None

    def __len__(self):
        return self.images_num

    def retrive_image_matrix(self, image_id):
        """
        Get the image in shape of (768, 768, 3) using image_id

        Args:
            image_id(string): image file name
        """

        image_path = os.path.join(self.path_to_images, image_id)
        img = PIL.Image.open(image_path)
        image_mat = np.array(img) # (768, 768, 3)
        return image_mat

    def __getitem__(self, idx):
        """
        get a data point for segmentation/classfication based on the 'idx'

        output for classification:
        image with ship: (<3x768x768 tenser of 0-255 values>, [0,1])
        image without ship: (<3x768x768 tenser of 0-255 values>, [1,0])

        output for segmentation:
        image with ship: (<3x768x768 tenser of 0-255 values>, <1x768x768 tenser of 0's and 1's>)
        image without ship: (<3x768x768 tenser of 0-255 values>, <1x768x768 tenser of 0's>)

        """

        # if with replacement, then it's the same as randomly selecting an image
        if self.replacement:
            idx = random.randint(0, self.images_num)

        image_id = self.image_filenames[idx]
        no_ship_list = self.image_dict["w/ships"]
        image_mat = self.retrive_image_matrix(image_id) # (768, 768, 3)

        if self.src_transform:
            image_mat = self.src_transform(image_mat)


        # dataset for classification
        # [1, 0] means it belongs to the first category (no ships)
        # [0, 1] means it belongs to the second category (ships present)
        if self.loader_type == "classification":
            if image_id in no_ship_list:
                sample = (image_mat, [1, 0])
            else:
                sample = (image_mat, [0, 1])

        # dataset for segmentation
        else:
            if image_id in no_ship_list:
                mask = np.zeros((768, 768), dtype = np.int16)
            else:
                mask_list = self.image_dict["ships"][image_id]
                rle_handler = RLEHandler(mask_list)
                mask = rle_handler.masks_as_image()

            if self.target_transform:
                mask = self.target_transform(mask).to(torch.long)

            sample = (image_mat, mask)

        return sample
