import os
import PIL
import torch
import random
import collections
import numpy as np
import pandas as pd
from helper import *
from torch.utils import data as D
from rle_handler import RLEHandler




class AirbusDataset(D.Dataset):
    """Data loader for airbus dataset."""
    
    def __init__(self, image_dict, path_to_images, loader_type, replacement=True, transform=None):
        """
        Args:
        image_dict: (dict) a dict of the following format:
                        {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}
        path_to_images: (string) the path to images  
        loader_type: (string) "classification" or "segmentation"
        replacement: (bool) sample with/without replacement
        transform: (callable, optional) Optional transform to be applied on a sample.
        
        Attributes:
        image_filenames:a list containing all the names of the images
        """
        self.image_dict = image_dict
        self.path_to_images = path_to_images
        self.loader_type = loader_type
        self.replacement = replacement
        self.image_filenames = list(image_dict["ships"].keys()) + image_dict["w/ships"]
        self.transform = transform
        self.resize_transform = None
        random.shuffle(self.image_filenames) 
    
    def __len__(self):
        return len(self.image_filenames)

    def retrive_image_matrix(self, image_id):
        """
        Get the image in shape of (768, 768, 3) using image_id
        
        Args:
        image_id: (string) image file name
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


        image_id = self.image_filenames[idx]
        no_ship_list = image_dict["w/ships"]
        image_mat = self.retrive_image_matrix(image_id) # (768, 768, 3)
        image_mat = np.reshape(image_mat, (3, 768, 768))
        
        
        # dataloader for classification
        # [1, 0] means it belongs to the first category (no ships)
        # [0, 1] means it belongs to the second category (ships present)
        if self.loader_type == "classification":
            if image_id in no_ship_list:
                sample = (image_mat, [1, 0])
            else:
                sample = (image_mat, [0, 1])
        
        # dataloader for segmentation 
        else: 
            if image_id in no_ship_list:
                mask = np.zeros((768, 768), dtype = np.int16)
            else:
                mask_list = self.image_dict["ships"][image_id]
                rle_handler = RLEHandler(mask_list)
                mask = rle_handler.masks_as_image() 
            mask = np.reshape(mask, (1, 768, 768))
            sample = (image_mat, mask)

        
        # remove image id if sample without replacement
        if self.replacement:
            self.image_filenames.remove(image_id)

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __random_sampler__(self):
        n_image_list = len(self.image_filenames)
        
        # randomly choose a idx
        random_idx = random.randint(0,n_image_list)
        sample = self.__getitem__(random_idx)
        return sample