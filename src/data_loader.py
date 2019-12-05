import os
import torch
import random
import collections
import numpy as np
import pandas as pd



class ImageSegmentaionDataset():
    """Image Segmentation dataset"""
    
    def __init__(self, image_dict, loader_type, replacement=True, transform=None):
        
        """
        Args:
        image_dict (dict): a dict of the following format:
                        {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}
        loader_type: (string)"classification" or "segmentation"
        replacement: (bool) sample with/without replacement
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.image_dict = image_dict
        self.loader_type = loader_type
        self.replacement = replacement
        # images_list is the name 
        self.images_list = list(image_dict["ships"].keys()) + image_dict["w/ships"]
        random.shuffle(self.images_list) 
        self.transform = transform
    
    def __len__(self):
        return len(self.images_list)
    
    def rle_decode(self, mask_rle, shape=(768, 768)):
        """
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction
    
    def masks_as_image(self, in_mask_list):
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768), dtype = np.int16)
        #if isinstance(in_mask_list, list):
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks += self.rle_decode(mask)
        return np.expand_dims(all_masks, -1)

    def __getitem__(self, idx):
        image_id = self.images_list[idx]
        no_ship_list = image_dict["w/ships"]
        # dataloader for classification
        if self.loader_type == "classification":
            if image_id in no_ship_list:
                sample = {"image_id": image_id, "class": 0}
            else:
                sample = {"image_id": image_id, "class": 1}
        
        # dataloader for segmentation 
        else: 
            mask_list = image_dict["ships"][image_id]
            mask = self.masks_as_image(mask_list) 
            sample = {"image_id": image_id, "mask": mask}

        
        # remove image id if sample without replacement
        if self.replacement:
            self.images_list.remove(image_id)

         
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __random_sampler__(self):
        n_image_list = len(self.images_list)
        
        # randomly choose a idx
        random_idx = random.randint(0,n_image_list)
        sample = self.__getitem__(random_idx)
        return sample


# helper functions

# input: file name of the segmentation csv file
# output: a dictionary {"ships": {"imageId": list of segmentation pixels}， “w/ships”: ["imageId"]}
def build_images_dict(csv_fn):
    """build dictionaries of images with/without ships"""
    
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


# test the function
seg_fn = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv"
image_dict = build_images_dict(seg_fn) 
image_dict = process_images_dict(image_dict)

# dataloader for classfication 
dataloader_c = ImageSegmentaionDataset(image_dict, "classification")   
# dataloader for segmentation
dataloader_s = ImageSegmentaionDataset(image_dict, "segmentation") 

# {'image_id': '5a363988f.jpg', 'mask': array([[[0],.....,[0]]], dtype=int16)}
# mask is with shape of 768 * 768
sample_s = dataloader_s.__random_sampler__()


# {'image_id': 'fab214049.jpg', 'class': 0}
sample_c = dataloader_c.__random_sampler__()
