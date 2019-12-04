import os
import torch
import random
import pandas as pd


# build dictionaries of images with/without ships
# input: file name of the segmentation csv file
# output: a dictionary {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}

def build_images_dict(csv_fn):
    data = pd.read_csv(csv_fn)
    no_ships_list = []
    ship_dict = {}  # {"imageId": "segmentation pixels"}
    for index, row in data.iterrows():
        image_id = row["ImageId"]
        encoded_pixels = row["EncodedPixels"]
        if pd.isnull(data.loc[index, "EncodedPixels"]) :
            no_ships_list.append(image_id)
        else:
            ship_dict[image_id] = encoded_pixels

    image_dict = dict({"ships": ship_dict, "w/ships": no_ships_list})
    return(image_dict)


# test the function
seg_fn = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv"
image_dict = build_images_dict(seg_fn)
image_to_pix_dict = image_dict["ships"]
no_ships_images = image_dict["w/ships"]


# preprocess the image dict to have same images with/without ships
def process_images_dict(image_dict):
    n = len(image_dict["ships"])
    image_dict["w/ships"] = random.choices(image_dict["w/ships"], k=n)
    return image_dict

# dict with same images with/without ships
# a dictionary {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}
images_dict = process_images_dict(image_dict)



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
            if image_id in no_ship_list:
                sample = {"image_id": image_id, "mask": None}
            else:
                sample = {"image_id": image_id, "mask": image_dict["ships"][image_id]}
                
        
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


# test ImageSegmentaionDataset dataloader
dataloader_c = ImageSegmentaionDataset(image_dict, "classification")   
dataloader_s = ImageSegmentaionDataset(image_dict, "segmentation")         

# get a random sample
dataloader_s.__random_sampler__() 

'''
{'image_id': 'dd1760612.jpg',
 'mask': '11149 3 11913 7 12677 12 13442 15 14210 15 14979 14 15747 15 16515 15 17283 15 18052 14 18820 15 19588 15 20356 15 21125 15 21893 15 22661 15 23430 14 24198 15 24966 15 25734 15 26503 14 27271 15 28039 15 28807 15 29576 15 30344 15 31112 15 31881 14 32649 15 33417 15 34185 15 34954 14 35722 15 36490 15 37258 12 38027 7 38795 3'}
'''

dataloader_c.__random_sampler__() 

#{'image_id': 'ed64caab6.jpg', 'class': 1}
