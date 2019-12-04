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



class ImageSegmentaionDataset(Dataset):
    """Image Segmentation dataset"""

    def __init__(self, image_dict, loader_type, transform=None):
    """
    Args:
        image_dict (dict): a dict of the following format:
                        {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}
        loader_type: "classification" or "segmentation"
        transform (callable, optional): Optional transform to be applied on a sample.
    """
        self.image_dict = image_dict
        # images_list is the name 
        self.images_list = image_dict["ships"].keys() + image_dict["w/ships"]
        random.shuffle(self.images_list) 
        self.transform = transform
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        if self.loader_type == "classification":
            
        else:
            
        if self.transform:
            sample = self.transform(sample)

        return sample



dataset = ImageSegmentaionDataset(csv_file='/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv',
                                  root_dir='/kaggle/input/airbus-ship-detection/train_v2')
