# build dictionaries of images with/without ships
# input: file name of the segmentation csv file
# output: a dictionary {"ships": {"imageId": "segmentation pixels"}， “w/ships”: ["imageId"]}

def build_images_dict(csv_fn):
    data = pd.read_csv(csv_fn)
    no_ships_list = []
    ship_dict = {}  # {"imageId": "segmentation pixels"}
    for index, row in data.iterrows():
        image_id = row['ImageId']
        encoded_pixels = row['EncodedPixels']
        if pd.isnull(data.loc[index, 'EncodedPixels']) :
            no_ships_list.append(image_id)
        else:
            ship_dict[image_id] = encoded_pixels
    
    image_dict = dict({"ships": ship_dict, "w/ships": no_ships_list})
    return(image_dict)


# test the function
seg_fn = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv"
image_dict = build_images_dict(seg_fn)
image_to_pix_dict = image_dict['ships']
no_ships_images = image_dict['w/ships']
