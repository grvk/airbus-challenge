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