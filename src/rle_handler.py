import numpy as np

class RLEHandler(): 
    """
    Take the individual ship masks and 
    create a single mask array for all ships
    
    """
    def __init__(self, encoded_masks):
        self.encoded_masks = encoded_masks

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
    
    def masks_as_image(self):
        """ Create a single mask array for all ships. """
        all_masks = np.zeros((768, 768), dtype = np.int16)
        
        for mask in self.encoded_masks:
            if isinstance(mask, str):
                all_masks += self.rle_decode(mask)
        return np.expand_dims(all_masks, -1)