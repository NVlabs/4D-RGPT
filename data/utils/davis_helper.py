import glob
import numpy as np
from PIL import Image

# from data.data_helper import DATASETS

# DAVIS = DATASETS['davis']

def get_bboxes_from_mask(label_map: Image.Image) -> dict:
    """
    Given a HxW segmentation mask, return a dict of {label: (xmin, ymin, xmax, ymax)}.
    """
    bboxes = {}
    num_labels = np.array(label_map).max() + 1

    for label in range(num_labels):
        if label == 0:
            continue  # skip background (optional)
        
        ys, xs = np.where(label_map == label)
        if ys.size == 0 or xs.size == 0:
            continue
        
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        bboxes[label] = (xmin, ymin, xmax, ymax)
    
    return bboxes

