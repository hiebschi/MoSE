"""
Data utils
---------------------------
Helper functions used for loading and handling data.

If a function gets defined once and could be used over and over, it'll go in here.
"""

# import required packages

import os

# import numpy as np
# import geopandas as gpd

# import sklearn

# import matplotlib.pyplot as plt

# import torch
# import torchvision
# from torch import nn


def extract_section_and_id(file_name):
    """
    Extracts section and patch_id from the file_name of masks and of preprocessed and compressed patches.
    
    Args:
        file_name (str): file name of an preprocessd patch (.npy.npz) or an mask (_mask.npy)
    
    """
    
    parts = file_name.split("_") # split condition: _
    section = parts[0]  # extract section from file_name, e.g. "A01"
    patch_id = parts[2].replace(".npy.npz", "").replace("_mask", "") #  extract patch_id, e.g. 481
    
    return section, patch_id


def has_mask(patch_name, masks_dir):
    """ 
    Check if a patch has a corresponding mask in the masks directory.

    Args:
        patch_name (str): Name of the patch (e.g., 'A01_patch_481.npy.npz').
        masks_dir (str): Directory where masks are stored.
    """

    mask_path = os.path.join(masks_dir, patch_name.replace(".npy.npz", "_mask.npy")) # load corresponding mask path
    
    return os.path.exists(mask_path) # check if path exists


