"""
Data utils
---------------------------
Helper functions used for loading and handling data.

If a function gets defined once and could be used over and over, it'll go in here.
"""

# import required packages

import os

import numpy as np

from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset


# import configs.py-file
import importlib
from scripts import data_utils
importlib.reload(data_utils) # reload changes


###################################
# Delete background patches

# Funktion, um Patches ohne Masken (Background) zu löschen
def delete_background_patches(data_list, patches_dir, masks_dir):
    background_patches = [f for f in data_list if not data_utils.has_mask(f, masks_dir)]
    
    print(f"Deleting {len(background_patches)} background patches...")
    for patch in background_patches:
        patch_path = os.path.join(patches_dir, patch)  # Erstelle den vollständigen Pfad zur Datei
        if os.path.exists(patch_path):
            os.remove(patch_path)  # Datei löschen
            # print(f"Deleted: {patch_path}")
        else:
           print(f"File not found (skipped): {patch_path}")
