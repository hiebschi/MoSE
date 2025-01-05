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


# helper-function to load a single .npz file and extract the first array
# function for undoing the .npz-compression! unzip!

def load_npz_patch(patch_npz_name, patches_npz_dir):

    """ 
    Load patch .npz file and undo the .npz compression by extracting the first array and returning it as npy-array.

    Args:
        patch_npz_name (str): Name of the patch .npz file.
        patches_npz_dir (str): Directory where patch .npz files are stored.

    Returns:
        Unzipped patch image data as npy-array.
    """

    patch_npz_path = os.path.join(patches_npz_dir, patch_npz_name) # path to .npz-file

    try:
        with np.load(patch_npz_path, mmap_mode='r') as data:  # Load the patch .npz-file
        # use memory mapping (mmap_mode='r') for more efficient loading
        # (without loading the total content into RAM)

            array_keys = list(data.keys())  # Get all keys (array names)

            if len(array_keys) > 1:  # Print a warning if multiple arrays are present
                print(f".npz-file '{patch_npz_name}' contains {len(array_keys)} arrays: {array_keys}")

            patch_name = patch_npz_name.replace(".npz", "")  # Remove '.npz' to get the base name

            patch_image = data[array_keys[0]]  # Extract the first array


            return (patch_name, patch_image)

    except Exception as e:  # Handle any errors during file loading
        print(f"Error loading {patch_npz_name}: {e}")
        return None


def parallel_load_npz(patches_npz_list, patches_npz_dir):
    """
    For parallel loading and unzipping of the .npz-patches.
    """

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda fname: load_npz_patch(fname, patches_npz_dir), patches_npz_list))
    
    return [res for res in results if res is not None]  # Filter out failed loads



# Dataset

class PatchDataset(Dataset):
    def __init__(self, patches_npz_list, patches_npz_dir, masks_dir=None, transform=None, preload = False): 
        # initializes the dataset by saving list of .npz-patches, the directory of the .npz-patches and the masks 
        # and optional transformations and preloads

        """
        Custom Dataset for loading .npz patches and optional masks.
        Args:
            patches_npz_list (list): List of the patch .npz-files.
            patches_npz_dir (str): Directory containing patch .npz-files.
            masks_dir (str): Directory containing mask.npy files (optional).
            transform (callable, optional): Transformation to be applied to the data.
            preload (bool): Whether to preload all patches into memory.
        """

        self.patches_npz_list = patches_npz_list
        self.patches_npz_dir = patches_npz_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preload = preload

        if preload:
            # Parallel loading of patches using ThreadPoolExecutor
            print("Preloading patches...")
            self.preloaded_patches = parallel_load_npz(patches_npz_list, patches_npz_dir)
        else:
            self.preloaded_patches = None

    def __len__(self):

        """
        Returns the number of .npz-patches in the dataset.
        """

        return len(self.patches_npz_list) # returns the number of .npz-patches for the DataLoader

    def __getitem__(self, idx): # loads patch and corresponding mask

        """
        Returns the patch and its corresponding mask.

        Args:
            idx (int): Index of the patch in the dataset.
        Returns:
            tuple: A tuple containing the patch and its mask.
        """

        if self.preload and self.preloaded_patches is not None:
            # Use preloaded patch
            patch_name, patch = self.preloaded_patches[idx] # save patch name and patch image data
        else:
            # Load .npz-patch dynamically with patch loading function (see 4.1)
            patch_name, patch = load_npz_patch(self.patches_npz_list[idx], self.patches_npz_dir) # save patch name and patch image data

        # Convert patch into Tensor and change dtype to float32
        patch = torch.tensor(patch, dtype=torch.float32)

        # Load the mask if available
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, patch_name.replace(".npy", "_mask.npy"))
            if os.path.exists(mask_path):
                mask = np.load(mask_path) # load mask
                mask = torch.tensor(mask, dtype=torch.float32) # convert mask into Tensor and change datatype to float32
            else:
                mask = torch.zeros((9, patch.shape[1], patch.shape[2]), dtype=torch.float32)  # Create default background mask = all pixels in all channels (= classes) are zeros
        else:
            mask = torch.zeros((9, patch.shape[1], patch.shape[2]), dtype=torch.float32)  # Default background mask

        # Apply any transformations if needed
        if self.transform:
            patch, mask = self.transform(patch, mask)

        # Ensure mask has the correct number of channels
        if mask.shape[0] != 9:  # If mask doesn't have 9 channels
          print("WARNING: NOT 9 CHANNELS!")
          mask = mask.unsqueeze(0)  # Add a channel dimension to the beginning to make it (1, H, W)
          mask = mask.repeat(9, 1, 1) # Repeat this along the channel dimension 9 times to get the desired shape (9, H, W)

        return patch_name, patch, mask