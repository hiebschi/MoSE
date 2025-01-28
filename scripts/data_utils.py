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
from configs import configs_sc
importlib.reload(configs_sc) # reload changes


def extract_section_and_id(file_name):
    """
    Extracts section and patch_id from the file_name of masks and of preprocessed and compressed patches.
    
    Args:
        file_name (str): file name of an preprocessed patch (.npy) or an mask (_mask.npy)
    
    """
    
    parts = file_name.split("_") # split condition: _
    section = parts[0]  # extract section from file_name, e.g. "A01"
    patch_id = parts[2].replace(".npy", "").replace("_mask", "") #  extract patch_id, e.g. 481
    
    return section, patch_id


def has_mask(patch_name, masks_dir):
    """ 
    Check if a patch has a corresponding mask in the masks directory.

    Args:
        patch_name (str): Name of the patch (e.g., 'A01_patch_481.npy').
        masks_dir (str): Directory where masks are stored.
    """

    mask_path = os.path.join(masks_dir, patch_name.replace(".npy", "_mask.npy")) # load corresponding mask path
    

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
    def __init__(self, patches_list, patches_dir, masks_dir=None, transform=None, preload = False): 
        # initializes the dataset by saving list of .npz-patches, the directory of the .npz-patches and the masks 
        # and optional transformations and preloads

        """
        Custom Dataset for loading .npy patches and optional masks.
        Args:
            patches_list (list): List of the patch .npy-files.
            patches_dir (str): Directory containing patch .npy-files.
            masks_dir (str): Directory containing mask.npy files (optional).
            transform (callable, optional): Transformation to be applied to the data.
            preload (bool): Whether to preload all patches into memory.
        """

        self.patches_list = patches_list
        self.patches_dir = patches_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preload = preload

        # ONLY FOR COMPRESSED .npy.NPZ-PATCHES (see commit: "!UNZIP AND CHANGE FUNCTIONS TO .NPY!")
        # if preload:
        #     # Parallel loading of patches using ThreadPoolExecutor
        #     print("Preloading patches...")
        #     self.preloaded_patches = parallel_load_npz(patches_npz_list, patches_npz_dir)
        # else:
        #     self.preloaded_patches = None

    def __len__(self):

        """
        Returns the number of .npy-patches in the dataset.
        """

        return len(self.patches_list) # returns the number of .npy-patches for the DataLoader

    def __getitem__(self, idx): # loads patch and corresponding mask

        """
        Returns the patch and its corresponding mask.

        Args:
            idx (int): Index of the patch in the dataset.
        Returns:
            tuple: A tuple containing the patch and its mask.
        """

        # if self.preload and self.preloaded_patches is not None:
        #     # Use preloaded patch
        #     patch_name, patch = self.preloaded_patches[idx] # save patch name and patch image data
        # else:
        #     # Load .npz-patch dynamically with patch loading function (see 4.1)
        #     patch_name, patch = load_npz_patch(self.patches_npz_list[idx], self.patches_npz_dir) # save patch name and patch image data

        # Load .npy-patch dynamically
        patch_name = self.patches_list[idx]
        patch_path = os.path.join(self.patches_dir, patch_name)
        patch = np.load(patch_path)

        # Convert patch into Tensor and change dtype to float32
        patch = torch.tensor(patch, dtype=torch.float32)

        # Load the mask if available
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, patch_name.replace(".npy", "_mask.npy"))
            if os.path.exists(mask_path):
                mask = np.load(mask_path) # load mask
                mask = torch.tensor(mask, dtype=torch.float32) # convert mask into Tensor and change datatype to float32
            else:
                mask = torch.zeros((configs_sc.HYPERPARAMETERS["num_classes"], patch.shape[1], patch.shape[2]), dtype=torch.float32)  # Create default background mask = all pixels in all channels (= classes) are zeros
        else:
            mask = torch.zeros((configs_sc.HYPERPARAMETERS["num_classes"], patch.shape[1], patch.shape[2]), dtype=torch.float32)  # Default background mask

        # Apply any transformations if needed
        if self.transform:
            patch, mask = self.transform(patch, mask)

        # Ensure mask has the correct number of channels
        if mask.shape[0] != configs_sc.HYPERPARAMETERS["num_classes"]:  # If mask doesn't have the right number of channels
          print("WARNING: NOT THE RIGHT NUMBER OF CHANNELS!")
          mask = mask.unsqueeze(0)  # Add a channel dimension to the beginning to make it (1, H, W)
          mask = mask.repeat(configs_sc.HYPERPARAMETERS["num_classes"], 1, 1) # Repeat this along the channel dimension to get the desired shape (NUM_CLASSES, H, W)

        return patch_name, patch, mask
    


##################
# Data exploration: How many pixels per class and how many are background pixels?

def average_pixel_distribution_dataloader(data_loader, num_classes, device, showprint=True):
    """
    Calculates the average pixel distribution per class (including background) over an entire DataLoader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        num_classes (int): Total number of classes (excluding background).
        device (torch.device): Device to which the tensors should be moved.
        showprint (bool): Whether to print the results. Defaults to True.

    Returns:
        dict: Dictionary containing the average percentage of pixels per class (including background).
    """
    # Initialize counters
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_pixels = 0

    # Iterate through the DataLoader
    for _, _, masks in data_loader:  # Assume DataLoader returns [names, images, masks]
        # Send masks to the specified device
        masks = masks.to(device)

        # Flatten spatial dimensions and sum pixel counts for each class
        batch_class_counts = masks.sum(dim=(0, 2, 3))  # Sum over batch, height, and width for each class
        class_pixel_counts += batch_class_counts

        # Add to total pixel count (batch_size * height * width)
        total_pixels += masks.numel() // masks.shape[1]  # Total pixels in batch

    # Calculate background pixels
    background_pixels = total_pixels - class_pixel_counts.sum().item()

    # Calculate percentage for each class
    results = {"Background": background_pixels / total_pixels * 100}
    for cls in range(num_classes):
        results[f"Class {cls + 1}"] = class_pixel_counts[cls].item() / total_pixels * 100

    # Optional: Print results
    if showprint:
        print("Average Pixel Distribution (%):")
        for class_name, percentage in results.items():
            print(f"{class_name}: {percentage:.2f}%")
    
    return results
