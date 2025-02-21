"""
Data utils
---------------------------
Helper functions used for loading and handling data.
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


# helper-function to load a  .npz file and extract the first array
# function for undoing the .npz-compression! unzip!

def unzip_npz_patch(patch_npz_name, patches_npz_dir, patches_unzipped_dir):
    """
    Loads a .npz patch file and extracts the first contained array, 
    then saves it as an uncompressed .npy file in the output directory.

    Args:
        patch_npz_name (str): Name of the .npz patch file.
        patches_npz_dir (str): Directory where the .npz files are stored.
        patches_unzipped_dir (str): Target directory where the unzipped .npy files will be saved.

    Returns:
        tuple: (patch_name, output_path) if successful, otherwise None.
    """
    import os
    import numpy as np

    # Construct the path to the .npz file
    patch_npz_path = os.path.join(patches_npz_dir, patch_npz_name)

    try:
        # Load the .npz file using memory mapping for more efficient loading
        with np.load(patch_npz_path, mmap_mode='r') as data:
            # Retrieve all keys (array names) from the .npz file
            array_keys = list(data.keys())
            if len(array_keys) > 1:
                print(f".npz file '{patch_npz_name}' contains {len(array_keys)} arrays: {array_keys}")

            # Remove the '.npz' extension to obtain the base patch name
            patch_name = patch_npz_name.replace(".npz", "")
            # Extract the first array from the .npz file
            patch_image = data[array_keys[0]]

        # Define the output path for the uncompressed .npy file
        output_path = os.path.join(patches_unzipped_dir, patch_name)
        # Save the patch image as an uncompressed .npy file
        np.save(output_path, patch_image)
        return patch_name, output_path

    except Exception as e:
        print(f"Error loading {patch_npz_name}: {e}")
        return None


# Dataset

class PatchDataset(Dataset):
    def __init__(self, patches_list, patches_dir, masks_dir=None, transform=None, filter_class=None): 
        # initializes the dataset by saving list of .npy-patches, the directory of the .npy-patches and the masks 
        # and performs optional transformations or filters one specific class

        """
        Custom Dataset for loading .npy patches and optional masks.
        Args:
            patches_list (list): List of the patch .npy-files.
            patches_dir (str): Directory containing patch .npy-files.
            masks_dir (str): Directory containing mask.npy-files (optional).
            transform (callable, optional): Transformation to be applied to the data.
            filter_class (int): Class to filter masks by (optional)
        """

        self.patches_list = patches_list
        self.patches_dir = patches_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filter_class = filter_class

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

        # Load .npy-patch dynamically
        patch_name = self.patches_list[idx]
        patch_path = os.path.join(self.patches_dir, patch_name)


        # Debugging: EOFError 
        try:
            patch = np.load(patch_path, allow_pickle=True)
        except EOFError:
            print(f"EOFError: file {patch_path} is truncated or corrupt. Skipping this file.")
            return None

        # Convert patch into Tensor and change dtype to float32
        patch = torch.tensor(patch, dtype=torch.float32)

        # Load the mask if available
        if self.masks_dir:   
            mask_path = os.path.join(self.masks_dir, patch_name.replace(".npy", "_mask.npy"))
            
            if os.path.exists(mask_path):   
                mask = np.load(mask_path) # load mask
                mask = torch.tensor(mask, dtype=torch.float32) # convert mask into Tensor and change datatype to float32

                if self.filter_class is not None:
                    # convert one-hot mask into label-map
                    label_map = torch.argmax(mask, dim=0)  # Shape: (H, W)
    
                    # filter: keep only background class (class 0) and desired class (e.g. woody debris, class 1)
                    # all pixels, which match neither 0 nor self.filter_class, set to 0.
                    label_map[(label_map != 0) & (label_map != self.filter_class)] = 0
    
                    # convert back in one-hot encoded masks:
                    mask = torch.nn.functional.one_hot(label_map.long(), num_classes=configs_sc.HYPERPARAMETERS["num_classes"])
                    mask = mask.permute(2, 0, 1).float()  # Shape: (num_classes, H, W)
                    # print(mask)

            else:
                print("WARNING: Mask does not exist!")
        else:
            print("WARNING: Directory does not exist!")

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
# Data exploration: How many pixels per class?

def pixel_distribution_dataloader(data_loader, num_classes, device, showprint=True):
    """
    Calculates the average pixel distribution per class over an entire DataLoader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        num_classes (int): Total number of classes.
        device (torch.device): Device to which the tensors should be moved.
        showprint (bool): Whether to print the results. Defaults to True.

    Returns:
        dict: Dictionary containing the average percentage of pixels per class.
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

    # Calculate percentage for each class
    results = {}
    for cls in range(num_classes):
        results[f"Class {cls + 1}"] = class_pixel_counts[cls].item() / total_pixels * 100

    # Optional: Print results
    if showprint:
        print("Pixel Distribution (%):")
        for class_name, percentage in results.items():
            print(f"{class_name}: {percentage:.2f}%")
    
    return results
