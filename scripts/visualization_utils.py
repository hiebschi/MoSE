"""
Visualization utils
---------------------------
Helper functions used for the visual evaluation and depiction of the results 
of the segmentation model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

##########################################################
# Plot one RGB patch

def norm_plot_patch(patch, patch_name):
    """
    Normalizes and plots a single preprocessed patch.
    
    Args:
        patch (numpy.ndarray): preprocessed patch/ image data.
        patch_name (str): name of this patch.
    
    Returns: 
        Plot of normalized image.
    """

    # Normalization
    patch_normalized = patch - np.min(patch) # set minimum to 0
    patch_normalized = patch_normalized / np.max(patch_normalized)  # maximize to 1

    # Plot the preprocessed image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(patch_normalized.transpose(1, 2, 0))  # transpose for RGB depiction
    ax.set_title(f"Preprocessed Patch: {patch_name}")
    plt.show()


##################################
# Plot targets in






# plot class-wise loss for a specific epoch

def plot_classwise_loss(epoch_num, class_wise_losses_per_epoch, num_classes):
    """
    Plots a bar chart of the class-wise loss for a specific epoch.
    
    Args:
        epoch_num (int): Epoch number to visualize.
        class_wise_losses_per_epoch (list of numpy arrays): Class-wise losses per epoch.
        num_classes (int): Number of classes.
    """
    if epoch_num >= len(class_wise_losses_per_epoch):
        print(f"Epoch {epoch_num} is out of range! Max epoch: {len(class_wise_losses_per_epoch) - 1}")
        return

    class_losses = class_wise_losses_per_epoch[epoch_num]  # loss values of chosen epoch 
    
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(num_classes), class_losses, color="skyblue")
    plt.xlabel("Class Index")
    plt.ylabel("Loss")
    plt.title(f"Class-wise Loss for Epoch {epoch_num}")
    plt.xticks(np.arange(num_classes))
    plt.show()



