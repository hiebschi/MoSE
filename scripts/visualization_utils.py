"""
Visualization utils
---------------------------
Helper functions used for the visual evaluation and depiction of the results 
of the segmentation model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import torch

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
# Plot mask of class-index-format

def plot_mask_idxformat(mask_idxformat, mask_name, reversed_codes, custom_colors):

    """
    Visualizes a ground truth mask.
    
    Args:
        mask_idxformat (numpy.ndarray): ground truth mask in class-index format.
        mask_name (str): name of the mask.
        reversed_codes (list): list of class names corresponding to class indices.
        custom_colors (list): list of color definitions for each class.
    
    Returns: 
        Plot of one mask.
    """

    # ensure that the number of colors matches the classes
    assert len(custom_colors) >= len(reversed_codes), "not enough colors!"

    # create colormap
    cmap = mcolors.ListedColormap(custom_colors[:len(reversed_codes)])

    # plot mask with colormap
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_idxformat, cmap=cmap, vmin=0, vmax=len(reversed_codes) - 1, interpolation='nearest') # deactivate interpolation
    plt.title(f"Patch: {mask_name}")
    plt.axis("off")

    # legend with colors and class names
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=custom_colors[i], markersize=10, linestyle="None", label=reversed_codes[i])
        for i in range(len(reversed_codes))
    ]
    plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()



###############################################################
# visualize RGB patch, true mask and predicted mask

def visualize_prediction(patch_name, test_loader, model, device, reversed_codes, custom_colors):
    """
    Visualizes the RGB image, ground truth mask, and predicted mask from a batch.
    
    Args:
        patch_name (str): name of the desired patch.
        test_loader (DataLoader): PyTorch DataLoader containing test images and masks.
        model (torch.nn.Module): trained model for prediction.
        device (torch.device): device to perform computations on.
        reversed_codes (list): list of class names corresponding to class indices.
        custom_colors (list): list of color definitions for each class.
    """
    
    found = False 
    
    # search for batch containing desired patch
    for i, (names, images, masks) in enumerate(test_loader):
        if patch_name in names:  # check if patch is contained
            patch_idx = names.index(patch_name)  # index of the patch in the batch

            # load data to device
            images, masks = images.to(device), masks.to(device)

            # convert and normalize rgb image
            t_image = images[patch_idx].permute(1, 2, 0).cpu().numpy()
            t_image = (t_image - t_image.min()) / (t_image.max() - t_image.min())  

            # convert mask into class-index format
            t_true_mask = masks[patch_idx].cpu().numpy().argmax(axis=0)

            # create prediction
            t_logits = model(images[patch_idx].unsqueeze(dim=0))
            t_pred_mask = torch.round(torch.sigmoid(t_logits)).cpu().detach().squeeze().numpy().argmax(axis=0)
            # print(t_pred_mask.shape)
            # print(np.unique(t_pred_mask))

            found = True
            break  # stop the search when the correct batch has been found

    if not found:
        print(f"Patch '{patch_name}' has not been found!")
        return
    
    # normalize RGB image
    t_image = (t_image - t_image.min()) / (t_image.max() - t_image.min())
    
    cmap = mcolors.ListedColormap(custom_colors[:len(reversed_codes)])
    
    # plot images and masks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(t_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(t_true_mask, cmap=cmap, vmin=0, vmax=len(reversed_codes) - 1, interpolation='nearest')
    axes[1].set_title("True Mask")
    axes[1].axis("off")
    
    axes[2].imshow(t_pred_mask, cmap=cmap, vmin=0, vmax=len(reversed_codes) - 1, interpolation='nearest')
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    # add legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=custom_colors[i], markersize=10, linestyle="None", label=reversed_codes[i])
        for i in range(len(reversed_codes))
    ]
    axes[2].legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()



###################################################
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



