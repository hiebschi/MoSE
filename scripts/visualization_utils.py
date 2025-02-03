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

def plot_mask_idxformat(mask_idxformat, mask_name, reversed_codes):

    """
    Visualizes a ground truth mask.
    
    Args:
        mask_idxformat (numpy.ndarray): ground truth mask in class-index format.
        mask_name (str): name of the mask.
        reversed_codes (list): list of class names corresponding to class indices.
    
    Returns: 
        Plot of one mask.
    """

    # customized colors for each class
    custom_colors = [
        (0.12, 0.47, 0.61),  # 0: blue
        (0.84, 0.15, 0.16),  # 1: RED (not existent)
        (0.40, 0.34, 0.29),  # 2: darkbrown
        (0.84, 0.15, 0.16),  # 3: RED (not existent)
        (0.65, 0.44, 0.29),  # 4: brown
        (0.84, 0.15, 0.16),  # 5: RED (not existent)
        (0.94, 0.74, 0.13),  # 6: orange
        (0.84, 0.15, 0.16),  # 7: RED (not existent)
        (0.74, 0.74, 0.13),  # 8: olive
        (0.54, 0.74, 0.13),  # 9: green
    ]  

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

def visualize_prediction(batch_idx, patch_idx, test_loader, model, device, reversed_codes):
    """
    Visualizes the RGB image, ground truth mask, and predicted mask from a batch.
    
    Parameters:
        batch_idx (int): index of the batch to visualize.
        patch_idx (int): index of the patch within the batch.
        test_loader (DataLoader): PyTorch DataLoader containing test images and masks.
        model (torch.nn.Module): trained model for prediction.
        device (torch.device): device to perform computations on.
        reversed_codes (list): list of class names corresponding to class indices.
    """
    # retrieve batch
    for i, (names, images, masks) in enumerate(test_loader):
        if i == batch_idx:
            images, masks = images.to(device), masks.to(device)
            
            patch_name = names[patch_idx]
            print("Visualizing patch:", patch_name)
            
            # extract image and masks
            t_image = images[patch_idx].permute(1, 2, 0).cpu().numpy()
            t_true_mask = masks[patch_idx].cpu().numpy().argmax(axis=0)
            
            # model prediction
            with torch.no_grad():
                t_logits = model(images[patch_idx].unsqueeze(dim=0))
                t_pred_mask = torch.sigmoid(t_logits).squeeze().cpu().numpy().argmax(axis=0)
            
            break
    else:
        print("Batch index out of range!")
        return
    
    # normalize RGB image
    t_image = (t_image - t_image.min()) / (t_image.max() - t_image.min())
    
    # define custom colors
    custom_colors = [
        (0.12, 0.47, 0.61),  # 0: blue
        (0.84, 0.15, 0.16),  # 1: RED (not existent)
        (0.40, 0.34, 0.29),  # 2: darkbrown
        (0.84, 0.15, 0.16),  # 3: RED (not existent)
        (0.65, 0.44, 0.29),  # 4: brown
        (0.84, 0.15, 0.16),  # 5: RED (not existent)
        (0.94, 0.74, 0.13),  # 6: orange
        (0.84, 0.15, 0.16),  # 7: RED (not existent)
        (0.74, 0.74, 0.13),  # 8: olive
        (0.54, 0.74, 0.13),  # 9: green
    ]
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



