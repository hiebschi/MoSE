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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm

##########################################################
# Plot one RGB patch

def norm_plot_patch(patch, patch_name):
    """
    Normalizes and plots a single patch.
    
    Args:
        patch (numpy.ndarray): patch/ image data.
        patch_name (str): name of this patch.
    
    Returns: 
        Plot of normalized image.
    """

    # Normalization
    patch_normalized = patch - np.min(patch) # set minimum to 0
    patch_normalized = patch_normalized / np.max(patch_normalized)  # maximize to 1
    print(patch_name)

    # Plot the preprocessed image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(patch_normalized.transpose(1, 2, 0))  # transpose for RGB depiction
    ax.set_title(f"Original Image Patch")

    # ADD SCALE BAR
    # pixel_size_m is the physical size of one pixel in meters = 0.024 m
    pixel_size_m = 0.024  # 2.4 cm per pixel

    # desired physical length for the scale bar in meters 
    desired_length_m = 1.0

    # Convert the desired physical length to pixel length
    scale_bar_length_px = desired_length_m / pixel_size_m 

    # Create a FontProperties object for the scale bar label
    fontprops = fm.FontProperties(size=10)

    # Create the AnchoredSizeBar with the computed pixel length
    scalebar = AnchoredSizeBar(ax.transData,
                           scale_bar_length_px,      # Length of scale bar in pixels
                           f"{desired_length_m} m",   # Label for the scale bar
                           "lower right",             # Location in the plot
                           pad=0.1,                   # Padding between scale bar and plot edge
                           color="white",             # Color of the scale bar and text
                           frameon=False,             
                           size_vertical=2)           # Thickness of the scale bar

    # Add the scale bar to the axis
    ax.add_artist(scalebar)
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
    print(mask_name)

    # ensure that the number of colors matches the classes
    assert len(custom_colors) >= len(reversed_codes), "not enough colors!"

    # create colormap
    cmap = mcolors.ListedColormap(custom_colors[:len(reversed_codes)])

    # plot mask with colormap
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_idxformat, cmap=cmap, vmin=0, vmax=len(reversed_codes) - 1, interpolation='nearest') # deactivate interpolation
    plt.title(f"Ground Truth Mask")

    plt.axis("off")

    # legend with colors and class names
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=custom_colors[i], markersize=10, linestyle="None", label=reversed_codes[i])
        for i in range(len(reversed_codes))
    ]
    plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


#########################################################
# Plot RGB image and ground truth mask (no prediction)

def visualize_image_and_mask(patch, mask_idxformat, patch_name, reversed_codes, custom_colors, pixel_size_m=0.024, scalebar_length_m=1.0):
    """
    Plots the original image patch and the corresponding mask side by side.
    
    Args:
        patch (np.ndarray): RGB image patch with shape (3, H, W).
        mask_idxformat (np.ndarray): Mask in class-index format (H, W).
        patch_name (str): Name of the patch/mask.
        reversed_codes (list): List of class names.
        custom_colors (list): Colors for the classes.
        pixel_size_m (float): Physical pixel size in meters. Default is 0.024 m (2.4 cm).
        scalebar_length_m (float): Length of the scale bar in meters. Default is 1.0 m.
    """
    
    # Normalize patch
    patch_norm = patch - np.min(patch)
    patch_norm = patch_norm / np.max(patch_norm)

    # Colormap for mask
    cmap = mcolors.ListedColormap(custom_colors[:len(reversed_codes)])

    # Start plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot original image
    axes[0].imshow(patch_norm.transpose(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Add scale bar
    scalebar_px = scalebar_length_m / pixel_size_m
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(axes[0].transData,
                               scalebar_px,
                               f'{scalebar_length_m} m',
                               'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fontprops)
    axes[0].add_artist(scalebar)

    # Plot mask
    im = axes[1].imshow(mask_idxformat, cmap=cmap, vmin=0, vmax=len(reversed_codes) - 1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Add scale bar
    scalebar_px = scalebar_length_m / pixel_size_m
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(axes[1].transData,
                               scalebar_px,
                               f'{scalebar_length_m} m',
                               'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fontprops)
    axes[1].add_artist(scalebar)

    # Legend for mask
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=custom_colors[i], markersize=10, linestyle="None", label=reversed_codes[i])
        for i in range(len(reversed_codes))
    ]
    axes[1].legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(patch_name)
    plt.tight_layout()
    plt.show()



###################################################
# Plot class-wise loss for a specific epoch

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



#############################################
# Plots loss curves per class

def plot_loss_curves_per_class(class_wise_loss, dataset_type, num_classes, reversed_codes, custom_colors):
    """
    Plots the loss curves per epoch for each class.
    
    Args:
        class_wise_loss (list): List of arrays (or tuples), each containing per-class loss values for one epoch.
        dataset_type (str): "Train" or "Test", indicating which dataset is plotted.
        num_classes (int): Number of classes.
        reversed_codes (dict): Dictionary mapping class indices to class names.
        custom_colors (list): A list of RGB tuples for each class.
    """

    # Create a list of epoch numbers (1, 2, ..., number of epochs)
    epochs = list(range(1, len(class_wise_loss) + 1))

    # Create the plot
    plt.figure(figsize=(10, 5))
    
    for i in range(num_classes):
        
        # Extract the loss for class i over all epochs
        class_losses = [epoch_losses[i] for epoch_losses in class_wise_loss]
        
        # Use reversed_codes to get the class name (fallback: "class i")
        class_name = reversed_codes.get(i, f"Class {i}")
        plt.plot(epochs, class_losses, marker = 'o', linestyle='-', 
                 color=custom_colors[i],
                 label=f"Loss Class {i}: {class_name}")

    plt.xlabel("Epoch []") # Set the x-axis label
    plt.ylabel("Cross Entropy Loss []") # Set the y-axis label
    plt.title(f"{dataset_type} Loss per Class over Epochs") # Set the plot title
    plt.xlim(left=1) # Force x-axis to start at epoch 0
    plt.ylim(bottom=0) # Force y-axis to start at 0
    plt.legend() # Display the legend
    plt.grid(True) # Enable grid for better readability
    plt.show() # Display the plot
   









#########################################################
# Denormalizes an image
def denormalize(image, mean, std):
    """
    Reverse the normalization of an image.
    Assumes `image` is a NumPy array with shape [H, W, C] and pixel values after A.Normalize.
    """
    mean = np.array(mean)
    std = np.array(std)
    # Reverse normalization: multiply by std and add mean.
    image = image * std + mean
    return image

    


#########################################################
# Visualize RGB image, ground truth mask, and predicted mask

def visualize_prediction(patch_name, test_loader, model, device, reversed_codes, custom_colors, show=False):
    """
    Visualizes the RGB image, ground truth mask, and predicted mask for a given patch.
    
    Args:
        patch_name (str): Name of the desired patch.
        test_loader (DataLoader): PyTorch DataLoader containing test images and masks.
        model (torch.nn.Module): Trained model for prediction.
        device (torch.device): Device to perform computations on.
        reversed_codes (dict): Dictionary mapping class indices to class names.
        custom_colors (list): List of color definitions for each class.
        show (bool): If True, displays the plot immediately.
    
    Returns:
        fig (Figure): Matplotlib figure containing the visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    import numpy as np

    # Import the configuration to determine the number of classes
    from configs import configs_sc

    # Determine number of classes from hyperparameters (e.g. 2 or 5)
    n_classes = configs_sc.HYPERPARAMETERS["num_classes"]
    
    # -------------------------------------------------------------------
    # 1. Search for the Patch in the Test DataLoader
    # -------------------------------------------------------------------
    found = False
    for i, (names, images, masks) in enumerate(test_loader):
        if patch_name in names:
            patch_idx = names.index(patch_name)
            # Move data to device
            images, masks = images.to(device), masks.to(device)
            
            # Extract and convert the RGB image (from CHW to HWC)
            t_image = images[patch_idx].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize the image using ImageNet mean and std
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t_image = denormalize(t_image, mean, std)
            # Normalize to [0,1] for display
            t_image = (t_image - t_image.min()) / (t_image.max() - t_image.min())
            
            # Convert one-hot encoded mask to class-index mask
            t_true_mask = masks[patch_idx].cpu().numpy().argmax(axis=0)
            
            # Create prediction for the patch
            t_logits = model(images[patch_idx].unsqueeze(dim=0))
            t_pred_mask = torch.round(torch.sigmoid(t_logits)).cpu().detach().squeeze().numpy().argmax(axis=0)
            
            found = True
            break
    if not found:
        print(f"Patch '{patch_name}' has not been found!")
        return

    # -------------------------------------------------------------------
    # 2. Set Up Visualization Parameters
    # -------------------------------------------------------------------
    # Create a colormap using the custom colors for 5 classes
    # (Ensure custom_colors has at least 5 colors)
    cmap = mcolors.ListedColormap(custom_colors[:5])
    
    # Scale bar parameters
    pixel_size_m = 0.024           # Physical size of one pixel in meters (e.g., 2.4 cm)
    desired_length_m = 1.0         # Desired scale bar length in meters
    scale_bar_length_px = desired_length_m / pixel_size_m  # Convert physical length to pixel units
    
    # -------------------------------------------------------------------
    # 3. Create the Figure and Subplots
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- Original Image Plot ---
    axes[0].imshow(t_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    # Add scale bar to the original image subplot
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(axes[0].transData, scale_bar_length_px, f"{desired_length_m} m",
                               "lower right", pad=0.1, color="white", frameon=False, size_vertical=2)
    axes[0].add_artist(scalebar)
    
    # --- Ground Truth Mask Plot ---
    axes[1].imshow(t_true_mask, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    # Add scale bar to the ground truth mask subplot
    scalebar = AnchoredSizeBar(axes[1].transData, scale_bar_length_px, f"{desired_length_m} m",
                               "lower right", pad=0.1, color="white", frameon=False, size_vertical=2)
    axes[1].add_artist(scalebar)
    
    # --- Predicted Mask Plot ---
    axes[2].imshow(t_pred_mask, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    # Add scale bar to the predicted mask subplot
    scalebar = AnchoredSizeBar(axes[2].transData, scale_bar_length_px, f"{desired_length_m} m",
                               "lower right", pad=0.1, color="white", frameon=False, size_vertical=2)
    axes[2].add_artist(scalebar)
    
    # -------------------------------------------------------------------
    # 4. Create Legend for the Classes
    # -------------------------------------------------------------------
    # Only use class indices 0 to n_classes-1
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=custom_colors[i], markersize=10,
                   linestyle="None", label=reversed_codes.get(i, f"Class {i}"))
        for i in range(n_classes)
    ]
    axes[2].legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # -------------------------------------------------------------------
    # 5. Add Overall Title with Patch Name and Final Layout Adjustments
    # -------------------------------------------------------------------
    fig.suptitle(f"Patch: {patch_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if show:
        plt.show()
    
    return fig




######################################################################################
# Visualize all test patches and save as pdf!

def visualize_all_test_patches(test_loader, model, device, reversed_codes, custom_colors, output_pdf_base):
    """
    Iterates over all patches in the test DataLoader, splits them into 8 groups,
    and saves each group as a separate PDF.
    
    Args:
        test_loader (DataLoader): PyTorch DataLoader containing test images and masks.
        model (torch.nn.Module): Trained model for generating predictions.
        device (torch.device): Device to perform computations on.
        reversed_codes (dict): Dictionary mapping class indices to class names.
        custom_colors (list): List of custom color definitions for each class.
        output_pdf_base (str): Base file path/name for the output PDFs.
    """
    
    # Collect all patch names from the test dataset
    patch_names = []
    for batch in test_loader:
        names, _, _ = batch  # only the names
        patch_names.extend(names)
    total_patches = len(patch_names)

    # Create a tqdm progress bar for all patches
    # pbar = tqdm(total=total_patches, desc="Plotting patches")
    
    # Define the number of chunks (PDFs) to split into
    num_chunks = 8
    # Calculate chunk size (round up to include all patches)
    chunk_size = math.ceil(total_patches / num_chunks)
    
    # Process each chunk separately
    for chunk in range(num_chunks):
        # Define the output PDF filename for the current chunk
        pdf_filename = f"{output_pdf_base}_part{chunk+1}.pdf"
        with PdfPages(pdf_filename) as pdf:
            # Determine the patch names for this chunk
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, total_patches)
            current_chunk = patch_names[start_idx:end_idx]
            
            # Iterate through the patch names in the current chunk
            for patch_name in current_chunk:
                # Call the visualization function for the current patch (without showing the plot)
                visualize_prediction(
                    patch_name, test_loader, model, device, reversed_codes, custom_colors, show=False
                )
                # Retrieve the current figure
                fig = plt.gcf()
                # Save the figure as a new page in the PDF
                pdf.savefig(fig)
                # Close the figure to free memory
                plt.close(fig)
                # Update progress bar by 1 patch
                # pbar.update(1)
            
        print(f"Saved PDF: {pdf_filename}")

    pbar.close()
    print("All PDF parts have been saved.")


























