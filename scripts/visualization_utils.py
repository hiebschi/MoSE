"""
Visualization utils
---------------------------
Helper functions used for the visual evaluation and depiction of the results 
of the segmentation model implementation.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import numpy as np
import matplotlib.pyplot as plt
import math



def norm_plot_patch(patch):
    """
    Normalizes and plots a single preprocessed patch.
    
    Args:
        patch (numpy.ndarray): preprocessed patch/ image data
    
    Returns: 
        Plot of normalized image
    """

    # Normalization
    patch_normalized = patch - np.min(patch ) # set minimum to 0
    patch_normalized = patch_normalized / np.max(patch_normalized)  # maximize to 1

    # Plot the preprocessed image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(patch_normalized.transpose(1, 2, 0))  # transpose for RGB depiction
    ax.set_title("Preprocessed Patch")
    plt.show()


