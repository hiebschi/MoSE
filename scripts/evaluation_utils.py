"""
Evaluation utils
---------------------------
Helper functions used for the evaluation of the trained segmentation model.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import torch

# Classification metrics

# Overall accuracy

def accuracy_fn(true_mask, pred_mask):
    """
    Calculates pixel-wise accuracy for segmentation tasks.

    Args:
        true_mask (torch.Tensor): Ground truth labels (shape: [batch_size, class channels, height, width]).
        pred_mask (torch.Tensor): Predicted labels (shape: [batch_size, class channels, height, width]).

    Returns:
        float: Pixel-wise accuracy as a percentage.
    """

    # Flatten the tensors to compute over all pixels
    true_mask = true_mask.view(-1)  # Shape: [total_pixels]
    pred_mask = pred_mask.view(-1)  # Shape: [total_pixels]

    # Calculate the number of correctly classified pixels
    correct = torch.eq(true_mask, pred_mask).sum().item()

    # Calculate the total number of pixels
    total_pixels = true_mask.numel()

    # Compute accuracy
    acc = (correct / total_pixels) * 100
    return acc


# Dice similarity coefficient (DSC) per class

def DSC_per_class(true_mask, pred_mask, showprint = True):
    """
    Calculates the class-wise Dice similarity coefficient (DSC) / F1-score = size of the intersection as 
    a fraction of the average size of the two sets. It is the  harmonic mean of the precision and recall. 
    It thus symmetrically represents both precision and recall in one metric. 

    Args:
        true_mask (torch.Tensor): Ground truth labels (shape: [batch_size, class channels, height, width]).
        pred_mask (torch.Tensor): Predicted labels (shape: [batch_size, class channels, height, width]).
        showprint (bool) = True: Whether the results should be printed or should not.

    Returns:
        float: Class-wise DSC (percentage).
    """

    results = {}
    for i in range(true_mask.shape[0]): # iterate through the class channels
        
        class_pred = pred_mask[i, :, :]  # look at channel i = class i + 1
        class_true = true_mask[i, :, :]

        # Convert to boolean for bitwise operations
        class_pred = class_pred.type(torch.bool) 
        class_true = class_true.type(torch.bool) 

        # logical operations (intersection and union)
        intersection = (class_pred & class_true).sum().item()
        union = (class_pred | class_true).sum().item()

        dice = 2 * intersection / (union + 1e-6) # DSC formula
        results[i] = dice # save results

        if showprint:
            print(f"Class {i + 1}: Dice = {dice:.4f}")
    
    return results

