"""
Evaluation utils
---------------------------
Helper functions used for the evaluation of the trained segmentation model.
"""

import torch
from torch import nn
import torch.nn.functional as F

# Classification metrics
####################################


# No. 1: Overall accuracy
####################################

def oa_accuracy_fn(true_targets, pred_targets):
    """
    OVERALL ACCURACY
    Calculates overall accuracy per batch: correct classified pixels / all pixels.
    Compares the number of correct classifications to all the classifications to be made.
    All arguments have to be in class-index-format!

    Args:
        true_targets (torch.Tensor): ground truth targets in class-index-format (shape: [batch_size, height, width]).
        pred_targets (torch.Tensor): predicted targets in class-index-format (shape: [batch_size, height, width]).

    Returns:
        float: Pixel-wise accuracy as a percentage.
    """

    # Flatten the tensors to compute over all pixels
    true_targets = true_targets.view(-1)  # Shape: [total_pixels]
    pred_targets = pred_targets.view(-1)  # Shape: [total_pixels]

    # Calculate the number of correctly classified pixels
    correct = torch.eq(true_targets, pred_targets).sum().item()

    # Calculate the total number of pixels
    total_pixels = true_targets.numel()

    # Compute overall accuracy
    acc = (correct / total_pixels) * 100
    return acc

    


##############################################
# F1 Score / Dice similarity coefficient (DSC) per class

def F1_per_class(true_mask, pred_mask, showprint = True):
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




#############################################
# Class-wise loss

def calculate_classwise_loss(train_logits, train_targets, num_classes):
    """
    Calculates loss per class.
    
    Args:
        train_logits (torch.Tensor): raw model output with shape [batch_size, num_classes, H, W] and decimal numbers between -7 and 6
        train_targets (torch.Tensor): true/ ground truth targets with shape [batch_size, H, W] and integers between 0 and [num_classes - 1]
        num_classes (int): number of classes.
    
    Returns:
        classwise_loss (list): list with loss per class.
    """
    
    loss_without_weights_fn = F.binary_cross_entropy_with_logits

    train_class_wise_loss = torch.zeros(num_classes)  # Class-wise loss

    
    for cls_idx in range(num_classes): # loop over all classes
            
        # creates masks with TRUE values for each pixel that actually (in reality) 
        # belongs to the class with the index cls_idx
        class_mask = (train_targets == cls_idx) # shape: [batch_size, H, W]; dtype: bool
        # print(class_mask.shape, class_mask.dtype)

        if class_mask.sum() > 0:  # avoid division by zero -> if there are any pixels for this class, do this:

            print(f"cls_idx: {cls_idx}")
            
            # Isolate logits and targets for the current class
            class_logits = train_logits[:, cls_idx, :, :]  # Shape: [batch_size, H, W]
            # class_logits = class_logits.unsqueeze(1) 
            #print("class_logits:", class_logits)
            #print("class_logits:", class_logits.shape, class_logits.dtype)
            
            class_targets = (train_targets == cls_idx)
            #print("train_targets.shape:", train_targets.shape)
            #print("train_targets dtype:", train_targets.dtype)
            #print("train_targets min/max:", train_targets.min(), train_targets.max())
            #print("unique values in train_targets:", torch.unique(train_targets))
            class_targets = class_targets.float()  
            #print("class_targets:", class_targets)
            #print("class_targets.shape:", class_targets.shape)
            #print("class_targets dtype:", class_targets.dtype)
            #print("class_targets min/max:", class_targets.min(), class_targets.max())
            #print("unique values in class_targets:", torch.unique(class_targets))
            
            # Calculate loss for the current class
            class_loss = loss_without_weights_fn(class_logits, class_targets)
            train_class_wise_loss[cls_idx] += class_loss.item()
            
            print("Train Class-Wise Loss:", train_class_wise_loss)
            # Train Class-Wise Loss: tensor([0.5633, 0.0000, 0.7312, 0.0000, 0.6060, 0.0000, 0.6437, 0.4540, 0.5208,
            #0.0000])


def calculate_classwise_loss2(train_logits, train_targets, num_classes):

    train_class_wise_loss = torch.zeros(num_classes)  # Class-wise loss
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # no mean calculation for masking
    loss = loss_fn(train_logits, train_targets)  # loss.shape: [batch_size, H, W]
    print(loss)
    
    for cls_idx in range(num_classes):
        # Maske für Klasse cls_idx
        mask = (train_targets == cls_idx).float()  # [batch_size, H, W]
        class_loss = (loss * mask).sum() / (mask.sum() + 1e-6)  # Durchschnitt über die Maske
        train_class_wise_loss[cls_idx] += class_loss.item()
            
        print("Train Class-Wise Loss:", train_class_wise_loss)
        # Train Class-Wise Loss: tensor([1.8030, 2.9368, 2.0074, 0.0000, 2.2978, 0.0000, 2.1147, 0.0000, 0.0000,
        # 0.0000])



