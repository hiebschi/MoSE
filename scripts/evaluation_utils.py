"""
Evaluation utils
---------------------------
Helper functions used for the evaluation of the trained segmentation model.
"""

import torch
from torch import nn
import torch.nn.functional as F





########################
# INSIDE TRAINING LOOP 
# evaluate on-the-fly
########################


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



#############################################
# No. 2: Class-wise loss

def calculate_classwise_loss(logits, targets, num_classes, device):
    """
    Calculates loss per class for one batch.
    
    Args:
        logits (torch.Tensor): raw model output with shape [batch_size, num_classes, H, W] and decimal numbers between -7 and 6
        targets (torch.Tensor): true/ ground truth targets with shape [batch_size, H, W] and integers between 0 and [num_classes - 1]
        num_classes (int): number of classes.
        device (torch.device): Device that compute is running on.
    
    Returns:
        classwise_loss (torch.Tensor): Tensor with loss values per class.
    """
    
    # initialize tensor for class-wise-loss
    class_wise_loss = torch.zeros(num_classes).to(device)

    # calculate loss for each individual pixel 
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # reduction = none -> means no averaging of the loss over the entire batch (no mean calculation)
    loss = loss_fn(logits, targets).to(device)  # loss.shape: [batch_size, H, W]

    # iterate over all classes
    for cls_idx in range(num_classes):
        
        # mask for class cls_idx
        mask = (targets == cls_idx).float().to(device)  # [batch_size, H, W]
        # all pixels belonging to the class cls_idx get the value 1, all others 0

        # add up the loss values of the pixels in the class and calculate the mean value
        class_loss = (loss * mask).sum() / (mask.sum() + 1e-6)  # + 1e-6 to prevent division by 0
        class_loss = class_loss.to(device)

        # save loss values in a tensor
        class_wise_loss[cls_idx] += class_loss.item()
            
    return class_wise_loss




#########################
# OUTSIDE TRAINING LOOP 
# evaluate trained model
#########################



# EIG GETRENNT!

# MAKE_PREDICTIONS AND EVALUATE THESE PREDICTIONS!!!




##############################################
# No. 3: Confusion Matrix
# AND
# No. 4: F1 Score


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model_with_testdata(model, test_loader, accuracy_fn, num_classes, device):
    """
    Evaluates a trained model on the entire/ a part of test dataset and computes the confusion matrix. 
    Additionaly it calculates the overall accuracy and if desired also the F1-Score.
    
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_classes (int): Number of classes.
        device (str): Device to run the model on ("cpu" or "cuda").
    
    Returns:
        None (plots the confusion matrix)
    """
    
    model.eval()  # Set model to evaluation mode

    test_acc = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():  # no gradient calculation needed
        for names, test_images, test_masks in test_loader:

            # Send data to device
            test_images, test_masks = test_images.to(device), test_masks.to(device)

            # Targets: convert one-hot-encoded masks into class-index-format
            test_targets = torch.argmax(test_masks, dim=1) # index of the highest class

            # Get model predictions
            # Do the forward pass
            test_logits = model(test_images) 

            # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
            test_pred_probs = torch.sigmoid(test_logits)

            # go from prediction probabilities to prediction labels (binary: 0 or 1)
            test_preds = torch.round(test_pred_probs)

            # convert one-hot-encoded predictions into class-index-format
            test_preds_idxformat = torch.argmax(test_preds, dim=1) # model output

            # Accuracy
            # Compare true masks/targets with predicted masks/targets
            test_acc += accuracy_fn(test_targets, test_preds_idxformat) # added up accuracy

            # store predictions and targets
            all_test_preds.append(test_preds_idxformat.cpu().numpy().flatten())  
            all_test_targets.append(test_targets.cpu().numpy().flatten())

    # convert lists to numpy arrays
    all_test_preds = np.concatenate(all_test_preds)
    all_test_targets = np.concatenate(all_test_targets)

    test_acc /= len(test_loader) # len(test_loader)
    print(f"Test accuracy: {test_acc:.2f}%\n")

    # compute Confusion Matrix
    cm = confusion_matrix(all_test_targets, all_test_preds, labels=np.arange(num_classes))

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel("Predicted Class per Pixel")
    plt.ylabel("True Class per Pixel")
    plt.title("Confusion Matrix on Test Dataset")
    plt.show()

















##############################################
# No. 4: F1 Score / Dice similarity coefficient (DSC) per class

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




































# OLD VERSION (WRONG)
# def calculate_classwise_loss(train_logits, train_targets, num_classes):
#     """
#     Calculates loss per class.
    
#     Args:
#         train_logits (torch.Tensor): raw model output with shape [batch_size, num_classes, H, W] and decimal numbers between -7 and 6
#         train_targets (torch.Tensor): true/ ground truth targets with shape [batch_size, H, W] and integers between 0 and [num_classes - 1]
#         num_classes (int): number of classes.
    
#     Returns:
#         classwise_loss (list): list with loss per class.
#     """
    
#     loss_without_weights_fn = F.binary_cross_entropy_with_logits

#     train_class_wise_loss = torch.zeros(num_classes)  # Class-wise loss

    
#     for cls_idx in range(num_classes): # loop over all classes
            
#         # creates masks with TRUE values for each pixel that actually (in reality) 
#         # belongs to the class with the index cls_idx
#         class_mask = (train_targets == cls_idx) # shape: [batch_size, H, W]; dtype: bool
#         # print(class_mask.shape, class_mask.dtype)

#         if class_mask.sum() > 0:  # avoid division by zero -> if there are any pixels for this class, do this:

#             print(f"cls_idx: {cls_idx}")
            
#             # Isolate logits and targets for the current class
#             class_logits = train_logits[:, cls_idx, :, :]  # Shape: [batch_size, H, W]
#             # class_logits = class_logits.unsqueeze(1) 
#             #print("class_logits:", class_logits)
#             #print("class_logits:", class_logits.shape, class_logits.dtype)
            
#             class_targets = (train_targets == cls_idx)
#             #print("train_targets.shape:", train_targets.shape)
#             #print("train_targets dtype:", train_targets.dtype)
#             #print("train_targets min/max:", train_targets.min(), train_targets.max())
#             #print("unique values in train_targets:", torch.unique(train_targets))
#             class_targets = class_targets.float()  
#             #print("class_targets:", class_targets)
#             #print("class_targets.shape:", class_targets.shape)
#             #print("class_targets dtype:", class_targets.dtype)
#             #print("class_targets min/max:", class_targets.min(), class_targets.max())
#             #print("unique values in class_targets:", torch.unique(class_targets))
            
#             # Calculate loss for the current class
#             class_loss = loss_without_weights_fn(class_logits, class_targets)
#             train_class_wise_loss[cls_idx] += class_loss.item()
            
#             print("Train Class-Wise Loss:", train_class_wise_loss)
#             # Train Class-Wise Loss: tensor([0.5633, 0.0000, 0.7312, 0.0000, 0.6060, 0.0000, 0.6437, 0.4540, 0.5208,
#             #0.0000])






