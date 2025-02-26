"""
Evaluation utils
---------------------------
Helper functions used for the evaluation of the trained segmentation model.
"""

# packages
import torch
from torch import nn
import torch.nn.functional as F
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


##############################################
# No. 1: Confusion Matrix with overall and class-wise accuracy.
# No. 2: F1 Score

def evaluate_model_with_testdata(model, test_loader, accuracy_fn, num_classes, device, F1_analysis=False):
    """
    Evaluates a trained model on the test dataset and computes the confusion matrix.
    Additionally, it calculates the overall and class-wise accuracy and, if desired,
    the F1-Score based on the confusion matrix computed incrementally.
    
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_classes (int): Number of classes.
        device (str): Device to run the model on ("cpu" or "cuda").
        F1_analysis (bool): If True, the F1 score is calculated from the confusion matrix.
    
    Returns:
        None (plots the confusion matrix and prints evaluation metrics)
    """
    
    model.eval()  # Set model to evaluation mode

    test_acc = 0
    # Initialize a total confusion matrix (incremental accumulation)
    cm_total = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():  # no gradient calculation needed
        for names, test_images, test_masks in test_loader:
            # Send data to device
            test_images, test_masks = test_images.to(device), test_masks.to(device)

            # Convert one-hot-encoded masks into class-index format (ground truth)
            test_targets = torch.argmax(test_masks, dim=1)  # shape: [batch_size, H, W]

            # Forward pass
            test_logits = model(test_images)  # model output
            test_pred_probs = torch.sigmoid(test_logits)
            test_preds = torch.round(test_pred_probs)
            test_preds_idxformat = torch.argmax(test_preds, dim=1)  # predicted class indices

            # Accuracy: compare true masks with predicted masks
            test_acc += accuracy_fn(test_targets, test_preds_idxformat)

            # Compute batch confusion matrix (flatten arrays for pixel-wise comparison)
            targets_np = test_targets.cpu().numpy().flatten()
            preds_np = test_preds_idxformat.cpu().numpy().flatten()
            cm_batch = confusion_matrix(targets_np, preds_np, labels=np.arange(num_classes))
            cm_total += cm_batch

    # Average accuracy over batches
    test_acc /= len(test_loader)
    print(f"Test accuracy: {test_acc:.2f}%\n")

    # Calculate class-wise accuracy based on the accumulated confusion matrix
    test_class_acc = []
    for i in range(num_classes):
        correct = cm_total[i, i]  # correct classified pixels of class i
        total = cm_total[i, :].sum()  # total pixels of class i
        if total == 0:
            test_class_acc.append(-999)  # avoid division by zero
        else:
            test_class_acc.append(correct / total)
    print(f"Test class-wise accuracies: {test_class_acc}\n")

    # Plot Confusion Matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel("Predicted Class per Pixel")
    plt.ylabel("True Class per Pixel")
    plt.title("Confusion Matrix on Test Dataset")
    plt.show()

    # F1-Score Analysis based on confusion matrix
    if F1_analysis:
        
        # lists for precision, recall, and F1-score for each class
        f1_scores = []
        precisions = []
        recalls = []
        
        for i in range(num_classes):
            
            # True positives for class i
            true_pos = cm_total[i, i]
            
            # Total predicted positives for class i (sum of column i)
            predicted_pos_total = cm_total[:, i].sum()
            # Total actual positives for class i (sum of row i)
            true_pos_total = cm_total[i, :].sum()
            
            # Calculate precision (avoid division by zero)
            precision = true_pos / predicted_pos_total if predicted_pos_total > 0 else 0

            # Calculate recall
            recall = true_pos / true_pos_total if true_pos_total > 0 else 0

            # save values per class
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate F1-score: if precision + recall is 0, define F1 as 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        # Compute macro averages for precision, recall, and F1-score
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        # Create a table with class-wise metrics
        results_table = pd.DataFrame({
            "Class": list(range(num_classes)),
            "Precision": np.round(precisions, 4),
            "Recall": np.round(recalls, 4),
            "F1-Score": np.round(f1_scores, 4)
        })
        print("Class-wise evaluation metrics:")
        print(results_table)
        print(f"\nMacro Precision: {np.round(macro_precision, 4)}")
        print(f"Macro Recall: {np.round(macro_recall, 4)}")
        print(f"Macro F1-Score: {np.round(macro_f1, 4)}")



# ##############################################
# # No. 6: F1 Score / Dice similarity coefficient (DSC) per class

# def F1_per_class(true_mask, pred_mask, showprint = True):
#     """
#     Calculates the class-wise Dice similarity coefficient (DSC) / F1-score = size of the intersection as 
#     a fraction of the average size of the two sets. It is the  harmonic mean of the precision and recall. 
#     It thus symmetrically represents both precision and recall in one metric. 

#     Args:
#         true_mask (torch.Tensor): Ground truth labels (shape: [batch_size, class channels, height, width]).
#         pred_mask (torch.Tensor): Predicted labels (shape: [batch_size, class channels, height, width]).
#         showprint (bool) = True: Whether the results should be printed or should not.

#     Returns:
#         float: Class-wise DSC (percentage).
#     """

#     results = {}
#     for i in range(true_mask.shape[0]): # iterate through the class channels
        
#         class_pred = pred_mask[i, :, :]  # look at channel i = class i + 1
#         class_true = true_mask[i, :, :]

#         # Convert to boolean for bitwise operations
#         class_pred = class_pred.type(torch.bool) 
#         class_true = class_true.type(torch.bool) 

#         # logical operations (intersection and union)
#         intersection = (class_pred & class_true).sum().item()
#         union = (class_pred | class_true).sum().item()

#         dice = 2 * intersection / (union + 1e-6) # DSC formula
#         results[i] = dice # save results

#         if showprint:
#             print(f"Class {i + 1}: Dice = {dice:.4f}")
    
#     return results





# #############################################
# # No. 2: Class-wise accuracy

# def class_wise_acc_fn(true_targets, pred_targets, num_classes):
#     """
#     CLASS-WISE ACCURACY
#     Calculates class-wise accuracy per batch.

#     Args:
#         true_targets (torch.Tensor): ground truth targets in class-index-format (shape: [batch_size, height, width]).
#         pred_targets (torch.Tensor): predicted targets in class-index-format (shape: [batch_size, height, width]).
#         num_classes (int): Number of classes.

#     Returns:
#         class_acc (list): list of accuracy values per class.
#     """

#      # Flatten the tensors to compute over all pixels and covert into numpy-arrays
#     true_targets = true_targets.cpu().numpy().flatten() # Shape: [total_pixels]
#     pred_targets = pred_targets.cpu().numpy().flatten()  # Shape: [total_pixels]

#     # Calculate the confusion matrix
#     cm = confusion_matrix(true_targets, pred_targets, labels=np.arange(num_classes))

#     # Calculate accuracy for each class
#     class_acc = []
#     for i in range(num_classes):
#         correct = cm[i, i]  # correct classified samples of class i
#         total = cm[i, :].sum()  # total number of samples of class i
#         if total == 0:
#             class_acc.append(0.0)  # avoid division with zero
#         else:
#             class_acc.append(correct / total)

#     return class_acc