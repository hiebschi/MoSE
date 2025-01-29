"""
Train utils
---------------------------
Helper functions used for the training of the segmentation model.
"""

import torch
import torch.nn.functional as F

############################################################
# Timing function

def print_train_time(start: float, end: float, device: torch.device = None):
    """
    Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")



############################################################
# Training loop

def train_step(model: torch.nn.Module,
               num_classes,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               showprint = True):

    """
    Performs a training loop step with model trying to learn on data_loader.
    
    Args:
        model (torch.nn.Module): Segmentation model.
        num_classes (int): Number of classes.
        data_loader (torch.utils.data.DataLoader): Batch-wise-grouped DataLoader [names, images, masks].
        loss_fn (torch.nn.Module): Function for calculating the loss.
        optimizer (torch.optim.Optimizer): Function for optimizing the parameters of the model.
        accuracy_fn: Calculates the overall accuracy between predicted and true masks.
        device (torch.device): Device that compute is running on.
        showprint (bool) = True: Whether the results should be printed or should not.

    Returns: 
         [torch.float]: Train loss value of current epoch.
    """

    # (start of epoch loop)

    # Send model to device
    model.to(device)

    # Put model into training mode
    model.train()

    # Define training loss and accuracy
    train_loss_epoch, train_acc_epoch = 0, 0
    train_class_wise_loss = {}  # loss for each class

    # loop through the batches
    for batch, (names, train_images, train_masks) in enumerate(data_loader):
        # for each batch, go over each patch and mask inside it
        
        # Send data to device
        train_images, train_masks = train_images.to(device), train_masks.to(device)

        # 1. Forward pass
        train_logits = model(train_images)
        train_logits = train_logits.float()
        # Shape: torch.Size([batch_size, num_classes, 512, 512])
        # Decimal numbers between -7 and 6

        # 2. Train loss and accuracy

        # Masks shape: torch.Size([batch_size, num_classes, 512, 512]) 
        # -> convert one-hot-encoded masks into class-index-format
        train_targets = torch.argmax(train_masks, dim=1) # index of the highest class
        train_targets = train_targets.long() 
        # .shape: [batch_size, 512, 512]
        # .dtype: torch.int64; Integers between 0 and [num_classes - 1]

        # 2.1 Loss per batch
        loss_batch = loss_fn(train_logits, train_targets)
        train_loss_epoch += loss_batch.item() # accumulatively add up the loss >> added up loss in one epoch

        # 2.2 Class-wise loss
        
        # for cls_idx in range(num_classes): # loop over all classes
        #     # creates masks with TRUE values for each pixel that actually (in reality) 
        #     # belongs to the class with the index cls_idx
        #     class_mask = (train_targets == cls_idx) # shape: [batch_size, 512, 512]; dtype: bool

        #     if class_mask.sum() > 0:  # avoid division by zero -> if there are any pixels for this class, do this:
                
        #         # extract logits of current class
        #         class_logits = train_logits[:, cls_idx, :, :] # Shape: [batch_size, 512, 512]
                
        #         # calculate loss only for relevant pixel
        #         loss = F.cross_entropy(class_logits, train_targets, reduction='none')  # Shape: [batch_size, 512, 512]
        #         class_loss = loss[class_mask].mean() # mean for relevant pixel
                
        #         # save loss for this class
        #         train_class_wise_loss[cls_idx] = class_loss.item()

        #         print("Train Class-Wise Loss:", train_class_wise_loss)


        # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
        train_pred_probs = torch.sigmoid(train_logits) 
        # Shape: torch.Size([batch_size, num_classes, 512, 512])
        # Decimal numbers between 0 and 1

        # go from prediction probabilities to prediction labels (binary: 0 or 1)
        train_preds = torch.round(train_pred_probs)
        # Shape: torch.Size([batch_size, num_classes, 512, 512])
        # Only 0 or 1

        # 2.3 Accuracy
        train_acc_epoch += accuracy_fn(train_masks, train_preds) # added up accuracy in one epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss_batch.backward()

        # 5. Optimizer step
        optimizer.step() # optimizer updates model's parameters once per batch

        # Print out how many batches have been seen
        if showprint:
            if batch % 100 == 0:
                print(f"Train Loss of [Batch {batch}/{len(data_loader)}]: {loss_batch.item():.4f}")


    # outside of the batch loop
    # (inside of the epoch loop)

    # Calculate average loss and average accuracy of current epoch
    train_loss_epoch /= len(data_loader) # divide the added up loss through the number of batches
    # >> average loss of current epoch 
    train_acc_epoch /= len(data_loader) # len(data_loader) = number of batches
    # >> average accuracy of current epoch
    # train_loss_class /= len(data_loader)
    # >> average loss per class of current epoch

    # Log training loop results
    print(f"Train loss: {train_loss_epoch:.5f} | Train accuracy: {train_acc_epoch:.2f}%")
    # print(f"Train Class-wise Loss: {train_loss_class.tolist()}")

    return train_loss_epoch



############################################################
# Testing loop

def test_step(model: torch.nn.Module,
              num_classes,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    """
    Performs a testing loop step with model trying to learn on data_loader.
    
    Args:
        model (torch.nn.Module): Segmentation model.
        num_classes (int): Number of classes.
        data_loader (torch.utils.data.DataLoader): Batch-wise-grouped DataLoader [names, images, masks].
        loss_fn (torch.nn.Module): Function for calculating the loss.
        accuracy_fn: Calculates the overall accuracy between predicted and true masks.
        device (torch.device): Device that compute is running on.
        
    Returns: 
         [torch.float]: Train loss value of current epoch.
    """

    # (inside of the epoch loop)

    # Send model to device
    model.to(device)

    # Put model into evaluation mode
    model.eval()

    # Define test loss and accuracy
    test_loss_epoch, test_acc_epoch = 0, 0
    ###### !!!! test_loss_class = torch.zeros(num_classes, device=device)  # loss for every class
    
    # Turn on inference context manager
    with torch.inference_mode():
        for names, test_images, test_masks in data_loader:
        # for batch_idx, (names, test_images, test_masks) in enumerate(data_loader):

            # Send data to device
            test_images, test_masks = test_images.to(device), test_masks.to(device)

            # 1. Forward pass
            test_logits = model(test_images) 

            # 2. Test loss and accuracy

            # Targets: convert one-hot-encoded masks into class-index-format
            test_targets = torch.argmax(test_masks, dim=1) # index of the highest class
            test_targets

            # 2.1 Loss
            test_loss_epoch += loss_fn(test_logits, test_targets) # accumulatively add up the loss >> added up loss in one epoch
            
            # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
            test_pred_probs = torch.sigmoid(test_logits)
        
            # go from prediction probabilities to prediction labels (binary: 0 or 1)
            test_preds = torch.round(test_pred_probs) 
            
            # 2.2 Accuracy
            test_acc_epoch += accuracy_fn(test_masks, test_preds) # added up accuracy in one epoch

            # # Update class-wise loss
            # for cls in range(num_classes):
            #     test_loss_class[cls] += loss_fn(
            #         test_pred_probs[:, cls, :, :], test_masks[:, cls, :, :]
            #     ).item()

        # Calculate average loss and average accuracy of current epoch
        test_loss_epoch /= len(data_loader) # divide the added up loss through the number of batches
        # >> average loss of current epoch 
        test_acc_epoch /= len(data_loader) # len(data_loader) = number of batches
        # >> average accuracy of current epoch
        # test_loss_class /= len(data_loader)
        # >> average loss per class of current epoch
        
        # Log testing loop results
        print(f"Test loss: {test_loss_epoch:.5f} | Test accuracy: {test_acc_epoch:.2f}%\n")
        # print(f"Test Class-wise Loss: {test_loss_class.tolist()}")

        # (end of epoch loop)

        return test_loss_epoch