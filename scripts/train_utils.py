"""
Train utils
---------------------------
Helper functions used for the training of the segmentation model.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import torch

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
    train_loss_class = torch.zeros(num_classes, device=device)  # loss for every class

    # loop through the batches
    for batch, (names, train_images, train_masks) in enumerate(data_loader):
        # for each batch, go over each patch and mask inside it
        
        # Send data to device
        train_images, train_masks = train_images.to(device), train_masks.to(device)

        # 1. Forward pass
        train_logits = model(train_images)
        # Shape: torch.Size([8, 10, 512, 512])
        # Decimal numbers between -7 and 6
        # print(train_logits.shape)
        # print(torch.min(train_logits))
        # print(torch.max(train_logits))

        # 2. Calculate loss and accuracy per batch
        # Train loss
        # loss_batch = loss_fn(train_logits, train_masks) # loss value over current batch (all patches inside)
        # Masks shape: torch.Size([8, 10, 512, 512])
        # convert one-hot-encoded masks into class-index-format
        train_targets = torch.argmax(train_masks, dim=1) # index of the highest class
        train_targets
        # print(train_targets.shape)  # shape: [batch_size, 512, 512]
        # print(train_targets.dtype) # torch.int64
        print(torch.unique(train_targets))  # check, if all values between 0 and 9
        
        # print(torch.min(train_targets))
        # print(torch.max(train_targets))

        print(f"Logits Shape: {train_logits.shape}, Min: {torch.min(train_logits)}, Max: {torch.max(train_logits)}")
        print(f"Targets Shape: {train_targets.shape}, Unique: {torch.unique(train_targets)}")

        loss_batch = loss_fn(train_logits, train_targets)
        train_loss_epoch += loss_batch.item() # accumulatively add up the loss >> added up loss in one epoch

        # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
        train_pred_probs = torch.sigmoid(train_logits) 
        # Shape: torch.Size([8, 10, 512, 512])
        # Decimal numbers between 0 and 1

        # go from prediction probabilities to prediction labels (binary: 0 or 1)
        train_preds = torch.round(train_pred_probs)
        # Shape: torch.Size([8, 10, 512, 512])
        # Only 0 or 1

        # Accuracy
        train_acc_epoch += accuracy_fn(train_masks, train_preds) # added up accuracy in one epoch

        # # Update class-wise loss
        # for cls in range(num_classes):
        #     train_loss_class[cls] += loss_fn(
        #         train_pred_probs[:, cls, :, :], train_masks[:, cls, :, :]
        #     ).item()

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
    train_loss_class /= len(data_loader)
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
    test_loss_class = torch.zeros(num_classes, device=device)  # loss for every class
    
    # Turn on inference context manager
    with torch.inference_mode():
        for names, test_images, test_masks in data_loader:
        # for batch_idx, (names, test_images, test_masks) in enumerate(data_loader):

            # Send data to device
            test_images, test_masks = test_images.to(device), test_masks.to(device)

            # 1. Forward pass
            test_logits = model(test_images)
            # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
            test_pred_probs = torch.sigmoid(test_logits) 

            # 2. Calculate loss and accuracy
            # Test loss
            test_loss_epoch += loss_fn(test_pred_probs, test_masks) # accumulatively add up the loss >> added up loss in one epoch

            # go from prediction probabilities to prediction labels (binary: 0 or 1)
            test_preds = torch.round(test_pred_probs) 
            # Accuracy
            test_acc_epoch += accuracy_fn(test_masks, test_preds) # added up accuracy in one epoch

            # Update class-wise loss
            for cls in range(num_classes):
                test_loss_class[cls] += loss_fn(
                    test_pred_probs[:, cls, :, :], test_masks[:, cls, :, :]
                ).item()

        # Calculate average loss and average accuracy of current epoch
        test_loss_epoch /= len(data_loader) # divide the added up loss through the number of batches
        # >> average loss of current epoch 
        test_acc_epoch /= len(data_loader) # len(data_loader) = number of batches
        # >> average accuracy of current epoch
        test_loss_class /= len(data_loader)
        # >> average loss per class of current epoch
        
        # Log testing loop results
        print(f"Test loss: {test_loss_epoch:.5f} | Test accuracy: {test_acc_epoch:.2f}%\n")
        print(f"Test Class-wise Loss: {test_loss_class.tolist()}")

        # (end of epoch loop)

        return test_loss_epoch