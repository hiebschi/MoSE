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
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time



############################################################
# Training loop

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               showprint = True):

    """
    Performs a training loop step with model trying to learn on data_loader.
    
    
    
    
    """
    # (inside of the epoch loop)

    # Send model to device
    model.to(device)

    # Put model into training mode
    model.train()

    # Define training loss and accuracy
    train_loss_epoch, train_acc_epoch = 0, 0

    # loop through the batches
    for batch, (name, train_image, train_mask) in enumerate(data_loader):
        # for each batch, go over each patch and mask inside it
        
        # Send data to device
        train_image, train_mask = train_image.to(device), train_mask.to(device)

        # 1. Forward pass
        train_logits = model(train_image)
        # calculate the prediction probabilities for every pixel (to fit in a specific class or not)
        train_pred_probs = torch.sigmoid(train_logits) 

        # 2. Calculate loss and accuracy per batch
        # Train loss
        loss_batch = loss_fn(train_pred_probs, train_mask) # loss value over current batch (all patches inside)
        train_loss_epoch += loss_batch.item() # accumulatively add up the loss >> added up loss in one epoch

        # go from prediction probabilities to prediction labels (binary: 0 or 1)
        train_preds = torch.round(train_pred_probs) 
        # Accuracy
        train_acc_epoch += accuracy_fn(train_mask, train_preds) # added up accuracy in one epoch

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

    print(f"Train loss: {train_loss_epoch:.5f} | Train accuracy: {train_acc_epoch:.2f}%")

    return train_loss_epoch



############################################################
# Testing loop

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device,
              showprint = True):
    """
    Performs a testing loop step with model trying to learn on data_loader.
    
    
    
    
    """

    test_loss_epoch, test_acc_epoch = 0, 0
    model.to(device)

    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

