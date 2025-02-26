"""
Model utils
---------------------------
Helper functions used for the segmentation model implementation.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

# import configs.py-file
import importlib
from configs import configs_sc
importlib.reload(configs_sc) # reload changes

model_0 = smp.Unet(   # -------------------->> ADJUSTABLE
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (3 for RGB)
    classes=configs_sc.HYPERPARAMETERS["num_classes"],       # model output channels (number of classes)
)


# DiceLoss_fn = smp.losses.DiceLoss(mode="multilabel")
# CrossEntropyLoss_fn = nn.CrossEntropyLoss()



