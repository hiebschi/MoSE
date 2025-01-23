"""
Model utils
---------------------------
Helper functions used for the segmentation model implementation.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import segmentation_models_pytorch as smp

# import configs.py-file
import importlib
from configs import configs_gc
importlib.reload(configs_gc) # reload changes

model_0 = smp.Unet(   # -------------------->> ADJUSTABLE
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (3 for RGB)
    classes=configs_gc.HYPERPARAMETERS["num_classes"],       # model output channels (number of classes)
)


DiceLoss_fn = smp.losses.DiceLoss(mode="multilabel")



