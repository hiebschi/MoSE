"""
Model utils
---------------------------
Helper functions used for the segmentation model implementation.

If a function gets defined once and could be used over and over, it'll go in here.
"""

import segmentation_models_pytorch as smp

model_0 = smp.Unet(   # -------------------->> ADJUSTABLE
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (3 for RGB)
    classes=9,       # model output channels (number of classes)
)


DiceLoss_fn = smp.losses.DiceLoss(mode="multilabel")



