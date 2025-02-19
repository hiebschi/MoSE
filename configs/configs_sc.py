# configs.py
"""
Configs for Scientific Computing
---------------------------
Central configuration file for all important hyperparameters and paths for implementation on university servers (SC Infrastructure).
"""

######################################################
# import required packages

import os
import numpy as np

import torch
import random


######################################################
# Directories

# Base Directory
BASE_DIR = '/lscratch/data'

# Data Paths
DATA_DIR = {
    "preprocessed_patches_unz": os.path.join(BASE_DIR, 'preprocessed_patches_unz'),
    "masks": os.path.join(BASE_DIR, 'masks'),
    "masks_ohe": os.path.join(BASE_DIR, 'masks_ohe'),
    "targets": os.path.join(BASE_DIR, 'targets'),
    "codes": os.path.join(BASE_DIR, 'classes'),
    "labels": os.path.join(BASE_DIR, 'shapefiles'),
    "saved_models": os.path.join('/home/sc.uni-leipzig.de/rf37uqip/MoSE/saved_models/')
}

######################################################
# Hyperparameters

HYPERPARAMETERS = {
    "train_sections": ["A01", "A02", "A03", "A04", "A05", "A08"],
    "test_sections": ["A06", "A07"],
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 30,
    "learning_rate": 1e-6,     # Hyperparameter Tuning: 1e-5 (first successfull model: loss going down, but overfitting) 
                                # >> therefore: 1e-6 and 1e-7 # 1e-6 is the best
    "seed": 42,
    "custom_colors": # customized colors for each class
       [
        (0.12, 0.47, 0.61),  # 0: blue
        (0.84, 0.15, 0.16),  # 1: RED (not existent)
        (0.40, 0.34, 0.29),  # 2: darkbrown
        (0.84, 0.15, 0.16),  # 3: RED (not existent)
        (0.65, 0.44, 0.29),  # 4: brown
        (0.84, 0.15, 0.16),  # 5: RED (not existent)
        (0.94, 0.74, 0.13),  # 6: orange
        (0.84, 0.15, 0.16),  # 7: RED (not existent)
        (0.74, 0.74, 0.13),  # 8: olive
        (0.54, 0.74, 0.13),  # 9: green
        ]  
}

# Set Random Seeds for Reproducibility
SEED = HYPERPARAMETERS["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

