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
    "patches": os.path.join(BASE_DIR, 'patches'),
    "masks": os.path.join(BASE_DIR, 'masks'), # one-hot-encoded masks
    "codes": os.path.join(BASE_DIR, 'codes'),
    # "models": os.path.join('/home/sc.uni-leipzig.de/rf37uqip/MoSE/models/') # save trained models in repository!
    "models": os.path.join('/work/rf37uqip-MoSE_thesis/data/models/'), # save trained models in workspace!
    "figures": os.path.join(BASE_DIR, 'figures')
}

######################################################
# Hyperparameters

HYPERPARAMETERS = {
    "num_classes": 5,
    "epochs": 50,
    "data_augmentation": False, 
    "learning_rate": 1e-6,     # Hyperparameter Tuning showed that lr of 1e-6 leads to the best performance
    "train_sections": ["A01", "A02", "A03", "A05", "A06"],
    "test_sections": ["A04", "A07", "A08"],
    "batch_size": 8,
    "seed": 42,
    "custom_colors": # customized colors for each class
       [
        (0.12, 0.47, 0.61),  # 0: blue
        (0.65, 0.44, 0.29),  # 1: brown
        (0.40, 0.34, 0.29),  # 2: darkbrown
        (0.94, 0.74, 0.13),  # 3: orange
        (0.74, 0.74, 0.13),  # 4: olive
        ]  
}


# Set Random Seeds for Reproducibility
SEED = HYPERPARAMETERS["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

