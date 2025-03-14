# configs.py
"""
Configs for Google Colab
---------------------------
Central configuration file for all important hyperparameters and paths for implementation on google colab.
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
BASE_DIR = '/content/drive/My Drive/Dokumente.GD/FS06 SS24/BACHELORARBEIT/MoSE/data'
# BASE_DIR = r"C:\Users\simon\Meine Ablage\Dokumente.GD\FS06 SS24\BACHELORARBEIT\MoSE\data"

# Data Paths
DATA_DIR = {
    "preprocessed_patches_unz": os.path.join(BASE_DIR, 'preprocessed_patches_unz'),
    "masks": os.path.join(BASE_DIR, 'masks'),
    "masks_ohe": os.path.join(BASE_DIR, 'masks_ohe'),
    "targets": os.path.join(BASE_DIR, 'targets'),
    "codes": os.path.join(BASE_DIR, 'classes'),
    "labels": os.path.join(BASE_DIR, 'shapefiles'),
    "saved_models": os.path.join(BASE_DIR, 'saved_models')
}

######################################################
# Hyperparameters

HYPERPARAMETERS = {
    "train_sections": ["A01", "A02", "A03", "A04", "A05", "A08"],
    "test_sections": ["A06", "A07"],
    "num_classes": 10,
    "batch_size": 8,
    "epochs": 30,
    "learning_rate": 0.00001,
    "seed": 42
}

# Set Random Seeds for Reproducibility
SEED = HYPERPARAMETERS["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

