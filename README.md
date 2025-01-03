# MoSE Overview 
Project MoSE = Mapping of structure elements in water bodies based on drone data using machine learning

## Introduction


## Repository Structure

MoSE_repo/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_visualization.ipynb
│
├── scripts/
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── train_utils.py
│   ├── evaluation_utils.py
│   └── visualization_utils.py
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/          # Rohdaten (GeoTIFFs, Shapefiles)
│   ├── processed/    # Vorverarbeitete Daten (z. B. Patches)
│   └── results/      # Modelle, Logs, Visualisierungen
│
├── requirements.txt  # dependencies
└── README.md         # project overview

not in the repository:

data/
├── raw/          # raw data (GeoTIFFs, shapefiles)
├── processed/    # preprocessed data (e.g. preprocessed patches)
└── results/      # models, logs, visualizations

