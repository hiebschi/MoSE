# MoSE Overview 
Project MoSE = Mapping of structure elements in water bodies based on drone data using machine learning

## Introduction

hier fehlt Text

## Repository Structure

MoSE_repo/  
│    
├── **notebooks**/ # run it!   
│   ├── 01_preprocessing.ipynb # load and preprocess data  
│   ├── 02_training.ipynb # implement, train, test and save model   
│   ├── 03_evaluation.ipynb # evaluate model with various metrics   
│   └── 04_visualization.ipynb # visualize results   
│     
├── **scripts**/ # (helper) functions  
│   ├── data_utils.py # load data: dataset, dataloader   
│   ├── model_utils.py # model architecture, loss function and optimizer   
│   ├── train_utils.py # training and testing loop, learning rate scheduler   
│   ├── evaluation_utils.py # calculate evaluation metrics  
│   └── visualization_utils.py # visualise image data and masks,...    
│   
├── **configs**/   
│   └── config.py # hyperparameters and paths   
│   
├── **requirements**.txt # dependencies    
└── **README.md** # project overview    
   
not in the repository:   
    
data/     
├── raw/          # raw data (GeoTIFFs, shapefiles)   
├── processed/    # preprocessed data (e.g. preprocessed patches)    
└── results/      # models, logs, visualizations    



