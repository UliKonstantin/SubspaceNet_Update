# Data Directory

This directory contains datasets for the subspaceNet_Online_learning project.

## Structure

- `datasets/`: Contains the generated trajectory datasets used for training and testing
  - Files are in HDF5 format (.h5)
  - Datasets are named according to their parameters (trajectory type, field type, etc.)
  - Large dataset files are excluded from git via .gitignore

## Dataset Generation

Datasets are generated using the trajectory generators in `simulation/runners/data.py`.
You can create new datasets by running the training scripts with appropriate configuration files. 