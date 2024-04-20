# Project Overview

This repository contains all the necessary components for our machine learning project focusing on wildfire prediction in Alberta. It includes data preprocessing scripts, model training and evaluation pipelines, and research notebooks for experimentation.


## Directory Structure

- **data/**: Contains all datasets used in the project, divided into raw, processed, and predictions subdirectories.
  - **raw/**: Original, spatially cropped data files containing features of interest.
  - **processed/**: Preprocessed datasets ready for machine learning models.
  - **predictions/**: Predictions from various models for use in ensemble stacking techniques.

- **models/**: Contains Python scripts and saved models for each machine learning algorithm implemented.
  - Each subdirectory includes model-specific files including scripts and saved models.

- **notebooks/**: Jupyter notebooks for final training and evaluation of models.

- **research/**: Contains exploratory notebooks and experiments.

- **scripts/**: Reusable Python scripts for data acquisition, preprocessing, ensemble methods, and visualization. These scripts are intended to be imported into notebooks or other scripts to ensure code reusability and maintainability.


## Usage

**Training and Evaluation**:
- Training and model evaluation can be performed in the respective notebook under `notebooks/`
- Notebook outputs are saved in the corresponding sub-directory in `models/`
- Notebooks pull from model specific code found in `models/` or general code found in `scripts/`


## Research and Experiments

Dataset and model exploration and experimentation. Code quality in this directory is flexible to encourage experimentation and exploration of new ideas.

