# Project Overview

This project contains building blocks which allow experimenting on CL pipelines. While it has been made as abstract as possible, some assumptions must be correct to use it. Everything has been built to support an additional out-of-distribution class for every task, image-specific logic has been build for the untransformed MNIST dataset, so a lot of the image-specific functions assume only one channel and assume that 0 corresponds to black.

## File Descriptions

### `src/augmenting_dataloader.py`
This file contains code for augmenting the dataset within the DataLoader with the third out-of-distribution class. It includes techniques to apply real-time data augmentation during training and defines custom transformations for input data.

### `src/embedding_measurements.py`
Contains functions used to measure confusion, drift, etc across embeddings, used to quantify experiment results.

### `src/feature_attribution.py`
Logic used to quantify various metrics on the 'attention' of the model across tasks (SHAPC values, saliencies, etc)

### `src/globals.py`
Defines global constants and variables for the project.

### `src/image_utils.py`
Includes utility functions for image augmentation and visualization.

### `src/model.py`
Houses the definitions of the CNN architectures.

### `src/ogd.py`
Houses the definition and methods of the Orthogonal Gradient Descent optimiser.

### `src/pytorch_utils.py`
Provides helper functions for PyTorch data.

### `src/training_utils.py`
Implements utility functions for training, including some loss definitions, logic to calculate parameters used for certain losses such as EWC, and some data augmentation logic. A lot of these functions are unused legacy code for experiments that were scrapped, but they remain nonetheless in case there is a desire to expand experiments in the future.

### `src/training.py`
The main script for training the model. It integrates data loading, augmentation, training loops, evaluation, and logging to orchestrate the entire training process.

### `src/visualizations.py`
Contains visualization functions for analyzing and understanding model performance. Examples include plotting embeddings and visual matrices.

### `notebooks/experiment.ipynb`
This Jupyter Notebook is used for running and visualizing experiments. It allows to interactively test models, debug training, and visualize results like metrics, losses, confusion matrices, learned features and feature embeddings.

### `README.md`
This documentation file, providing an overview of the repository, its files, and their functionalities.

## Getting Started

1. **Installation**: Clone the repository and install dependencies using `pip install -r requirements.txt`.
2. **Running Training**: Use `training.py` to train the model. Update configurations in `globals.py` if necessary.
3. **Running Experiments**: Open `experiment.ipynb` for interactive experiments.

## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
