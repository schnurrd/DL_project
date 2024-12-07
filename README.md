# Project Overview

This project contains building blocks which allow experimenting on CL pipelines. While it has been made as abstract as possible, some assumptions must be correct to use it. Everything has been built to support an additional out-of-distribution class for every task, image-specific logic has been build for the untransformed MNIST dataset, so a lot of the image-specific functions assume only one channel and assume that 0 corresponds to black.

## File Descriptions

### `augmenting_dataloader.py`
This file contains code for augmenting the dataset within the DataLoader with the third out-of-distribution class. It includes techniques to apply real-time data augmentation during training and defines custom transformations for input data.

### `experiment.ipynb`
This Jupyter Notebook is used for running and visualizing experiments. It allows you to interactively test models, debug training, and visualize results like metrics, losses, confusion matrices, learned features and feature embeddings.

### `globals.py`
Defines global constants and variables for the project. Importantly -- if globals.OOD_CLASS is set to 1, then an additional class is added for every task which generates OOD data (logic for that is in augmenting_dataloader)

### `image_utils.py`
Includes utility functions for image augmentation and visualization, mostly used for generating OOD samples.

### `model.py`
Houses the definition of the neural network architecture.

### `pytorch_utils.py`
Provides helper functions for PyTorch data.

### `training_utils.py`
Implements utility functions for training, including some loss definitions, logic to calculate parameters used for certain losses such as EWC, and some data augmentation logic used for building a buffer during training. These functions help streamline the training loop in `training.py`.

### `training.py`
The main script for training the model. It integrates data loading, augmentation, training loops, evaluation, and logging to orchestrate the entire training process.

### `visualizations.py`
Contains visualization functions for analyzing and understanding model performance. Examples include plotting embeddings and visual matrices.

### `README.md`
This documentation file, providing an overview of the repository, its files, and their functionalities.

## Getting Started

1. **Installation**: Clone the repository and install dependencies using `pip install -r requirements.txt`.
2. **Running Training**: Use `training.py` to train the model. Update configurations in `globals.py` if necessary.
3. **Running Experiments**: Open `experiment.ipynb` for interactive experiments.

## Notes Regarding Orthogonal Gradient Descent

Notes from the authors in their paper:
- OGD-GTL slightly outperforms OGD_AVE and OGD-ALL
- They chose batch size 10 and a learning rate of 10^-3
- Network is a three-layer MLP with 100 hidden units in two layers and 10 logit outputs. Every layer except the final one uses ReLU activation. The loss is Softmax cross-entropy, and the optimizer is stochastic gradient descent.
- Storage size for their experiments was set to 200

Notes on my implementation:
- I implemented the OGD-GTL variant

## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
