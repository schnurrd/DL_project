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
- Default is with a storage size of 200 random sample gradients per task (max_basis_size=200)
- Default is to let the buffer grow over the different tasks (reduce_basis=False) 

**Comparison** (Performance averaged across 10 runs, with full_CE false, GPT build table from comments inside my ogd pull request so don't trust it 100% but I checked most of the values and everything seems to be correct):
| Configuration                               | Averaged SHAPC (↓)       | Mean Accuracy (±Std)     | Mean Total Confusion (±Std) | Mean Intra-Phase Confusion (±Std) | Mean Per-Task Confusion (±Std) | Mean Embedding Drift (±Std)    | Mean Attention Drift (±Std)         | Mean Attention Spread (±Std)    |
|---------------------------------------------|---------------------------|--------------------------|-----------------------------|-----------------------------------|--------------------------------|--------------------------------|--------------------------------------|--------------------------------|
| Baseline                                    | 2.5670e-06               | 0.65348 (0.03111)        | 0.43270 (0.01230)          | 0.43021 (0.01218)                 | 0.06989 (0.00539)              | 5.3576 (0.31395)               | 2.3871e-06 (2.2904e-07)              | 48.08096 (1.00881)             |
| Baseline 2                         | 2.5680e-06               | 0.60748 (0.04958)        | 0.43732 (0.01030)          | 0.43479 (0.01012)                 | 0.07184 (0.00535)              | 5.60164 (0.31787)              | 2.3885e-06 (2.0199e-07)              | 47.54174 (1.43070)             |
| Baseline 3                                  | 2.5982e-06               | 0.61216 (0.05226)        | 0.43230 (0.00965)          | 0.42976 (0.00979)                 | 0.07000 (0.00375)              | 5.49204 (0.27083)              | 2.4421e-06 (1.8576e-07)              | 47.56774 (1.38656)             |
| With ogd & reduce_basis=True                | 1.8457e-06               | 0.65065 (0.03858)        | 0.41011 (0.00955)          | 0.40792 (0.00934)                 | 0.06131 (0.00320)              | 5.05795 (0.33517)              | 1.9036e-06 (2.2346e-07)              | 49.76607 (1.61156)             |
| With ogd & reduce_basis=False               | 9.5078e-07               | 0.74067 (0.02976)        | 0.39415 (0.01151)          | 0.39172 (0.01134)                 | 0.06067 (0.00327)              | 4.07636 (0.22955)              | 1.0181e-06 (1.2154e-07)              | 53.35429 (1.77690)             |
| With ogd & reduce_basis=False (2nd exec.)   | 1.0607e-06               | 0.72012 (0.02208)        | 0.40270 (0.00962)          | 0.40020 (0.00944)                 | 0.06201 (0.00424)              | 4.20050 (0.21273)              | 1.0651e-06 (1.4154e-07)              | 51.04283 (1.99993)             |
## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
