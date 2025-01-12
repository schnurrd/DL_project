# Project Overview

This project contains building blocks which allow experimenting on CL pipelines.

## File Descriptions

### `src/augmenting_dataloader.py`
This file contains code for augmenting the dataset within the DataLoader with the third out-of-distribution class. It includes techniques to apply real-time data augmentation during training and defines custom transformations for input data.

### `src/embedding_measurements.py`
Contains functions used to measure confusion, drift, etc across embeddings, used to quantify experiment results. Note that some measurements, such as embedding drift, ultimately were not used in the project report.

### `src/feature_attribution.py`
Logic used to quantify various metrics on the 'attention' of the model across tasks (SHAPC values, saliencies, etc)

### `src/globals.py`
Defines global constants and variables for the project.

### `src/image_utils.py`
Includes utility functions for image augmentation and visualization, most were used during debugging and testing.

### `src/model.py`
Houses the definitions of the CNN architectures.

### `src/ogd.py`
Houses the definition and methods of the Orthogonal Gradient Descent optimiser. Note: This optimizer has been deprecated and was not used for the final report. We used it in preliminary experiments.

### `src/pytorch_utils.py`
Provides helper functions for PyTorch data.

### `src/training_utils.py`
Implements utility functions for training, including some loss definitions and logic to store parameters for some measurements. A lot of these functions are unused legacy code for experiments that were scrapped, but they remain nonetheless in case there is a desire to expand experiments in the future.

### `src/training.py`
The main script for training the model. It integrates data loading, augmentation, training loops, evaluation, and logging to orchestrate the entire training process.

### `src/visualizations.py`
Contains visualization functions for analyzing and understanding model performance. Examples include plotting embeddings and confusion matrices.

### `notebooks/experiment.ipynb`
This Jupyter Notebook is used for running and visualizing experiments. It allows to interactively test models, debug training, and visualize results like metrics, losses, confusion matrices, learned features and feature embeddings. This is done with the help of a `run_experiments()` method defined inside the notebook, which calls the `train_model()` and `train_model_CL()` methods from `src/training.py`. For more details on the different parameters used to reproduce the different configurations, take a look at the description of `train_model_CL()` inside `src/training.py`.

### `README.md`
This documentation file, providing an overview of the repository, its files, and their functionalities.

## Getting Started

1. **Installation**: Clone the repository and install dependencies using `pip install -r requirements.txt`.
2. **Downloading Datasets**: CIFAR-10 and MNIST are downloaded automatically using torch datasets when running experiments. Tiny ImageNet has to be downloaded manually. We downloaded it from http://cs231n.stanford.edu/tiny-imagenet-200.zip and preprocessed the dataset to make the structure of the val folder (the test set) identical to that of the train folder (the training set). Our preprocessed version is necessary to run the experiments for Tiny ImageNet and can be downloaded [here](https://drive.google.com/file/d/1hiQk0v9Nc0XhsLKrxGy91QvmAWJ2wp2O/view?usp=sharing). Download the zip archive and unzip the `tiny-imagenet-200` folder inside the `data` folder. Then Tiny ImageNet experiments can be ran.
3. **Running Experiments**: Open `experiment.ipynb` for interactive experiments.

## Reproducibility

All reported results can be reproduced using the `experiment.ipynb` notebook. This notebook is designed primarily for use on a GPU cluster that utilizes SLURM. The `submitit` library is then used to submit and track the cluster jobs.

### Running the experiments on the cluster

To run experiments on a cluster, make the following change:

1. Open the `experiment.ipynb` notebook.
2. In the second cell, locate the line `partition = ''`.
3. Replace the empty string (`''`) with the name of the cluster-specific partition (e.g., `partition = 'gpu_partition'`).

No other modifications are necessary to run the experiments on the cluster.

### Running the experiments locally

Running the experiments locally requires two additional adjustments:

1. **Device Configuration**:
   - In the second cell of the `experiment.ipynb` notebook, change:
     ```python
     DEVICE = torch.device("cuda:0")
     ```
     to:
     ```python
     DEVICE = globals.DEVICE
     ```
     This ensures the code uses the correct device configuration.

2. **Updating Function Calls**:
   - Lines that call `ex_parallel.submit` to submit jobs must be replaced with direct function calls. For example:
     ```python
     ex1 = ex_parallel.submit(run_all_experiments, dataset='mnist', n_runs=5)
     ```
     should be updated to:
     ```python
     ex1 = run_all_experiments(dataset='mnist', n_runs=5)
     ```
   - Apply this modification to every instance where `ex_parallel.submit` is used in the notebook.

## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
