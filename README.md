# Project Overview

This is the code repository for our course project "Decomposing Forgetting: Insights into Class-Incremental Learning Dynamics" for the ETH Zurich Deep Learning course. The repo project contains code building blocks which allow experimenting on continual learning pipelines to identify specific subproblems that contribute to critical forgetting.

## File Descriptions

### `src/augmenting_dataloader.py`
This file contains code for augmenting the dataset within the DataLoader with the third out-of-distribution class. It includes techniques to apply real-time data augmentation during training and defines custom transformations for input data.

### `src/data_utils.py`
Houses the method used for loading datasets and splitting them into tasks of train/test/validation sets.

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

1. **Installation**: Clone the repository and install the dependencies in a python 3.10 environment using `pip install -r requirements.txt`.
2. **Downloading Datasets**: CIFAR-10 and MNIST are downloaded automatically using torch datasets when running experiments. Tiny ImageNet has to be downloaded manually. We downloaded it from http://cs231n.stanford.edu/tiny-imagenet-200.zip and preprocessed the dataset to make the structure of the val folder (the test set) identical to that of the train folder (the training set). Our preprocessed version is necessary to run the experiments for Tiny ImageNet and can be downloaded [here](https://drive.google.com/file/d/1hiQk0v9Nc0XhsLKrxGy91QvmAWJ2wp2O/view?usp=sharing). Download the zip archive and unzip the `tiny-imagenet-200` folder inside the `data` folder. Then Tiny ImageNet experiments can be run.
3. **Running Experiments**: Open `experiment.ipynb` for interactive experiments.

## Reproducibility

All reported results can be reproduced using the `experiment.ipynb` notebook. This notebook is designed to work both on a GPU cluster that utilizes SLURM and locally on the current machine. The `submitit` library is then used to submit and track the cluster jobs.

### Running the experiments on the cluster

To run experiments on a cluster, make the following changes:

1. Open the `experiment.ipynb` notebook.
2. In the second cell, replace `RUN_LOCALLY = True` with `RUN_LOCALLY = False`
3. In the second cell, locate the line `partition = ''`.
4. Replace the empty string (`''`) with the name of the cluster-specific partition (e.g., `partition = 'gpu_partition'`).

Furthermore, one needs to check if all the datasets can be downloaded on the cluster, and if this is not possible, download them locally before. The datasets will be downloaded upon the first call of the `run_experiments` function with the respective dataset, and the Tiny Imagenet needs to be downloaded separately as given in Section "Getting Started".

### Running the experiments locally

Running the experiments locally requires only one check:

1. In the second cell, ensure that `RUN_LOCALLY` is set to `True`, if this is not the case, set it to `True`

## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
