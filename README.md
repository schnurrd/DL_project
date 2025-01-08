# Project Overview

This project contains building blocks which allow experimenting on CL pipelines. While it has been made as abstract as possible, some assumptions must be correct to use it. Everything has been built to support an additional out-of-distribution class for every task, image-specific logic has been build for the untransformed MNIST dataset, so a lot of the image-specific functions assume only one channel and assume that 0 corresponds to black.

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
This Jupyter Notebook is used for running and visualizing experiments. It allows to interactively test models, debug training, and visualize results like metrics, losses, confusion matrices, learned features and feature embeddings.

### `README.md`
This documentation file, providing an overview of the repository, its files, and their functionalities.

## Getting Started

1. **Installation**: Clone the repository and install dependencies using `pip install -r requirements.txt`.
2. **Running Training**: Use `training.py` to train the model. Update configurations in `globals.py` if necessary.
3. **Running Experiments**: Open `experiment.ipynb` for interactive experiments.

## Reproducibility

All reported results can be reproduced with the `experiment.ipynb` notebook. This notebook is mainly set up to run on a cluster that uses SLURM. Thereby the `submitit` library is used to submit and track the cluster jobs.

### Running the experiments on the cluster

The only thing that needs to be changed in this case is that in the second cell of the `experiment.ipynb` notebook, the line `partition = ''` needs to be filled out with the cluster-specific partition_name that should be used to run the jobs.

### Running the experiments locally

In this case, a few more changes need to be made. First of all, in the second cell of the `experiment.ipynb` notebook, the line `DEVICE = torch.device("cuda:0")` should be changed to `DEVICE = globals.DEVICE` to run the code on the correct device. In the notebook, this is currently hardcoded since cluster jobs are normally submitted on nodes without GPUs, while the actual execution nodes have GPUs.

Secondly, every line that looks similar to this one `ex1 = ex_parallel.submit(run_all_experiments, dataset='mnist', n_runs=5)`, where we call `ex_parallel.submit` on a function, needs to be replaced with just the function execution and the provided arguments. In the example from above the updated line would be `ex1 = run_all_experiments(dataset='mnist', n_runs=5)`. This change would then need to be applied to be applied for each experiment.

## Contribution
Feel free to contribute to the project by creating issues or submitting pull requests.

---

For further details, contact the repository maintainer.
