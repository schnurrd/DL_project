# CL Project

This project is used to statistically benchmark model performances for continual learning.

The script works for arbitrary models, training procedures, datasets and task amounts. It will retrain the model, benchmark its accuracy and write it to a file - a number of times equal to n_runs.

To benchmark multiple different pipelines, you can run a bash script which calls this script multiple times with different parameters.

## Arguments

Below is a description of each command-line argument that can be passed to the script.

| Argument | Short | Type | Required | Default | Description |
|----------|-------|------|----------|---------|-------------|
| `--n_runs` | - | `int` | Yes | - | Number of times to run the experiment. |
| `--tasks` | `-t` | `int` | Yes | - | Number of tasks -- must divide the total class numbere of your chosen dataset. |
| `--dataset` | `-d` | `str` | No | `MNIST` | Dataset to use. Supported values are `MNIST`. |
| `--model` | `-m` | `str` | No | `VenEtAlMLP` | Model to use. Supported values are `VenEtAlMLP`. |
| `--procedure` | `-p` | `str` | No | `sep_CE_comb_KD` | Training procedure to use. Supported values are `sep_CE_comb_KD` and `sep_CE_sep_KD`. |
| `--output_dir` | - | `str` | No | - | Directory to write results to. **Directory must exist!** |
| `--verbose` | `-v` | `bool` | No | `False` | Set to `True` for verbose training messages during each run. |
| `--loss_breaks_base` | - | `float`, `list` | Yes | - | List of loss breaks for training the base model (task 0). Provide multiple values separated by spaces. Adapt to chosen training procedure. |
| `--loss_breaks_CL` | - | `float`, `list` | Yes | - | List of loss breaks for training continual models (tasks > 0). Provide multiple values separated by spaces. Adapt to chosen training procedure. |
| `--n_epochs_base` | - | `int` | Yes | - | Maximum number of epochs to train the base model. |
| `--n_epochs_CL` | - | `int` | Yes | - | Maximum number of epochs to train continual learning (CL) models. |
| `--out_file` | - | `Path` | No | - | Path to a file where all verbose messages will be saved if specified. |
| `--optimiser` | - | `str` | No | - | Optimizer to use during training (e.g., `SGD`, `Adam`). Default is `SGD`. |
| `--lr` | - | `str` | No | - | Learning rate for the optimizer. |
| `--momentum` | - | `str` | No | - | Momentum for the optimizer (if applicable, such as with SGD). |

## Usage

Example usage:

```bash
python main.py -t 5 --n_runs 10 --n_epochs_base 10 --n_epochs_CL 15 --loss_breaks_base 0.03 --loss_breaks_CL 0.03 0.03 --dataset MNIST --procedure sep_ce_sep_kd --verbose
```

This will train the model on 5 tasks on the (default) MNIST dataset -- so, 2 classes per task. It will do it 10 times and output the accuracies to a txt file in the default results folder (found in parent folder of Code). It will train the base model for up to 10 epochs or until it reaches 0.03 loss. It will train CL models for up to 15 epochs or until they reach under 0.03 CE loss and 0.03 KD loss. It will use CE separately on the current task and KD separately for each previous task. It will be verbose and print messages during training.

## To add your own dataset

In dataset.py, update Dataset's `__init__` method to properly load your training and testing datasets. Make sure you set the input_size attribute of your dataset correctly - it corresponds to the size of one datapoint and is used during model initialization. If you download the dataset, make sure to use the `./data` folder - it is in gitignore.
The rest of the methods abstractly deal with segragating any dataset into class-incremental subsets, you don't have to change them.
Update main.py handling of the `dataset` arg to load your new dataset when passed.

## To add your own model

In the models folder, create a model which inherits the abstract `CLModel` class. In addition to the standard `forward()` method, your model must be initializable with arbitrary input size and output classes and must implement the `CLCopy()` method, which copies the parameters from the model from the previous task - i.e. a model which outputs a number of classes fewer by `CLASSES_PER_INCREMENT`.  The method is used when "extending" models with new classes - since we cannot directly extend the model, we create a new one with additional output neurons and copy the other parameters from the old one. See example in simpleModels.py.
Update main.py handling of `model` arg to load your new model when passed.

## To add your own training procedure

In the training_procedures folder, create a procedure which inherits the abstract `TrainingProcedure` class. Your new procedure must implement the two methods `train_base()` and `train_CL()` which are used to train the base model (the model for the first task, when there are still no previous models and tasks), and the next models which extend the previous ones.
Update main.py handling of the `procedure` arg to load your new procedure when passed.

## To add your own optimiser

Update main.py handling of the `optimiser` arg to load your new optimiser when passed. Follow the example of the existing ones. If your optimiser requires any new arguments, add them to the script.

## TODO

- Implement handling of different transforms for datasets
- Add "playground" section for jupyter notebook(s), which can make use of all the predefined classes here. While the scripts are useful for a fleshed out pipeline, it would be tedious to implement new modules from scratch inside the pipeline -- for that jupyter notebooks are more convenient
- Update the input_size attribute for datasets to preserve dimensions -- will likely be used in CNNs
- Add scripts which can aggregate the results files