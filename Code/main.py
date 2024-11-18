import argparse
import globals
import dataset as ds
import models
import os
import sys
import time

from pathlib import Path
from experiment import CL_train, test
from training_procedures.sep_CE_comb_KD import sep_CE_comb_KD
from training_procedures.sep_CE_sep_KD import sep_CE_sep_KD

from torch import optim
import torch

def handle_args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs",  required=True, type=int, help="Number of times to run experiment")
    parser.add_argument("--tasks", "-t", required=True, type=int, help="Number of tasks")
    parser.add_argument("--dataset", "-d", required=False, type=str, help="Dataset to use. Supported are (MNIST). Default is MNIST.")
    parser.add_argument("--model", "-m", required=False, type=str, help="Model to use. Supported are (VenEtAlMLP). Default is VenEtAlMLP")
    parser.add_argument("--procedure", "-p", required=False, type=str, help="Training procedure. Supported are (sep_CE_comb_KD, sep_CE_sep_KD). Default is sep_CE_comb_KD.")
    parser.add_argument("--output_dir", required=False, type=str, help="Directory to write results to. Directory must exist!")
    parser.add_argument("--verbose", "-v", action='store_true', required=False, help="Set to true for verbose training on every run")
    parser.add_argument("--loss_breaks_base", type=float, nargs='+', required=True, help="Set the loss breaks for the training of the base model (task 0)")
    parser.add_argument("--loss_breaks_CL", type=float, nargs='+', required=True, help="Set the loss breaks for the training of continual models (tasks > 0)")
    parser.add_argument("--n_epochs_base", type=int, required=True, help="Maximum number of epochs to train base model for")
    parser.add_argument("--n_epochs_CL", type=int, required=True, help="Maximum number of epochs to train CL models for")
    parser.add_argument("--out_file", type=Path, required=False, help="If specified, will output all verbose messages to a file")
    parser.add_argument("--optimiser", type=str, required=False, help="Optimiser to use during training")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate for optimiser")
    parser.add_argument("--momentum", type=float, required=False, help="Momentum for optimiser")
    args = parser.parse_args()

    return args

#malkata dreybi
def main():
    args = handle_args_parsing()

    globals.N_TASKS = args.tasks
    result_name = ""
    
    if args.dataset:
        if args.dataset.upper() in ["MNIST", "CIFAR10"]:
            dataset = args.dataset
            if 10%globals.N_TASKS != 0:
                raise Exception("Number of tasks must divide total number of classes")
            globals.CLASSES_PER_ITER = int(10/globals.N_TASKS)
        else:
            raise NotImplementedError("Unsupported dataset")
    else:
        dataset = "MNIST"
        if 10%globals.N_TASKS != 0:
            raise Exception("Number of tasks must divide total number of classes")
        globals.CLASSES_PER_ITER = int(10/globals.N_TASKS)
    result_name += dataset + '_'
    globals.dataset = ds.Dataset(dataset)

    if args.model:
        mod = args.model
        if mod.upper() == 'VENETALMLP':
            model = models.VenEtAlMLP
            result_name += 'VenEtAlMLP_'
        elif mod.upper() == 'VENETALMLPPERTASK':
            model = models.VenEtAlMLPPerTask
            result_name += 'VenEtAlMLPPerTask'
        else:
            raise NotImplementedError("Unsupported model type")
    else:
        model = models.VenEtAlMLP
        result_name += 'VenEtAlMLP_'

    result_name += str(globals.N_TASKS) + "tasks_"

    if args.procedure:
        proc = args.procedure
        if proc.upper() == 'SEP_CE_COMB_KD':
            trainingProcedure = sep_CE_comb_KD()
            result_name += 'sep_CE_comb_KD_'
        elif proc.upper() == 'SEP_CE_SEP_KD':
            trainingProcedure = sep_CE_sep_KD()
            result_name += 'sep_CE_sep_KD_'
        else:
            raise NotImplementedError("Unsupported training procedure")
    else:
        trainingProcedure = sep_CE_comb_KD()
        result_name += 'sep_CE_comb_KD_'

    if args.output_dir:
        output_dir = args.output_dir
    else:
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        output_dir = parent_dir + '/results'

    if args.verbose:
        globals.verbose = args.verbose

    if args.out_file:
        file = open(args.out_file, "a")
        sys.stdout = file  # Redirect stdout to the file
    
    dummy_param = [torch.nn.Parameter(torch.empty(0))] # the parameters are passed later in the training procedure
    if args.optimiser:
        optim_name = args.optimiser
        if args.optimiser.upper() == 'SGD':
            if args.lr:
                lr = args.lr
            else:
                lr = 0.001
            if args.momentum:
                m = args.momentum
            else:
                m = 0.9
            globals.optimiser = optim.SGD(dummy_param, lr=lr, momentum=m)
            result_name += "SGD_" + str(lr) + "_" + str(m)
        elif args.optimiser.upper() == "ADAM":
            if args.lr:
                lr = args.lr
            else:
                lr = 0.001
            result_name += "ADAM_" + str(lr)
            globals.optimiser = optim.Adam(dummy_param, lr=lr)
        else:
            raise NotImplementedError("Unsupported optimiser")
    else:
        lr = 0.001
        m = 0.9
        globals.optimiser = optim.SGD(dummy_param, lr=lr, momentum=m)
        result_name += "SGD_" + str(lr) + "_" + str(m) + "_"
    result_name += str(args.n_epochs_base) + "_" + str(args.n_epochs_CL)

    trainloaders, testloaders = globals.dataset.getLoadersPerTask()
    trainloader, testloader = globals.dataset.getFullLoaders()

    for i in range(args.n_runs):
        start_time = time.time()
        print("Starting run " + str(i+1))
        net = CL_train(model, trainloaders, testloaders, trainingProcedure, args.n_epochs_base, args.n_epochs_CL,  args.loss_breaks_base, args.loss_breaks_CL)
        with open(output_dir + '/' + result_name + '.txt', 'a') as output_file:
            accuracy = test(net, testloader)
            output_file.write(str(accuracy) + '\n')
        duration = time.time() - start_time
        minutes = duration // 60
        seconds = int(duration % 60)
        print("Finished run " + str(i+1) + " in " + str(minutes) + " minutes, " + str(seconds) + " seconds with accuracy " + str(accuracy))

if __name__ == "__main__":
    main()