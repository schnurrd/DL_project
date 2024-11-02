import globals
import torch
from torch.utils.data import DataLoader
import copy
from typing import List
from training_procedures import TrainingProcedure
import numpy as np
from models.CLModel import CLModel
from typing import Type, List

def get_labels(loader):
    labels = []
    for _, lab in loader:
        labels += lab.tolist()
    return torch.tensor(np.array(labels))

def c_print(*args, **kwargs):
    if globals.verbose:
        print(*args, **kwargs)

def get_features(loader):
    features = []
    for feat, _ in loader:
        features.append(feat)
        dtype = feat.dtype
    return torch.cat(features, dim=0).to(dtype=dtype)

def CL_train(model: Type[CLModel], trainloaders: List[DataLoader], testloaders: List[DataLoader], procedure: TrainingProcedure, n_epochs_base, n_epochs_CL, loss_breaks_base = [0.03], loss_breaks_CL = [0.03, 0.03]) -> CLModel:
    prevModel = None
    for i in range(globals.N_TASKS):
        net = model(globals.dataset.input_size, (i+1)*globals.CLASSES_PER_ITER)
        if prevModel is not None:
            net.CLCopy(prevModel, i)
        train_loader = trainloaders[i]
        if prevModel:
            procedure.train_CL(net, prevModel, train_loader, i, n_epochs_CL, loss_breaks_CL)
        else:
            procedure.train_base(net, train_loader, None, None, n_epochs_base, loss_breaks_base)
        c_print("ITERATION", i+1)
        c_print("ACCURACIES PER SET:")
        with torch.no_grad():
            for j in range(i+1):
                val_loader = testloaders[j]
                val_labels = get_labels(val_loader)
                pred = net(get_features(val_loader))
                sliced_pred = pred[:, j*globals.CLASSES_PER_ITER:(j+1)*globals.CLASSES_PER_ITER]
                _, predicted = torch.max(sliced_pred, 1)  # Get the class predictions
                predicted += j*globals.CLASSES_PER_ITER
                val_labels = get_labels(val_loader)
                correct = (predicted == val_labels).sum().item()  # Count how many were correct
                accuracy = correct / val_labels.size(0)  # Accuracy as a percentage
                c_print(str(accuracy), end=' ')
        c_print('\n')
        prevModel = copy.deepcopy(net)
    return net

def test(model: CLModel, testloader: DataLoader):
    features = get_features(testloader)
    labels = get_labels(testloader)
    with torch.no_grad():
        model.eval()
        pred = model(features)
        _, predicted = torch.max(pred, 1)  # Get the class predictions
        correct = (predicted == labels).sum().item()  # Count how many were correct
        accuracy = correct / labels.size(0)  # Accuracy as a percentage
    return accuracy

    