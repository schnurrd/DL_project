import os
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
import numpy as np
import builtins
import globals
import time
from torch.utils.data import DataLoader
from training_utils import (batch_compute_saliency_maps, 
                            mask_top_n_saliency_regions_batch, 
                            calculate_nonzero_percentage, 
                            create_nonzero_mask,
                            create_zero_mask,
                            store_additional_data,
                            calc_ewc_loss,
                            apply_mask_to_gradients,
                            store_test_embedding_centers,
                            CenterLoss)
from ogd import OrthogonalGradientDescent

from augmenting_dataloader import AugmentedOODTrainset, CutMixOODTrainset, FMixOODTrainset, SmoothMixOODTrainset, JigsawOODTrainset
from visualizations import plot_embeddings, plot_confusion_matrix
from image_utils import show_image


def train_model(net, 
                trainloader, 
                valloader, 
                verbose = False,
                report_frequency=1,
                timeout = None,
                load_path = None, 
                save_path = None, 
                epochs = 5,
                stopOnLoss = 0.03,
                ogd = False,
                ):
    """
    Used to train on first task of CL.
    For more details, see comment of train_model_CL, most of which is analogous to this function
    """

    start = time.time()
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    ITERATIONS = globals.ITERATIONS
    trainloaders = globals.trainloaders
    if not globals.ood_method:
        ds = trainloaders[0].dataset
    elif globals.ood_method == 'fmix':
        ds = FMixOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER)
        ds.display_ood_samples()
    elif globals.ood_method == 'smoothmixs':
        ds = SmoothMixOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER, mask_type = 'S')
        ds.display_ood_samples()
    elif globals.ood_method == 'smoothmixc':
        ds = SmoothMixOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER, mask_type = 'C')
        ds.display_ood_samples()
    elif globals.ood_method == 'jigsaw':
        ds = JigsawOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER)
        ds.display_ood_samples()
    else:
        ds = CutMixOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER, centered = True)
    trainloader = DataLoader(ds, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    net = net.to(DEVICE)
    lr = 0.001
    
    params = list(net.parameters())
    
    if ogd:
        optimizer = OrthogonalGradientDescent(net, optim.SGD(params, lr=lr, momentum=0.9), device=DEVICE)
    else:    
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)#, weight_decay = 0.001)
        
    model_path = load_path
    if model_path and os.path.isfile(model_path):
        # load trained model parameters from disk
        net.load_state_dict(torch.load(model_path))
    else:
        for epoch in range(epochs):
            epochCELoss = 0.0
            val_epoch_loss = 0.0
            for batch in trainloader:
                inputs, labels = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

                outputs, embeddings = net.get_pred_and_embeddings(inputs)
                ceLoss = criterion(outputs, labels) # CE Loss
                epochCELoss += ceLoss.item()
                loss = loss + ceLoss
                
                loss.backward()

                optimizer.step()

            if timeout is not None and time.time() - start > timeout:
                raise Exception("initial train timed out!")
            
            if globals.val_set_size != 0:
                net.eval()
                predicted_full = []
                all_val_labels = []
                for inputs, labels in valloader:
                    with torch.no_grad():
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = net(inputs)
                        if globals.OOD_CLASS == 1:
                            outputs = outputs[:, [i for i in range(outputs.size(1)) if (i + 1) % (CLASSES_PER_ITER+1) != 0]]
                        _, predicted = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        val_epoch_loss += loss.item()
                        optimizer.zero_grad()
                        predicted_full.extend(predicted.cpu().numpy())  # Move to CPU and convert to numpy for ease
                        all_val_labels.extend(labels.cpu().numpy())
                net.train()
                correct = sum(p == t for p, t in zip(predicted_full, all_val_labels))
                total_val_accuracy = correct / len(all_val_labels)
                if len(valloader) > 0:
                    val_epoch_loss /= len(valloader)
            epochCELoss /= len(trainloader)

            if verbose and epoch%report_frequency == 0:
                print(f"Epoch {epoch}, CE Loss: {epochCELoss:.4f}")
                if globals.val_set_size != 0:
                    print("Validation loss", val_epoch_loss, "validation accuracy", total_val_accuracy, '\n')
            if stopOnLoss is not None and epochCELoss < stopOnLoss:
                break
        if ogd:
            optimizer.update_basis(trainloaders[0].dataset)
        if save_path:
            torch.save(net.state_dict(), save_path)
        store_test_embedding_centers(net, 1)

def train_model_CL(net, 
                   prevModel, 
                   trainloader, 
                   valloader, 
                   iteration, 
                   verbose = False, 
                   n_epochs=4, 
                   validateOnAll = False,
                   full_CE = False,
                   kd_loss = 0,
                   report_frequency=1,
                   lr = 0.001,
                   momentum = 0.9,
                   stopOnLoss = 0.03,
                   stopOnValAcc = None,
                   ogd = False,
                   ):
    """
    Parameters:
    ----------
    net : torch.nn.Module
        The current model to be trained. Must implement some specific functions and attributes, take a look at model.py
    prevModel : torch.nn.Module or None
        The previously trained model, used for knowledge distillation.
    trainloader : DataLoader
        DataLoader for the training dataset.
    valloader : DataLoader
        DataLoader for the validation dataset.
    iteration : int
        The current iteration or task number in the continual learning setting.
    verbose : bool, optional (default=False)
        Whether to print detailed training logs.
    n_epochs : int, optional (default=4)
        Number of epochs to train the model.
    validateOnAll : bool, optional (default=False)
        If True, validate on all tasks seen so far instead of just the current task.
    full_CE : bool, optional (default=False)
        If True, apply cross entropy to all outputs, instead of only for last task
    kd_loss : float, optional (default=0)
        Strength of Knowledge Distillation (KD) loss to transfer knowledge from `prevModel`.
    report_frequency : int, optional (default=1)
        on how many epochs to report accuracies, confusion matrices, embeddings, etc. 
    lr : float, optional (default=0.001)
        learning rate of optimizer
    momentum : float, optional (default=0.9)
        momentum of optimizer (if applicable)
    stopOnLoss : float, optional (default=0.03)
        if not None, stop training when this cross entropy loss has been reached in training
    stopOnValAcc : float, optional (default=None)
        if not None, stop training when this accuracy has been reached during validation
    ogd : bool, optional (default=False)
        if True, use orthogonal gradient descent as opposed to SGD
    """

    torch.autograd.set_detect_anomaly = True
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    trainloaders = globals.trainloaders
    valloaders = globals.valloaders
    ood_label = (iteration+1)*globals.CLASSES_PER_ITER
    if not globals.ood_method:
        ds = trainloaders[iteration].dataset
    elif globals.ood_method == 'fmix':
        ds = FMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER)
        ds.display_ood_samples()
    elif globals.ood_method == 'smoothmixs':
        ds = SmoothMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, mask_type = 'S')
        ds.display_ood_samples()
    elif globals.ood_method == 'smoothmixc':
        ds = SmoothMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, mask_type = 'C')
        ds.display_ood_samples()
    elif globals.ood_method == 'jigsaw':
        ds = JigsawOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER)
        ds.display_ood_samples()
    else:
        ds = CutMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, centered = True)
    
    trainloader = DataLoader(ds, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    net, prevModel = net.to(DEVICE), prevModel.to(DEVICE) 
    prevModel.eval()

    ceLoss = nn.CrossEntropyLoss()
    klDivLoss = nn.KLDivLoss(reduction="batchmean")

    params = list(net.parameters())
        
    if ogd:
        optimizer = OrthogonalGradientDescent(net, optim.SGD(params, lr=lr, momentum=momentum), device=DEVICE)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)#, weight_decay = 0.001)
        
    prevModel.eval()
    epoch = 0
    for epoch in range(n_epochs): 
        epochCELoss = 0.0
        epochKDLoss = 0.0
        val_epochCELoss = 0.0
        val_epochKDLoss = 0.0
        labels_offset = (iteration if globals.OOD_CLASS == 1 else 0)
        for batch in trainloader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            '''
            if iteration == 4:
                for i, l in enumerate(labels):
                    print(l)
                    show_image(inputs[i][0])
            '''
            labels += labels_offset
            optimizer.zero_grad()

            outputs, embeddings = net.get_pred_and_embeddings(inputs)
            net.eval()
            outputsNoDropout = net(inputs)
            net.train()
            oldClassOutputs = outputsNoDropout[:, :-CLASSES_PER_ITER-globals.OOD_CLASS]
            with torch.no_grad():
                prevOutputs = prevModel(inputs)

            loss = torch.tensor(0.0, requires_grad = True)

            if full_CE:
                _ceLoss = ceLoss(outputs, labels)
            else:
                _ceLoss = ceLoss(outputs[:,-CLASSES_PER_ITER-globals.OOD_CLASS:], labels - iteration*(CLASSES_PER_ITER+globals.OOD_CLASS))
            epochCELoss += _ceLoss.item()
            loss = loss + _ceLoss

            if kd_loss != 0: # knowledge distillation loss
                _kd_loss = kd_loss*klDivLoss(F.log_softmax(oldClassOutputs, dim=-1), F.softmax(prevOutputs, dim=-1))
                epochKDLoss += _kd_loss.item()
                loss = loss + _kd_loss
            #for it in range(iteration):
            #    loss += klDivLoss(F.log_softmax(oldClassOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1), F.softmax(prevOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1))
            loss.backward()
            optimizer.step()

        if globals.val_set_size != 0:
            net.eval()
            predicted_eval_task = []
            task_val_labels = []
            for inputs, labels in valloader:
                with torch.no_grad():
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    #saliency_maps = batch_compute_saliency_maps(net, inputs, labels)
                        
                    # Mask high-saliency regions in a batch manner
                    #masked_images = mask_top_n_saliency_regions_batch(inputs, saliency_maps, 2, 8)
                    outputs = net(inputs)
                    outputsNoDropout = outputs
                    oldClassOutputs = outputsNoDropout[:, :-CLASSES_PER_ITER-globals.OOD_CLASS]
                    with torch.no_grad():
                        prevOutputs = prevModel(inputs)
                    loss = ceLoss(outputs[:,-CLASSES_PER_ITER-globals.OOD_CLASS:], labels - iteration*CLASSES_PER_ITER)
                    CELoss = loss.item()
                    val_epochCELoss += CELoss
                    #for it in range(iteration):
                    #    loss += klDivLoss(F.log_softmax(oldClassOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1), F.softmax(prevOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1))
                    loss += klDivLoss(F.log_softmax(oldClassOutputs, dim=-1), F.softmax(prevOutputs, dim=-1))
                    val_epochKDLoss += loss.item() - CELoss
                    if globals.OOD_CLASS == 1:
                        outputs_no_OOD = outputs[:, [i for i in range(outputs.size(1)) if (i + 1) % (CLASSES_PER_ITER+1) != 0]]
                    else:
                        outputs_no_OOD = outputs
                    _, predicted = torch.max(outputs_no_OOD, 1)
                    predicted_eval_task.extend(predicted.cpu().numpy())
                    task_val_labels.extend(labels.cpu().numpy())
                    optimizer.zero_grad()
        
            correct = sum(p == t for p, t in zip(predicted_eval_task, task_val_labels))
            task_val_accuracy = correct / len(task_val_labels)
            if validateOnAll:
                net.eval()
                with torch.no_grad():
                    predicted_full = []
                    all_val_labels = []
                    all_val_embeddings = []
                    for k in range(iteration+1):
                        for feat, lab in valloaders[k]:
                            feat, lab = feat.to(DEVICE), lab.to(DEVICE)
                            
                            # Get the model's predictions
                            outputs, embeddings = net.get_pred_and_embeddings(feat)
                            
                            if globals.OOD_CLASS == 1:
                                outputs = outputs[:, [i for i in range(outputs.size(1)) if (i + 1) % (CLASSES_PER_ITER+1) != 0]]
                            _, predicted = torch.max(outputs, 1)  # Assuming it's a classification task
                            
                            # Accumulate predictions and labels
                            predicted_full.extend(predicted.cpu().numpy())  # Move to CPU and convert to numpy for ease
                            all_val_labels.extend(lab.cpu().numpy())
                            all_val_embeddings.extend(embeddings.cpu().numpy())
                    correct = sum(p == t for p, t in zip(predicted_full, all_val_labels))
                    total_val_accuracy = correct / len(all_val_labels)
                    all_val_labels = np.array(all_val_labels)
                    all_val_embeddings = np.array(all_val_embeddings)
                net.train()
            net.train()

            if len(valloader) > 0:
                val_epochCELoss /= len(valloader)
                val_epochKDLoss /= len(valloader)

        epochCELoss /= len(trainloader)
        epochKDLoss /= len(trainloader)
        breakCondition = True
        if stopOnLoss is not None:
            breakCondition = epochCELoss < stopOnLoss
        if stopOnValAcc is not None:
            breakCondition = breakCondition and task_val_accuracy > stopOnValAcc
        if stopOnLoss is None and stopOnValAcc is None:
            breakCondition = False
        #breakCondition = task_val_accuracy > 0.925
        if verbose and epoch%report_frequency == 0:
            print("Epoch", epoch, f" CELoss: {epochCELoss:.4f}, KLLoss: {epochKDLoss:.4f}")
            print("Fraction of nonzero parameters", calculate_nonzero_percentage(net))
            if globals.val_set_size != 0:
                print("Validation losses:", val_epochCELoss, val_epochKDLoss)
                print("Validation accuracy (for last task)", task_val_accuracy)
                print("Total validation accuracy", total_val_accuracy)
                if validateOnAll and globals.dataset == 'mnist':# and breakCondition:
                    plot_confusion_matrix(predicted_full, all_val_labels, list(range(CLASSES_PER_ITER*(iteration+1))))
                    plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, None)
            print('\n')
        if breakCondition:
            break
    store_test_embedding_centers(net, iteration+1)
    if ogd:
        optimizer.update_basis(trainloaders[iteration].dataset)

    if globals.dataset == 'mnist' and verbose and globals.val_set_size != 0:
        plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, net.prev_test_embedding_centers)