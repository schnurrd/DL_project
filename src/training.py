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
                            apply_mask_to_gradients,
                            store_test_embedding_centers,
                            CenterLoss)
from ogd import OrthogonalGradientDescent

from augmenting_dataloader import JointTrainingNoOODTrainset, CutMixOODTrainset, FMixOODTrainset, SmoothMixOODTrainset, JigsawOODTrainset
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
                optimiser_type='sgd',
                patience=None,
                plotting = False,
                ogd_basis_size=200,
                only_output_layer=False,
                classes_per_iter=None
                ):
    """
    Used to train on first task of CL.
    For more details, see comment of train_model_CL, most of which is analogous to this function
    """
    if patience is not None:
        bestModel = None
        bestLoss = 9999999
        epochs_without_improvement = 0
    start = time.time()
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    ITERATIONS = globals.ITERATIONS
    trainloaders = globals.trainloaders
    if not globals.ood_method:
        if not only_output_layer:
            ds = trainloaders[0].dataset
        else:
            ds = globals.trainset
    elif only_output_layer:
        ds = JointTrainingNoOODTrainset(classes_per_iter)
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
    if only_output_layer:
        lr = 0.003 #decided not to change
        for name, module in net.named_children(): #remove dropouts
            if 'dr' in name:
                setattr(net, name, nn.Identity())
    else:
        lr = 0.003
    if only_output_layer:
        # Freeze all parameters
        for param in net.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of the output layer
        for param in net.output_layer.parameters():
            param.requires_grad = True

        # Pass only the output layer parameters to the optimizer
        params = list(net.output_layer.parameters())
    else:
        params = list(net.parameters())
    
    if optimiser_type=='ogd':
        optimizer = OrthogonalGradientDescent(net, optim.SGD(params, lr=lr, momentum=0.8), max_basis_size=ogd_basis_size, device=DEVICE)
    elif optimiser_type=='sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.8)#, weight_decay = 0.001)
    elif optimiser_type=='adam':
        optimizer = optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError("Unsupported optimiser type")
        
    model_path = load_path
    if model_path and os.path.isfile(model_path):
        # load trained model parameters from disk
        net.load_state_dict(torch.load(model_path))
    else:
        for epoch in range(epochs):
            epochCELoss = 0.0
            val_epoch_loss = 0.0
            epochCELoss_no_OOD = 0.0
            ood_label = globals.CLASSES_PER_ITER
            for batch in trainloader:
                inputs, labels = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

                outputs, embeddings = net.get_pred_and_embeddings(inputs)
                ceLoss = criterion(outputs, labels) # CE Loss
                if globals.OOD_CLASS == 1:
                    with torch.no_grad():
                        mask = labels != ood_label
                        if labels[mask].numel() != 0:
                            epochCELoss_no_OOD += criterion(outputs[mask], labels[mask]).item()
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
            epochCELoss_no_OOD /= len(trainloader)

            if verbose and epoch%report_frequency == 0:
                print(f"Epoch {epoch}, CE Loss: {epochCELoss:.4f}, CE Loss (no OOD): {epochCELoss_no_OOD:.4f}")
                if globals.val_set_size != 0:
                    print("Validation loss", val_epoch_loss, "validation accuracy", total_val_accuracy, '\n')
            if stopOnLoss is not None:
                breakCondition = epochCELoss_no_OOD < stopOnLoss
                if breakCondition:
                    break
            if patience is not None:
                if val_epoch_loss < bestLoss:
                    bestLoss = val_epoch_loss
                    bestModel = net
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        net = bestModel
                        break
        if optimiser_type=='ogd':
            optimizer.update_basis(trainloaders[0].dataset)
        if save_path:
            torch.save(net.state_dict(), save_path)
        store_test_embedding_centers(net, 1)
        return net

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
                   lr = 0.003,
                   momentum = 0.8,
                   stopOnLoss = 0.03,
                   stopOnValAcc = None,
                   optimiser_type = 'sgd',
                   plotting = False,
                   patience = None,
                   ogd_basis_size=200
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
    lr : float, optional (default=0.003)
        learning rate of optimizer
    momentum : float, optional (default=0.0)
        momentum of optimizer (if applicable)
    stopOnLoss : float, optional (default=0.03)
        if not None, stop training when this cross entropy loss has been reached in training (will use validation loss if possible)
    stopOnValAcc : float, optional (default=None)
        if not None, stop training when this accuracy has been reached during validation
    optimiser_type : string, optional (default='sgd')
        the type of optimiser used, supported are 'sgd', 'ogd', 'adam'
    plotting : bool, optional (default=False)
        if true and if verbose is true, will also attempt to plot confusion matrices / embeddings
    patience : int, optional (default=None)
        if set, will apply early stopping with this patience
    ogd_basis_size: int, optional (default=200)
        if set and if optimiser_type is 'ogd', sets the max basis size for orthogonal gradient descent
    """
    torch.autograd.set_detect_anomaly = True
    if patience is not None:
        bestModel = None
        bestLoss = 9999999
        epochs_without_improvement = 0
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    trainloaders = globals.trainloaders
    valloaders = globals.valloaders
    ood_label = (iteration+1)*(globals.CLASSES_PER_ITER+1) - 1
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
        
    if optimiser_type == 'ogd':
        optimizer = OrthogonalGradientDescent(net, optim.SGD(params, lr=lr, momentum=momentum), max_basis_size=ogd_basis_size, device=DEVICE)
    elif optimiser_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)#, weight_decay = 0.001)
    elif optimiser_type == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError("Unsupported optimiser type")
        
    prevModel.eval()
    epoch = 0
    for epoch in range(n_epochs): 
        epochCELoss = 0.0
        epochCELoss_no_OOD = 0.0
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
                if globals.OOD_CLASS == 1:
                    mask = labels != ood_label
                    with torch.no_grad():
                        if labels[mask].numel() != 0:
                            epochCELoss_no_OOD += ceLoss(outputs[mask], labels[mask] - iteration*(CLASSES_PER_ITER+globals.OOD_CLASS)).item()
            else:
                _ceLoss = ceLoss(outputs[:,-CLASSES_PER_ITER-globals.OOD_CLASS:], labels - iteration*(CLASSES_PER_ITER+globals.OOD_CLASS))
                if globals.OOD_CLASS == 1:
                    mask = labels != ood_label
                    with torch.no_grad():
                        if labels[mask].numel() != 0:
                            epochCELoss_no_OOD += ceLoss(outputs[mask,-CLASSES_PER_ITER-globals.OOD_CLASS:], labels[mask] - iteration*(CLASSES_PER_ITER+globals.OOD_CLASS)).item()
            epochCELoss += _ceLoss.item()
            loss = loss + _ceLoss

            if kd_loss != 0: # knowledge distillation loss
                T = 2
                _kd_loss = kd_loss*klDivLoss(F.log_softmax(oldClassOutputs/T, dim=-1), F.softmax(prevOutputs/T, dim=-1))*T*T
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

            if len(valloader) > 0:
                val_epochCELoss /= len(valloader)
                val_epochKDLoss /= len(valloader)

        epochCELoss /= len(trainloader)
        epochKDLoss /= len(trainloader)
        epochCELoss_no_OOD /= len(trainloader)
        breakCondition = True
        if stopOnLoss is not None:
            if globals.OOD_CLASS == 1:
                breakCondition = epochCELoss_no_OOD < stopOnLoss
            else:
                breakCondition = epochCELoss < stopOnLoss
        if stopOnValAcc is not None:
            breakCondition = breakCondition and task_val_accuracy > stopOnValAcc
        if stopOnLoss is None and stopOnValAcc is None:
            breakCondition = False
        #breakCondition = task_val_accuracy > 0.925
        if verbose and epoch%report_frequency == 0:
            print("Epoch", epoch, f" CELoss: {epochCELoss:.4f}, KLLoss: {epochKDLoss:.4f}, CELoss (no OOD): {epochCELoss_no_OOD:.4f}")
            print("Fraction of nonzero parameters", calculate_nonzero_percentage(net))
            if globals.val_set_size != 0:
                print("Validation losses:", val_epochCELoss, val_epochKDLoss)
                print("Validation accuracy (for last task)", task_val_accuracy)
                print("Total validation accuracy", total_val_accuracy)
                if validateOnAll and plotting:# and breakCondition:
                    plot_confusion_matrix(predicted_full, all_val_labels, list(range(CLASSES_PER_ITER*(iteration+1))))
                    plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, None)
            print('\n')
        if breakCondition:
            break
        if patience is not None:
            if val_epochCELoss < bestLoss:
                bestLoss = val_epochCELoss
                bestModel = net
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    net = bestModel
                    break
    store_test_embedding_centers(net, iteration+1)
    if optimiser_type == 'ogd':
        optimizer.update_basis(trainloaders[iteration].dataset)

    if plotting and verbose and globals.val_set_size != 0:
        plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, net.prev_test_embedding_centers)
    return net