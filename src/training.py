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

from augmenting_dataloader import AugmentedOODTrainset, CutMixOODTrainset, FMixOODTrainset, SmoothMixOODTrainset
from visualizations import plot_embeddings, plot_confusion_matrix
from image_utils import show_image

centerLoss = None
def build_buffer_data(inputs, labels, model, ood_label):
    #masked_images = mask_region_with_low_saliency(model, inputs, labels, 14)
    saliency_maps = batch_compute_saliency_maps(model, inputs, labels)
    masked_images = mask_top_n_saliency_regions_batch(inputs, saliency_maps, 2, 10)
    #return masked_images, torch.full_like(labels, ood_label)
    return masked_images, labels

def train_model(net, 
                trainloader, 
                valloader, 
                verbose = False,
                report_frequency=1,
                timeout = None,
                load_path = None, 
                save_path = None, 
                epochs = 5,
                l1_loss = 0, 
                center_loss = 0,
                freezeCenterLossCenters = None,
                stopOnLoss = 0.03,
                ogd = False,
                ):
    """
    Used to train on first task of CL.
    For more details, see comment of train_model_CL, most of which is analogous to this function
    """
    global centerLoss

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
    else:
        ds = CutMixOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER, centered = True)
    trainloader = DataLoader(ds, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    net = net.to(DEVICE)
    lr = 0.001
    
    params = list(net.parameters())
    
    centerLossLr = 0.005
    if center_loss > 0:
        centerLoss = CenterLoss(num_classes=(CLASSES_PER_ITER+globals.OOD_CLASS)*ITERATIONS, feat_dim=net.n_embeddings, use_gpu=True)
        if freezeCenterLossCenters is not None:
            pc = torch.stack(net.prev_train_embedding_centers)
            for k, t in enumerate(pc):
                #print("setting", k)
                centerLoss.centers.data[k] = t
        params += list(centerLoss.parameters())
    
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
            epoch_center_loss = 0.0
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

                if l1_loss != 0: # L1 loss
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    loss += l1_loss * l1_norm

                if center_loss != 0: # Center Loss -- see https://github.com/KaiyangZhou/pytorch-center-loss 
                    centerLossVal = centerLoss(embeddings, labels) * center_loss
                    epoch_center_loss += centerLossVal.item()
                    loss += centerLossVal
                
                loss.backward()
                if center_loss != 0: # custom logic for centerloss because it is implemented as a pytorch module
                    centerLoss.centers.grad.zero_()
                    for param in centerLoss.parameters():
                        # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                        param.grad.data *= (centerLossLr / (center_loss * lr))

                optimizer.step()

            if timeout is not None and time.time() - start > timeout:
                raise Exception("initial train timed out!")
            
            net.eval()
            for inputs, labels in valloader:
                with torch.no_grad():
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item()
                    optimizer.zero_grad()
            net.train()
            
            val_epoch_loss /= len(valloader)

            epochCELoss /= len(trainloader)
            epoch_center_loss /= (len(trainloader))

            if verbose and epoch%report_frequency == 0:
                print(f"Epoch {epoch}, CE Loss: {epochCELoss:.4f}, center loss: {epoch_center_loss:.4f}")
                print("Validation loss", val_epoch_loss)
                print("Fraction of nonzero parameters", calculate_nonzero_percentage(net), '\n')
            if stopOnLoss is not None and epochCELoss < stopOnLoss:
                break
        if ogd:
            optimizer.update_basis(trainloader.dataset)
        if save_path:
            torch.save(net.state_dict(), save_path)
        store_additional_data(net, trainloader.dataset, 1) # update fisher information, etc
        store_test_embedding_centers(net, 1)

def train_model_CL(net, 
                   prevModel, 
                   trainloader, 
                   valloader, 
                   iteration, 
                   verbose = False, 
                   n_epochs=4, 
                   validateOnAll = False, 
                   freeze_nonzero_params = False,
                   full_CE = False,
                   l1_loss = 0, 
                   ewc_loss = 0, 
                   kd_loss = 0,
                   distance_loss = 0, 
                   center_loss = 0,
                   param_reuse_loss = 0,
                   freezeCenterLossCenters = None, 
                   report_frequency=1,
                   lr = 0.001,
                   momentum = 0.9,
                   maxGradNorm = None,
                   stopOnLoss = 0.03,
                   stopOnValAcc = None,
                   timeout=None,
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
    freeze_nonzero_params : bool, optional (default=False)
        If True, freeze model parameters with non-zero values so that they are not trained.
        Used to run some experiments in combination with L1 loss, where possibly the model uses sparse parameters for the
        initial tasks, leaving more parameters available to tune for consecutive tasks
    full_CE : bool, optional (default=False)
        If True, apply cross entropy to all outputs, instead of only for last task
    l1_loss : float, optional (default=0)
        Strength of L1 regularization applied to the model's weights to encourage sparsity.
    ewc_loss : float, optional (default=0)
        Strength of Elastic Weight Consolidation (EWC) regularization to retain knowledge from previous tasks.
    kd_loss : float, optional (default=0)
        Strength of Knowledge Distillation (KD) loss to transfer knowledge from `prevModel`.
    distance_loss : float, optional (default=0)
        Strength of the loss term that encourages distances between embeddings of classes for current task
        and stored embedding centers of previous classes
    center_loss : float, optional (default=0)
        Strength of the center loss to enforce tighter clusters in the embedding space.
    param_reuse_loss : float, optional (default=0)
        Strength of param reuse loss - a customized loss which penalizes the model when it uses parameters which were important for previous tasks
    freezeCenterLossCenters : torch.Tensor or None, optional (default=None)
        Use stored center loss centers; if None, centers will be computed during training.
        Only works if first running some training without center loss on current task
    report_frequency : int, optional (default=1)
        on how many epochs to report accuracies, confusion matrices, embeddings, etc. 
    lr : float, optional (default=0.001)
        learning rate of optimizer
    momentum : float, optional (default=0.9)
        momentum of optimizer (if applicable)
    maxGradNorm : float, optional (default=None)
        if not None, gradients will be clipped to this norm
    stopOnLoss : float, optional (default=0.03)
        if not None, stop training when this cross entropy loss has been reached in training
    stopOnValAcc : float, optional (default=None)
        if not None, stop training when this accuracy has been reached during validation
    timeout : float, optional (default=None)
        if not None, raise an exception when training goes longer than this value in seconds (checked every epoch)
    """
    

    global centerLoss

    start = time.time()
    torch.autograd.set_detect_anomaly = True
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    trainloaders = globals.trainloaders
    valloaders = globals.valloaders

    if not globals.ood_method:
        ds = trainloaders[iteration].dataset
    elif globals.ood_method == 'fmix':
        ds = FMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER)
        ds.display_ood_samples()
    elif globals.ood_method == 'smoothmixs':
        ds = SmoothMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, mask_type = 'S')
        ds.display_ood_samples
    elif globals.ood_method == 'smoothmixc':
        ds = SmoothMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, mask_type = 'C')
        ds.display_ood_samples
    else:
        ds = CutMixOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER, centered = True)
    
    trainloader = DataLoader(ds, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    net, prevModel = net.to(DEVICE), prevModel.to(DEVICE) 
    prevModel.eval()

    ceLoss = nn.CrossEntropyLoss()
    klDivLoss = nn.KLDivLoss(reduction="batchmean")

    params = list(net.parameters())
    
    centerLossLr = 0.005
    if center_loss > 0:
        if freezeCenterLossCenters is not None:
            pc = torch.stack(net.prev_train_embedding_centers)
            ind = 0 # must account for OOD class centers
            for k, t in enumerate(pc):
                #print("setting", ind)
                centerLoss.centers.data[ind] = t
                ind  += 1
                if globals.OOD_CLASS == 1 and (k+1)%CLASSES_PER_ITER == 0:
                    ind += 1
        params += list(centerLoss.parameters())
        
    if ogd:
        optimizer = OrthogonalGradientDescent(net, optim.SGD(params, lr=lr, momentum=momentum), device=DEVICE)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)#, weight_decay = 0.001)
        
    prevModel.eval()
    epoch = 0
    if freeze_nonzero_params:
        nonzero_masks = create_nonzero_mask(net)
    if l1_loss:
        zero_masks = create_zero_mask(net) # l1 loss is only applied to parameters which have been zeroed out by previous model. Assuming nonzero parameters are important for past tasks
    for epoch in range(n_epochs): 
        epochCELoss = 0.0
        epochKDLoss = 0.0
        epochL1Loss = 0.0
        epochEWCLoss = 0.0
        epochCenterLoss = 0.0
        epochDistanceLoss = 0.0
        epochParamReuseLoss = 0.0
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

            if l1_loss != 0: # L1 loss
                _l1_loss = torch.tensor(0.0, requires_grad = True)
                for name, p in net.named_parameters():
                    l1_norm = torch.sum(zero_masks[name]*p.abs())
                    _l1_loss = _l1_loss + l1_loss * l1_norm
                epochL1Loss += _l1_loss.item()
                loss += _l1_loss
            
            if ewc_loss != 0: # elastic weight consolidation loss
                _ewc_loss = torch.tensor(0.0, requires_grad = True)
                _ewc_loss = _ewc_loss + ewc_loss * calc_ewc_loss(net)
                epochEWCLoss += _ewc_loss.item()
                loss = loss + _ewc_loss

            if distance_loss != 0: # custom loss, tries to maximize distance of current embeddings from saved centers of embeddings of old classes
                _distance_loss = torch.tensor(0.0, requires_grad = True)
                if globals.OOD_CLASS != 1:
                    interCenterMask = labels != (ood_label + labels_offset)
                else:
                    interCenterMask = torch.ones_like(labels, dtype=torch.bool)
                for emb_center in net.prev_train_embedding_centers:
                    _distance_loss = _distance_loss - distance_loss*torch.sum(torch.norm(embeddings[interCenterMask] - emb_center, dim=1))/inputs.shape[0]
                epochDistanceLoss += _distance_loss.item()
                loss = loss + _distance_loss

            if param_reuse_loss != 0:
                if globals.OOD_CLASS == 1:
                    mask = labels != iteration*(globals.CLASSES_PER_ITER+1) + globals.CLASSES_PER_ITER
                else:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                net.zero_grad()
                _output, _target = outputs[mask], labels[mask]
                _output = F.log_softmax(_output, dim=1)
                log_likelihood = _output[range(len(_target)), _target].mean()
                grads = autograd.grad(log_likelihood, net.parameters(), create_graph=True)
                fisher_information = [g**2 for g in grads]
                flattened_current_fisher = torch.cat([g.view(-1) for g in fisher_information])
                flattened_saved_fisher = torch.cat([param.view(-1) for param in net.fisher_information.values()])
                _param_reuse_loss = param_reuse_loss*torch.dot(flattened_current_fisher, flattened_saved_fisher)
                epochParamReuseLoss += _param_reuse_loss.item()
                loss = loss + _param_reuse_loss


            if center_loss != 0: # center loss -- see https://github.com/KaiyangZhou/pytorch-center-loss
                _center_loss = centerLoss(embeddings, labels) * center_loss
                epochCenterLoss += _center_loss.item()
                loss = loss + _center_loss
            loss.backward()
            # work after calculating gradients - clipping, masking, etc..
            if freeze_nonzero_params:
                apply_mask_to_gradients(net, nonzero_masks)
            if center_loss != 0:
                centerLoss.centers.grad.zero_()
                for param in centerLoss.parameters():
                    # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                    param.grad.data *= (centerLossLr / (center_loss * lr))
            if maxGradNorm is not None:
                prevNorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxGradNorm)
                if prevNorm > maxGradNorm and verbose:
                    print("clipped gradients!", prevNorm)
            
            optimizer.step()
            
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

        if timeout is not None and time.time() - start > timeout:
            raise Exception("CL train timed out!")
        
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
                        outputs = net(feat)
                        embeddings = net.get_embeddings(feat)
                        
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
        val_epochCELoss /= len(valloader)
        val_epochKDLoss /= len(valloader)

        epochCELoss /= len(trainloader)
        epochL1Loss /= len(trainloader)
        epochEWCLoss /= len(trainloader)
        epochKDLoss /= len(trainloader)
        epochCenterLoss /= len(trainloader)
        epochDistanceLoss /= len(trainloader)
        epochParamReuseLoss /= len(trainloader)
        breakCondition = True
        if stopOnLoss is not None:
            breakCondition = epochCELoss < stopOnLoss
        if stopOnValAcc is not None:
            breakCondition = breakCondition and task_val_accuracy > stopOnValAcc
        if stopOnLoss is None and stopOnValAcc is None:
            breakCondition = False
        #breakCondition = task_val_accuracy > 0.925
        if verbose and epoch%report_frequency == 0:
            print("Epoch", epoch, f" CELoss: {epochCELoss:.4f}, KLLoss: {epochKDLoss:.4f}, L1Loss: {epochL1Loss:.4f}, EWCLoss: {epochEWCLoss:.4f}, CenterLoss: {epochCenterLoss:.4f}, InterCenterLoss: {epochDistanceLoss:.4f}, ParamReuseLoss: {epochParamReuseLoss:.4f}")
            print("Validation losses:", val_epochCELoss, val_epochKDLoss)
            print("Validation accuracy (for last task)", task_val_accuracy)
            print("Fraction of nonzero parameters", calculate_nonzero_percentage(net))
            print("Total validation accuracy", total_val_accuracy)
            if validateOnAll:# and breakCondition:
                plot_confusion_matrix(predicted_full, all_val_labels, list(range(CLASSES_PER_ITER*(iteration+1))))
                plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, None)
            print('\n')
        if breakCondition:
            break
    store_additional_data(net, trainloader.dataset, iteration+1)
    store_test_embedding_centers(net, iteration+1)
    if ogd:
        optimizer.update_basis(trainloader.dataset)

    if verbose:
        plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, net.prev_train_embedding_centers)