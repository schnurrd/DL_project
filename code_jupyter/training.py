import os
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
                            update_EWC_data,
                            calc_ewc_loss,
                            apply_mask_to_gradients,
                            CenterLoss)

from augmenting_dataloader import AugmentedOODTrainset
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
                load_path = None, 
                save_path = None, 
                epochs = 5, 
                l1_reg_strength = 0, 
                centerLossStrength = 0, 
                withBuffer = False, 
                freezeCenterLossCenters = None, 
                report_frequency=1,
                stopOnLoss = None,
                timeout = None):
    """
    Used to train on first task of CL.
    For more details, see comment of train_model_CL, most of which is analogous to this function
    """
    start = time.time()
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    ITERATIONS = globals.ITERATIONS
    trainloaders = globals.trainloaders
    valloaders = globals.valloaders
    buffer_size = globals.BATCH_SIZE
    global centerLoss
    if epochs > 0:
        augmented_dataset = AugmentedOODTrainset(0, len(trainloaders[0].dataset)//CLASSES_PER_ITER)
        trainloader = DataLoader(augmented_dataset, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    buffer = []
    lr = 0.001
    centerLossLr = 0.005
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if centerLossStrength > 0:
        centerLoss = CenterLoss(num_classes=(CLASSES_PER_ITER+1)*ITERATIONS, feat_dim=net.n_embeddings, use_gpu=True)
        if freezeCenterLossCenters is not None:
            pc = torch.stack(net.prev_embedding_centers)
            print("len centers", centerLoss.centers.data.shape)
            for k, t in enumerate(pc):
                #print("setting", k)
                centerLoss.centers.data[k] = t
        params = list(net.parameters()) + list(centerLoss.parameters())
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)#, weight_decay = 0.001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)#, weight_decay = 0.001)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    model_path = load_path
    if model_path and os.path.isfile(model_path):
        # load trained model parameters from disk
        net.load_state_dict(torch.load(model_path))
    else:
        for epoch in range(epochs):  # loop over the dataset multiple times
            epoch_loss = 0.0
            epoch_buffer_loss = 0.0
            epoch_buffer_size = 0
            epoch_center_loss = 0.0
            val_epoch_loss = 0.0
            buffer_runs = 0
            bufferBatch = False
            iterator = iter(trainloader)
            batch = next(iterator, None)
            while batch is not None:
                #if bufferBatch:
                #    epoch_buffer_size += 1
                # get the inputs
                inputs, labels = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, embeddings = net.get_pred_and_embeddings(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                if l1_reg_strength != 0:
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    loss += l1_reg_strength * l1_norm
                if centerLossStrength != 0:
                    centerLossVal = centerLoss(embeddings, labels) * centerLossStrength
                    epoch_center_loss += centerLossVal.item()
                    loss += centerLossVal
                if bufferBatch:
                    epoch_buffer_loss += loss.item()
                loss.backward()
                if centerLossStrength != 0:
                    centerLoss.centers.grad.zero_()
                    for param in centerLoss.parameters():
                        # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                        param.grad.data *= (centerLossLr / (centerLossStrength * lr))
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                correct_preds = (preds == labels) & ((labels + 1) % (CLASSES_PER_ITER+1) != 0)
                buffered = bufferBatch
                bufferBatch = False
                if withBuffer and correct_preds.any() and not buffered:
                    # Compute saliency maps for the correctly classified images
                    buff_inputs, buff_labels = build_buffer_data(inputs[correct_preds], labels[correct_preds], net, CLASSES_PER_ITER)
                    
                    # Add masked images and labels to the buffer
                    buffer.extend(zip(buff_inputs, buff_labels))
                    
                    # Process buffer if it exceeds the buffer size
                    if len(buffer) >= buffer_size:
                        epoch_buffer_size += len(buffer)
                        buffer_runs += 1
                        bufferBatch = True
                        buffer_images, buffer_labels = zip(*buffer)
                        batch = torch.stack(buffer_images).to(DEVICE), torch.tensor(buffer_labels).to(DEVICE)
                        buffer = []
                if not bufferBatch:
                    batch = next(iterator, None)
            if timeout is not None and time.time() - start > timeout:
                raise Exception("initial train timed out!")
            net.eval()
            for inputs, labels in valloader:
                with torch.no_grad():
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    masked_images = inputs
                    outputs = net(masked_images)
                    loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item()
                    optimizer.zero_grad()
            net.train()
            
            val_epoch_loss /= len(valloader)
            if epoch_buffer_size != 0:
                epoch_loss /= (len(trainloader) + buffer_runs)
                epoch_center_loss /= (len(trainloader) + buffer_runs)
                epoch_buffer_loss /= buffer_runs
            else:
                epoch_loss /= len(trainloader)
                epoch_center_loss /= (len(trainloader))
            if verbose and epoch%report_frequency == 0:
                print(f"Epoch {epoch}, CE Loss: {epoch_loss:.4f}, center loss: {epoch_center_loss:.4f}, of which buffer loss: {epoch_buffer_loss:.4f} for buffer with size {epoch_buffer_size:.1f}")
                print("Validation loss", val_epoch_loss)
                print("Fraction of nonzero parameters", calculate_nonzero_percentage(net), '\n')
            if stopOnLoss is not None and epoch_loss < stopOnLoss:# and buffer_loss < 0.05:
            #if False:
                break
        if save_path:
            torch.save(net.state_dict(), save_path)
        update_EWC_data(net, trainloader.dataset, 1)

def train_model_CL(net, 
                   prevModel, 
                   trainloader, 
                   valloader, 
                   iteration, 
                   verbose = False, 
                   n_epochs=4, 
                   validateOnAll = False, 
                   freeze_nonzero_params = False, 
                   l1_reg_strength = 0, 
                   ewc_reg_strength = 0, 
                   kd_reg_strength = 0, 
                   withBuffer = False, 
                   distance_embeddings_strength = 0, 
                   centerLossStrength = 0, 
                   freezeCenterLossCenters = None, 
                   report_frequency=1,
                   lr = 0.001,
                   momentum = 0.9,
                   maxGradNorm = None,
                   stopOnLoss = None,
                   stopOnValAcc = None,
                   timeout=None):
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
    l1_reg_strength : float, optional (default=0)
        Strength of L1 regularization applied to the model's weights to encourage sparsity.
    ewc_reg_strength : float, optional (default=0)
        Strength of Elastic Weight Consolidation (EWC) regularization to retain knowledge from previous tasks.
    kd_reg_strength : float, optional (default=0)
        Strength of Knowledge Distillation (KD) loss to transfer knowledge from `prevModel`.
    withBuffer : bool, optional (default=False)
        If True, uses a custom buffer (NOT a replay buffer) which generates additional samples based on
        logic inside this function and build_buffer_data inside this file. The initial idea was to mask high-saliency
        regions for correctly guessed images and build additional batches with them, to encourage the model to learn
        more holistic features.
    distance_embeddings_strength : float, optional (default=0)
        Strength of the loss term that encourages distances between embeddings of classes for current task
        and stored embedding centers of previous classes
    centerLossStrength : float, optional (default=0)
        Strength of the center loss to enforce tighter clusters in the embedding space.
    freezeCenterLossCenters : torch.Tensor or None, optional (default=None)
        Use stored center loss centers; if None, centers will be computed during training.
        Only works if first running some training without center loss on current task
    """
    start = time.time()
    torch.autograd.set_detect_anomaly = True
    CLASSES_PER_ITER = globals.CLASSES_PER_ITER
    DEVICE = globals.DEVICE
    ITERATIONS = globals.ITERATIONS
    trainloaders = globals.trainloaders
    valloaders = globals.valloaders
    buffer_size = globals.BATCH_SIZE
    global centerLoss
    augmented_dataset = AugmentedOODTrainset(iteration, len(trainloaders[iteration].dataset)//CLASSES_PER_ITER)
    trainloader = DataLoader(augmented_dataset, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    net, prevModel = net.to(DEVICE), prevModel.to(DEVICE)
    prevModel.withDropout = False
    ceLoss = nn.CrossEntropyLoss()
    klDivLoss = nn.KLDivLoss(reduction="batchmean")
    centerLossLr = 0.005
    if centerLossStrength > 0:
        if freezeCenterLossCenters is not None:
            pc = torch.stack(net.prev_embedding_centers)
            ind = 0 # must account for OOD class centers
            for k, t in enumerate(pc):
                #print("setting", ind)
                centerLoss.centers.data[ind] = t
                ind  += 1
                if (k+1)%CLASSES_PER_ITER == 0:
                    ind += 1
        params = list(net.parameters()) + list(centerLoss.parameters())
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)#, weight_decay = 0.001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)#, weight_decay = 0.001)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    prevModel.eval()
    epoch = 0
    buffer = []
    if freeze_nonzero_params:
        nonzero_masks = create_nonzero_mask(net)
    if l1_reg_strength:
        zero_masks = create_zero_mask(net)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        epochCELoss = 0.0
        epochKLLoss = 0.0
        epochL1Loss = 0.0
        epochEWCLoss = 0.0
        epochCenterLoss = 0.0
        epochInterCenterLoss = 0.0
        buffer_epochLoss = 0.0
        epoch_buffer_size = 0
        val_epochCELoss = 0.0
        val_epochKLLoss = 0.0
        buffer_runs = 0
        bufferBatch = False
        iterator = iter(trainloader)
        batch = next(iterator, None)
        #onlyEWC = (ewc_reg_strength != 0 and (epoch+1)%3 == 0)
        onlyEWC = False
        while batch is not None:
            #if bufferBatch:
            #    epoch_buffer_size += 1
            # get the inputs
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            '''
            if iteration == 4:
                for i, l in enumerate(labels):
                    print(l)
                    show_image(inputs[i][0])
            '''
            if not bufferBatch:
                labels += iteration
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, embeddings = net.get_pred_and_embeddings(inputs)
            net.eval()
            outputsNoDropout = net(inputs)
            net.train()
            oldClassOutputs = outputsNoDropout[:, :-CLASSES_PER_ITER-1]
            with torch.no_grad():
                prevOutputs = prevModel(inputs)
            if not onlyEWC:
                loss = ceLoss(outputs[:,-CLASSES_PER_ITER-1:], labels - iteration*(CLASSES_PER_ITER+1))
            else:
                loss = torch.tensor(0.0, requires_grad = True)
            CELoss = loss.item()
            epochCELoss += CELoss
            kdLoss = kd_reg_strength*klDivLoss(F.log_softmax(oldClassOutputs, dim=-1), F.softmax(prevOutputs, dim=-1))
            loss = loss + kdLoss
            #for it in range(iteration):
            #    loss += klDivLoss(F.log_softmax(oldClassOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1), F.softmax(prevOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1))
            epochKLLoss += kdLoss
            if l1_reg_strength != 0:
                l1_loss = torch.tensor(0.0, requires_grad = True)
                for name, p in net.named_parameters():
                    l1_norm = torch.sum(zero_masks[name]*p.abs())
                    l1_loss = l1_loss + l1_reg_strength * l1_norm
                epochL1Loss += l1_loss.item()
                loss += l1_loss
            if ewc_reg_strength != 0:
                ewc_loss = torch.tensor(0.0, requires_grad = True)
                ewc_loss = ewc_loss + ewc_reg_strength * calc_ewc_loss(net)
                epochEWCLoss += ewc_loss.item()
                loss = loss + ewc_loss
            if distance_embeddings_strength != 0:
                emb_dist_loss = torch.tensor(0.0, requires_grad = True)
                interCenterMask = labels != augmented_dataset.ood_label
                for emb_center in net.prev_embedding_centers:
                    emb_dist_loss = emb_dist_loss - torch.sum(distance_embeddings_strength*torch.norm(embeddings[interCenterMask] - emb_center, dim=1))/inputs.shape[0]
                epochInterCenterLoss += emb_dist_loss.item()
                loss = loss + epochInterCenterLoss
            if centerLossStrength != 0:
                centerLossVal = centerLoss(embeddings, labels) * centerLossStrength
                epochCenterLoss += centerLossVal.item()
                loss = loss + centerLossVal
            if bufferBatch:
                buffer_epochLoss += loss.item()
            loss.backward()
            if freeze_nonzero_params:
                apply_mask_to_gradients(net, nonzero_masks)
            if centerLossStrength != 0:
                if centerLossStrength != 0:
                    centerLoss.centers.grad.zero_()
                for param in centerLoss.parameters():
                    # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                    param.grad.data *= (centerLossLr / (centerLossStrength * lr))
            if maxGradNorm is not None:
                prevNorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=maxGradNorm)
                if prevNorm > maxGradNorm:
                    print("clipped gradients!", prevNorm)
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(outputs[:,-CLASSES_PER_ITER-1:], 1)
                correct_preds = (preds == labels - iteration*(CLASSES_PER_ITER+1)) & ((labels - iteration*(CLASSES_PER_ITER+1) + 1) % (CLASSES_PER_ITER+1) != 0)
                buffered = bufferBatch
                bufferBatch = False
            if withBuffer and not buffered and correct_preds.any():
                # Compute saliency maps for the correctly classified images
                correct_images = inputs[correct_preds]
                correct_labels = labels[correct_preds]
                buff_inputs, buff_labels = build_buffer_data(correct_images, correct_labels, net, CLASSES_PER_ITER + iteration*(CLASSES_PER_ITER+1))
                
                # Add masked images and labels to the buffer
                buffer.extend(zip(buff_inputs, buff_labels))
                
                # Process buffer if it exceeds the buffer size
                if len(buffer) >= buffer_size:
                    epoch_buffer_size += len(buffer)
                    buffer_runs += 1
                    bufferBatch = True
                    buffer_images, buffer_labels = zip(*buffer)
                    batch = torch.stack(buffer_images).to(DEVICE), torch.tensor(buffer_labels).to(DEVICE)
                    buffer = []
            if not bufferBatch:
                batch = next(iterator, None)
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
                masked_images = inputs
                outputs = net(masked_images)
                outputsNoDropout = outputs
                oldClassOutputs = outputsNoDropout[:, :-CLASSES_PER_ITER-1]
                with torch.no_grad():
                    prevOutputs = prevModel(masked_images)
                loss = ceLoss(outputs[:,-CLASSES_PER_ITER-1:], labels - iteration*CLASSES_PER_ITER)
                CELoss = loss.item()
                val_epochCELoss += CELoss
                #for it in range(iteration):
                #    loss += klDivLoss(F.log_softmax(oldClassOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1), F.softmax(prevOutputs[:, it*CLASSES_PER_ITER:(it+1)*CLASSES_PER_ITER], dim=-1))
                loss += klDivLoss(F.log_softmax(oldClassOutputs, dim=-1), F.softmax(prevOutputs, dim=-1))
                val_epochKLLoss += loss.item() - CELoss
                outputs_no_OOD = outputs[:, [i for i in range(outputs.size(1)) if (i + 1) % (CLASSES_PER_ITER+1) != 0]]
                _, predicted = torch.max(outputs_no_OOD, 1)  # Assuming it's a classification task
                predicted_eval_task.extend(predicted.cpu().numpy())  # Move to CPU and convert to numpy for ease
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
        val_epochKLLoss /= len(valloader)
        if epoch_buffer_size != 0:
            buffer_epochLoss /= buffer_runs
            epochCELoss /= (len(trainloader) + buffer_runs)
            epochL1Loss /= (len(trainloader) + buffer_runs)
            epochEWCLoss /= (len(trainloader) + buffer_runs)
            epochKLLoss /= (len(trainloader) + buffer_runs)
            epochCenterLoss /= (len(trainloader) + buffer_runs)
            epochInterCenterLoss /= (len(trainloader) + buffer_runs)
        else:
            epochCELoss /= len(trainloader)
            epochL1Loss /= len(trainloader)
            epochEWCLoss /= len(trainloader)
            epochKLLoss /= len(trainloader)
            epochCenterLoss /= len(trainloader)
            epochInterCenterLoss /= len(trainloader)
        breakCondition = True
        if stopOnLoss is not None:
            breakCondition = epochCELoss < stopOnLoss
        if stopOnValAcc is not None:
            breakCondition = breakCondition and task_val_accuracy > stopOnValAcc
        if stopOnLoss is None and stopOnValAcc is None:
            breakCondition = False
        #breakCondition = task_val_accuracy > 0.925
        if verbose and epoch%report_frequency == 0:
            print("Epoch", epoch, f" CELoss: {epochCELoss:.4f}, KLLoss: {epochKLLoss:.4f}, L1Loss: {epochL1Loss:.4f}, EWCLoss: {epochEWCLoss:.4f}, CenterLoss: {epochCenterLoss:.4f}, InterCenterLoss: {epochInterCenterLoss:.4f}")
            print("Buffer loss: ", buffer_epochLoss, " buffer size ", epoch_buffer_size)
            print("Validation losses:", val_epochCELoss, val_epochKLLoss)
            print("Validation accuracy (for last task)", task_val_accuracy)
            print("Fraction of nonzero parameters", calculate_nonzero_percentage(net))
            print("Total validation accuracy", total_val_accuracy)
            if validateOnAll:# and breakCondition:
                plot_confusion_matrix(predicted_full, all_val_labels, list(range(CLASSES_PER_ITER*(iteration+1))))
                plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, None)
            print('\n')
        #if epochCELoss  < 0.08 and epochKLLoss < 0.02:# and buffer_epochCELoss 
        # 
        #  and buffer_epochKLLoss < 0.2:
        #if task_val_accuracy > 0.935:
        #if epochCELoss < 0.06:#
        if breakCondition:
            break
    update_EWC_data(net, trainloader.dataset, iteration+1)
    plot_embeddings(all_val_embeddings, all_val_labels, (iteration+1)*CLASSES_PER_ITER, net.prev_embedding_centers)