import torch
from torch import nn, optim
from torch.functional import F
import os
import globals
from .training_procedure import TrainingProcedure

class sep_CE_comb_KD(TrainingProcedure):
    def train_base(self, net, trainloader, verbose = False, load_path = None, save_path = None, n_epochs = 2, losses_breaks=[0.05]):
        criterion = nn.CrossEntropyLoss()
        optimizer = globals.optimiser
        optimizer.param_groups[0]['params'] = list(net.parameters())
        model_path = load_path
        if model_path and os.path.isfile(model_path):
            # load trained model parameters from disk
            net.load_state_dict(torch.load(model_path))
        else:
            for epoch in range(n_epochs):  # loop over the dataset multiple times

                epoch_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                epoch_loss /= len(trainloader)
                if verbose:
                    print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")
                if losses_breaks != [] and epoch_loss < losses_breaks[0]:
                    break
            if save_path:
                torch.save(net.state_dict(), save_path)
                
    def train_CL(self, net, prevModel, trainloader, task, verbose = False, n_epochs=4, losses_breaks=[0.03,0.03]):
        prevModel.withDropout = False
        ceLoss = nn.CrossEntropyLoss()
        klDivLoss = nn.KLDivLoss(reduction="batchmean")
        optimizer = globals.optimiser
        optimizer.param_groups[0]['params'] = list(net.parameters())
        prevModel.eval()
        epoch = 0
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            epochCELoss = 0.0
            epochKLLoss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                net.eval()
                outputsNoDropout = net(inputs)
                net.train()
                oldClassOutputs = outputsNoDropout[:, :-globals.CLASSES_PER_ITER]
                with torch.no_grad():
                    prevOutputs = prevModel(inputs)
                loss = ceLoss(outputs[:,-globals.CLASSES_PER_ITER:], labels - task*globals.CLASSES_PER_ITER)
                CELoss = loss.item()
                epochCELoss += CELoss
                loss += klDivLoss(F.log_softmax(oldClassOutputs, dim=-1), F.softmax(prevOutputs, dim=-1))
                epochKLLoss += loss.item() - CELoss
                loss.backward()
                optimizer.step()

            epochCELoss /= len(trainloader)
            epochKLLoss /= len(trainloader)
            if verbose:
                print("Epoch lossses: ", epochCELoss, epochKLLoss)
            if epochCELoss  < losses_breaks[0] and epochKLLoss < losses_breaks[1]:
                break