from abc import ABC, abstractmethod

# Define the abstract base class
class TrainingProcedure(ABC):
    @abstractmethod
    def train_base(self, net, trainloader, load_path, save_path, n_epochs, losses_breaks):
        """
        abstract method to train the base model (for task 0). Params:
        net: model to be traineed
        trainloader: loader with training data
        load_path: if not None, will load model parameters from path and skip training
        save_path: if not None, will save model parameters to path
        n_epochs: maximum number of training epochs
        losses_breaks: optional to pass a list of floats to break training when certain loss thresholds have been reached
        """
        pass

    @abstractmethod
    def train_CL(self, net, prevModel, trainloader, task, n_epochs, losses_breaks):
        """
        abstract method to train the model for CL (for tasks > 0). Params:
        net: model to be traineed
        prevModel: model from previous iteration, can be passed for knowledge distillation
        trainloader: loader with training data
        task: task number (counting from 0)
        n_epochs: maximum number of training epochs
        losses_breaks: optional to pass a list of floats to break training when certain loss thresholds have been reached
        """
        pass