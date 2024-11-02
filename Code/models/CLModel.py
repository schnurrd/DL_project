import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class CLModel(nn.Module, ABC):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    @abstractmethod
    def CLCopy(self, prevModel, task):
        """Abstract method to copy model trained up to previous tasks."""
        pass