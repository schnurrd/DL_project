import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List
import globals

class Dataset:
    def __init__(self, name: str, trans=[transforms.ToTensor()]):
        if name.upper() == "MNIST":
            self.full_trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose(trans))

            self.full_testset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                        transform=transforms.Compose(trans))
            self.n_classes = 10
            self.input_size = 28*28
        elif name.upper() == "CIFAR10":
            self.full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(trans))
            self.full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose(trans))
            self.n_classes = 10
            self.input_size = 3*32*32
        else:
            raise NotImplementedError("Unsupported dataset")
    def getFullLoaders(self, trainBatchSize = 15, testBatchSize = 15) -> Tuple[DataLoader, DataLoader]:
        '''
        Returns two loaders, the training data and the testing data
        '''
        trainloader = DataLoader(self.full_trainset, batch_size=trainBatchSize,shuffle=True)
        testloader = DataLoader(self.full_testset, batch_size=testBatchSize,shuffle=False)
        return trainloader, testloader
    
    def getLoadersPerTask(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        '''
        Returns two lists, the first is the list of training loaders, the latter is the list of testing loaders
        Element at index i is the loader containing all data for class i
        '''
        if self.n_classes%globals.N_TASKS != 0:
            raise Exception("Number of tasks must divide number of classes!")
        classesPerIter = int(self.n_classes/globals.N_TASKS)
        class_sets = [list(range(i*classesPerIter,(i+1)*classesPerIter)) for i in range(globals.N_TASKS)]
        trainloaders = []
        testloaders = []

        for i, class_set in enumerate(class_sets):
            subset_indices = [idx for idx, (_, label) in enumerate(self.full_trainset) if label in class_set]
            test_subset_indices = [idx for idx, (_, label) in enumerate(self.full_testset) if label in class_set]
            train_subset = Subset(self.full_trainset, subset_indices)
            trainloaders.append(DataLoader(train_subset, batch_size=4, shuffle=True))
            subset_indices = [idx for idx, (_, label) in enumerate(self.full_testset) if label in class_set]
            test_subset = Subset(self.full_testset, test_subset_indices)
            testloaders.append(DataLoader(test_subset, batch_size=4, shuffle=False))
        return trainloaders, testloaders
        