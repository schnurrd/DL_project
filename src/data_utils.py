from collections import defaultdict
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import globals

def initialize_data(dataset='mnist'):
    data_folder = './../data/'
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.5,), (0.5,))  # Normalizes to mean 0.5 and std 0.5 for the single channel
        ])
        globals.full_trainset = torchvision.datasets.MNIST(data_folder, train=True, download=True,
                                    transform=transform)
        targets = np.array(globals.full_trainset.targets)
        globals.testset = torchvision.datasets.MNIST(data_folder, train=False, download=True,
                                transform=transform)
    elif dataset == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.5,), (0.5,))  # Normalizes to mean 0.5 and std 0.5 for the single channel
        ])
        globals.full_trainset = datasets.ImageFolder(root=data_folder + "tiny-imagenet-200/train", transform=transform)
        globals.testset = datasets.ImageFolder(root=data_folder + "tiny-imagenet-200/val", transform=transform)
        targets = [sample[1] for sample in globals.full_trainset.samples]
        targets = np.array(targets)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize((0.5,), (0.5,))  # Normalizes to mean 0.5 and std 0.5 for the single channel
        ])
        globals.full_trainset = torchvision.datasets.CIFAR10(data_folder, train=True, download=True, transform=transform)
        targets = np.array(globals.full_trainset.targets)
        globals.testset = torchvision.datasets.CIFAR10(data_folder, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError("unsupported dataset")
    if globals.val_set_size != 0:
        # Perform stratified split
        train_indices, val_indices = train_test_split(
            np.arange(len(targets)),
            test_size=globals.val_set_size,
            stratify=targets
        )
    else:
        train_indices = np.arange(len(targets))
        val_indices = []

    # Create subsets
    globals.valset = Subset(globals.full_trainset, val_indices)
    globals.trainset = Subset(globals.full_trainset, train_indices)

    # Define class pairs for each subset
    class_pairs = [tuple(range(i*globals.CLASSES_PER_ITER,(i+1)*globals.CLASSES_PER_ITER)) for i in range(globals.ITERATIONS)]
    #print(class_pairs)

    # Dictionary to hold data loaders for each subset
    globals.trainloaders = []
    globals.testloaders = []
    globals.valloaders = []
    subset_indices = []

    def compute_class_to_indices(dataset):
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_to_indices[label].append(idx)
        return class_to_indices

    train_class_to_indices = compute_class_to_indices(globals.trainset)
    val_class_to_indices = compute_class_to_indices(globals.valset)
    test_class_to_indices = compute_class_to_indices(globals.testset)

    # Loop over each class pair
    for i, t in enumerate(class_pairs):
        # Get indices of images belonging to the specified class pair
        subs_ind = [idx for cls in t for idx in train_class_to_indices[cls]]
        val_subset_indices = [idx for cls in t for idx in val_class_to_indices[cls]]
        test_subset_indices = [idx for cls in t for idx in test_class_to_indices[cls]]

        # Create a subset for the current class pair
        train_subset = Subset(globals.trainset, subs_ind)
        globals.trainloaders.append(DataLoader(train_subset, batch_size=globals.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers = 0))
        subset_indices.append(subs_ind)
        
        val_subset = Subset(globals.valset, val_subset_indices)
        globals.valloaders.append(DataLoader(val_subset, batch_size=100, shuffle=False))

        test_subset = Subset(globals.testset, test_subset_indices)
        globals.testloaders.append(DataLoader(test_subset, batch_size=100, shuffle=False))
