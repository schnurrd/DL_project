import torch

ITERATIONS = 5
CLASSES_PER_ITER = 2
SEED = 42
BATCH_SIZE = 32
WITH_DROPOUT = False
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda:0") if torch.cuda.is_available() else
    torch.device("cpu")
)
EXPERIMENT_N_RUNS = 3
full_trainset = None
trainset = None
valset = None
testset = None
trainloaders = None
valloaders = None
testloaders = None
val_set_size = 0.1
OOD_CLASS = 0
ood_method = 'jigsaw'

def toggle_OOD(method = 'jigsaw'):
    '''
    supported methods:
    smoothmixc
    smoothmixs
    cutmix
    fmix
    jigsaw
    '''
    global OOD_CLASS
    global ood_method
    OOD_CLASS = 1
    ood_method = method

def disable_OOD():
    global OOD_CLASS
    global ood_method
    OOD_CLASS = 0
    ood_method = None