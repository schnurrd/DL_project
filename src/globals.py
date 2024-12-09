import torch

ITERATIONS = 5
CLASSES_PER_ITER = 2
SEED = 42
VAR_INFERENCE = False
BATCH_SIZE = 40
WITH_DROPOUT = False
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda:0") if torch.cuda.is_available() else
    torch.device("cpu")
)
full_trainset = None
trainset = None
testset = None
trainloaders = None
valloaders = None
testloaders = None
ood_method = 'smoothmixc' 