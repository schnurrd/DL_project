import torch

ITERATIONS = 5
CLASSES_PER_ITER = 2
SEED = 42
VAR_INFERENCE = False
BATCH_SIZE = 4
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
OOD_CLASS = 1 # 1 if training with an additional "out of distribution class" with random data. 0 otherwise