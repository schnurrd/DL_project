import torch

ITERATIONS = 5
CLASSES_PER_ITER = 2
SEED = 42
VAR_INFERENCE = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
full_trainset = None
trainset = None
testset = None
trainloaders = None
valloaders = None
testloaders = None
centerLoss = None