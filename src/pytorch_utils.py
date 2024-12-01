import torch
import numpy as np

def get_labels(loader):
    labels = []
    for _, lab in loader:
        labels += lab.tolist()
    return torch.tensor(np.array(labels))

def get_features(loader):
    features = []
    for feat, _ in loader:
        features.append(feat)
        dtype = feat.dtype
    return torch.cat(features, dim=0).to(dtype=dtype)