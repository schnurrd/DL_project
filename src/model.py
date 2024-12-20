import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import globals
import torch.nn.init as init

class RandomSquareDropout(nn.Module):
    def __init__(self, square_size=10):
        super(RandomSquareDropout, self).__init__()
        self.square_size = square_size

    def forward(self, img):
        if not self.training:
            return img

        # Get the shape of the input image (batch, channels, height, width)
        batch_size, channels, height, width = img.size()

        # Randomly decide which images in the batch will get dropout squares
        mask = torch.ones_like(img)

        # Apply dropout to each image in the batch
        for i in range(batch_size):
                # Randomly select a starting point for the square
                y = torch.randint(0, height - self.square_size + 1, (1,)).item()
                x = torch.randint(0, width - self.square_size + 1, (1,)).item()

                # Set pixels within the square to 0
                mask[i, :, y:y + self.square_size, x:x + self.square_size] = 0
        return img * mask
    
# Model for MNIST
class MnistCNN(nn.Module):
    def __init__(self, n_classes):
        withDropout = globals.WITH_DROPOUT
        super(MnistCNN, self).__init__()
        self.n_classes = n_classes
        if withDropout:
            self.imgdr = RandomSquareDropout(8)
        else:
            self.imgdr = nn.Identity()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        if withDropout:
            self.dr1 = nn.Dropout(0.2)
        else:
            self.dr1 = nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        if withDropout:
            self.dr2 = nn.Dropout(0.2)
        else:
            self.dr2 = nn.Identity()
        self.output_layer = nn.Linear(64 * 5 * 5, 128)
        if withDropout:
            self.dr3 = nn.Dropout(0.5)
        else:
            self.dr3 = nn.Identity()
        self.n_embeddings = 128
        self.fc2 = nn.Linear(self.n_embeddings, n_classes)
        self.prev_train_embedding_centers = []
        self.prev_test_embedding_centers = []
        self.ogd_basis = torch.empty((0, 0), device=globals.DEVICE)

    def forward(self, x):
        x = self.imgdr(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dr1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dr2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.output_layer(x))
        x = self.dr3(x)
        x = self.fc2(x)
        return x
        
    def get_pred_and_embeddings(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = self.dr1(x)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dr2(x)
            x = x.view(-1, 64 * 5 * 5)
            embeddings = F.relu(self.output_layer(x))
            x = self.dr3(embeddings)
            x = self.fc2(x)
            return x, embeddings
    
    def copyPrev(self, prevModel):
        self.conv1.weight = copy.deepcopy(prevModel.conv1.weight)
        self.conv1.bias = copy.deepcopy(prevModel.conv1.bias)
        self.conv2.weight = copy.deepcopy(prevModel.conv2.weight)
        self.conv2.bias = copy.deepcopy(prevModel.conv2.bias)
        self.output_layer.weight = copy.deepcopy(prevModel.output_layer.weight)
        self.output_layer.bias = copy.deepcopy(prevModel.output_layer.bias)
        self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.bias)
        
        self.prev_train_embedding_centers = prevModel.prev_train_embedding_centers
        self.prev_test_embedding_centers = prevModel.prev_test_embedding_centers
        self.n_embeddings = prevModel.n_embeddings
        self.ogd_basis = copy.deepcopy(prevModel.ogd_basis)
        
        self.old_param_size_map = {}
        pointer = 0
        for name, param in prevModel.named_parameters():
            param_size = param.numel()
            end_idx = pointer + param_size
            self.old_param_size_map[name] = param_size
            pointer = end_idx

# Model for Tiny Imagenet
class TinyImageNetCNN(nn.Module): # Modified AlexNet https://github.com/DennisHanyuanXu/Tiny-ImageNet
    def __init__(self, n_classes):
        super(TinyImageNetCNN, self).__init__()
        self.n_classes = n_classes
        withDropout = globals.WITH_DROPOUT

        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)

        if withDropout:
            self.imgdr = RandomSquareDropout(21)
            self.dr1 = nn.Dropout(0.2)
            self.dr2 = nn.Dropout(0.2)
            self.dr3 = nn.Dropout(0.2)
            self.dr4 = nn.Dropout(0.2)
            self.dr5 = nn.Dropout(0.2)
        else:
            self.imgdr = nn.Identity()
            self.dr1 = nn.Identity()
            self.dr2 = nn.Identity()
            self.dr3 = nn.Identity()
            self.dr4 = nn.Identity()
            self.dr5 = nn.Identity()
        # Define pooling layer

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.output_layer = nn.Linear(4096, n_classes)
        if withDropout:
            self.dr6 = nn.Dropout(0.5)
            self.dr7 = nn.Dropout(0.5)
        else:
            self.dr6 = nn.Identity()
            self.dr7 = nn.Identity()
        ####
        self.n_embeddings = 4096
        self.prev_train_embedding_centers = []
        self.prev_test_embedding_centers = []
        self.ogd_basis = torch.empty((0, 0), device=globals.DEVICE)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = self.mp3(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr6(F.relu(self.fc1(x)))
        x = self.dr7(F.relu(self.fc2(x)))
        x = self.output_layer(x)
        return x
        
    def get_pred_and_embeddings(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = self.mp3(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr6(F.relu(self.fc1(x)))
        embeddings = F.relu(self.fc2(x))
        x = self.dr7(embeddings)
        x = self.output_layer(x)
        return x, embeddings
    
    def copyPrev(self, prevModel):
        self.conv1.weight = copy.deepcopy(prevModel.conv1.weight)
        self.conv1.bias = copy.deepcopy(prevModel.conv1.bias)
        self.conv2.weight = copy.deepcopy(prevModel.conv2.weight)
        self.conv2.bias = copy.deepcopy(prevModel.conv2.bias)
        self.conv3.weight = copy.deepcopy(prevModel.conv3.weight)
        self.conv3.bias = copy.deepcopy(prevModel.conv3.bias)
        self.conv4.weight = copy.deepcopy(prevModel.conv4.weight)
        self.conv4.bias = copy.deepcopy(prevModel.conv4.bias)
        self.conv5.weight = copy.deepcopy(prevModel.conv5.weight)
        self.conv5.bias = copy.deepcopy(prevModel.conv5.bias)
        self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
        self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
        self.fc2.weight = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias = copy.deepcopy(prevModel.fc2.bias)
        self.output_layer.weight[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.output_layer.weight)
        self.output_layer.bias[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.output_layer.bias)
        self.prev_train_embedding_centers = prevModel.prev_train_embedding_centers
        self.prev_test_embedding_centers = prevModel.prev_test_embedding_centers
        self.n_embeddings = prevModel.n_embeddings
        self.ogd_basis = copy.deepcopy(prevModel.ogd_basis)
        
        self.old_param_size_map = {}
        pointer = 0

        for name, param in prevModel.named_parameters():
            param_size = param.numel()
            end_idx = pointer + param_size
            self.old_param_size_map[name] = param_size
            pointer = end_idx

# Model for CIFAR10
class Cifar10CNN(nn.Module): # Modified AlexNet https://github.com/DennisHanyuanXu/Tiny-ImageNet
    def __init__(self, n_classes):
        super(Cifar10CNN, self).__init__()
        self.n_classes = n_classes
        withDropout = globals.WITH_DROPOUT

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Adjusted kernel size and stride
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling reduces to 16x16
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling reduces to 8x8
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling reduces to 4x4

        if withDropout:
            self.imgdr = RandomSquareDropout(10)
            self.dr1 = nn.Dropout(0.2)
            self.dr2 = nn.Dropout(0.2)
            self.dr3 = nn.Dropout(0.2)
            self.dr4 = nn.Dropout(0.2)
            self.dr5 = nn.Dropout(0.2)
        else:
            self.imgdr = nn.Identity()
            self.dr1 = nn.Identity()
            self.dr2 = nn.Identity()
            self.dr3 = nn.Identity()
            self.dr4 = nn.Identity()
            self.dr5 = nn.Identity()
        # Define pooling layer

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # Adjusted input size after 4x4 pooling
        self.fc2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, n_classes)
        if withDropout:
            self.dr6 = nn.Dropout(0.5)
            self.dr7 = nn.Dropout(0.5)
        else:
            self.dr6 = nn.Identity()
            self.dr7 = nn.Identity()
        ####
        self.n_embeddings = 512
        self.prev_train_embedding_centers = []
        self.prev_test_embedding_centers = []
        self.ogd_basis = torch.empty((0, 0), device=globals.DEVICE)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = self.mp3(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr6(F.relu(self.fc1(x)))
        x = self.dr7(F.relu(self.fc2(x)))
        x = self.output_layer(x)
        return x
        
    def get_pred_and_embeddings(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = self.mp3(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr6(F.relu(self.fc1(x)))
        embeddings = F.relu(self.fc2(x))
        x = self.dr7(embeddings)
        x = self.output_layer(x)
        return x, embeddings
    
    def copyPrev(self, prevModel):
        self.conv1.weight = copy.deepcopy(prevModel.conv1.weight)
        self.conv1.bias = copy.deepcopy(prevModel.conv1.bias)
        self.conv2.weight = copy.deepcopy(prevModel.conv2.weight)
        self.conv2.bias = copy.deepcopy(prevModel.conv2.bias)
        self.conv3.weight = copy.deepcopy(prevModel.conv3.weight)
        self.conv3.bias = copy.deepcopy(prevModel.conv3.bias)
        self.conv4.weight = copy.deepcopy(prevModel.conv4.weight)
        self.conv4.bias = copy.deepcopy(prevModel.conv4.bias)
        self.conv5.weight = copy.deepcopy(prevModel.conv5.weight)
        self.conv5.bias = copy.deepcopy(prevModel.conv5.bias)
        self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
        self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
        self.fc2.weight = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias = copy.deepcopy(prevModel.fc2.bias)
        self.output_layer.weight[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.output_layer.weight)
        self.output_layer.bias[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.output_layer.bias)
        self.prev_train_embedding_centers = prevModel.prev_train_embedding_centers
        self.prev_test_embedding_centers = prevModel.prev_test_embedding_centers
        self.n_embeddings = prevModel.n_embeddings
        if hasattr(prevModel, "ogd_basis"):
            self.ogd_basis = prevModel.ogd_basis  # Save ogd_basis
            prevModel.ogd_basis = None
        
        self.old_param_size_map = {}
        pointer = 0

        for name, param in prevModel.named_parameters():
            param_size = param.numel()
            end_idx = pointer + param_size
            self.old_param_size_map[name] = param_size
            pointer = end_idx
