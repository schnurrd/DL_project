import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import globals

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
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        if withDropout:
            self.dr3 = nn.Dropout(0.5)
        else:
            self.dr3 = nn.Identity()
        self.n_embeddings = 128
        self.fc2 = nn.Linear(self.n_embeddings, n_classes)
        self.fisher_information = {}
        self.estimated_means = {}
        self.prev_train_embedding_centers = []
        self.prev_test_embedding_centers = []
        self.ogd_basis = torch.empty((0, 0), device=globals.DEVICE)
        for name, param in self.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param).to(globals.DEVICE)
            self.estimated_means[name] = torch.zeros_like(param).to(globals.DEVICE)

    def forward(self, x):
        x = self.imgdr(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dr1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dr2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
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
            embeddings = F.relu(self.fc1(x))
            x = self.dr3(embeddings)
            x = self.fc2(x)
            return x, embeddings
    
    def copyPrev(self, prevModel):
        self.conv1.weight = copy.deepcopy(prevModel.conv1.weight)
        self.conv1.bias = copy.deepcopy(prevModel.conv1.bias)
        self.conv2.weight = copy.deepcopy(prevModel.conv2.weight)
        self.conv2.bias = copy.deepcopy(prevModel.conv2.bias)
        self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
        self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
        self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.bias)
        
        self.fisher_information = {}
        self.estimated_means = {}
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

        for name, param in self.named_parameters():
            if prevModel.fisher_information[name].shape == param.shape:
                # Directly copy Fisher Information and means for matching dimensions
                self.fisher_information[name] = prevModel.fisher_information[name].clone().to(globals.DEVICE)
                self.estimated_means[name] = prevModel.estimated_means[name].clone().to(globals.DEVICE)
            else:
                # Initialize new parameter with expanded dimensions
                new_fisher = torch.zeros_like(param)
                new_means = torch.zeros_like(param)  # Start with current values for new parameters

                # Determine matching dimensions dynamically
                matching_slices = tuple(slice(0, min(dim_new, dim_old)) 
                                        for dim_new, dim_old in zip(param.shape, prevModel.fisher_information[name].shape))

                # Copy over existing values for matching dimensions
                new_fisher[matching_slices] = prevModel.fisher_information[name][matching_slices]
                new_means[matching_slices] = prevModel.estimated_means[name][matching_slices]

                self.fisher_information[name] = new_fisher.to(globals.DEVICE)
                self.estimated_means[name] = new_means.to(globals.DEVICE)

class TinyImageNetCNN(nn.Module): # Modified AlexNet https://github.com/DennisHanyuanXu/Tiny-ImageNet
    def __init__(self, n_classes):
        super(TinyImageNetCNN, self).__init__()
        withDropout = globals.WITH_DROPOUT

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Input: (3, 64, 64), Output: (64, 64, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: (128, 64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Output: (256, 32, 32)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Output: (256, 32, 32)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Output: (512, 16, 16)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # Output: (512, 16, 16)

        if withDropout:
            self.imgdr = RandomSquareDropout(21)
            self.dr1 = nn.Dropout(0.2)
            self.dr2 = nn.Dropout(0.2)
            self.dr3 = nn.Dropout(0.2)
            self.dr4 = nn.Dropout(0.2)
            self.dr5 = nn.Dropout(0.2)
            self.dr6 = nn.Dropout(0.2)
        else:
            self.imgdr = nn.Identity()
            self.dr1 = nn.Identity()
            self.dr2 = nn.Identity()
            self.dr3 = nn.Identity()
            self.dr4 = nn.Identity()
            self.dr5 = nn.Identity()
            self.dr6 = nn.Identity()
        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, n_classes)
        if withDropout:
            self.dr7 = nn.Dropout(0.5)
            self.dr8 = nn.Dropout(0.5)
        else:
            self.dr7 = nn.Identity()
            self.dr8 = nn.Identity()
        ####
        self.n_embeddings = 1024
        self.fisher_information = {}
        self.estimated_means = {}
        self.prev_train_embedding_centers = []
        self.prev_test_embedding_centers = []
        self.ogd_basis = torch.empty((0, 0), device=globals.DEVICE)
        for name, param in self.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param).to(globals.DEVICE)
            self.estimated_means[name] = torch.zeros_like(param).to(globals.DEVICE)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Output: (64, 32, 32)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # Output: (256, 8, 8)
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)  # Output: (512, 4, 4)
        x = self.dr6(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr7(F.relu(self.fc1(x)))
        x = self.dr8(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
        
    def get_pred_and_embeddings(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Output: (64, 32, 32)
        x = self.dr1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Output: (128, 16, 16)
        x = self.dr2(x)

        x = F.relu(self.conv3(x))
        x = self.dr3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # Output: (256, 8, 8)
        x = self.dr4(x)

        x = F.relu(self.conv5(x))
        x = self.dr5(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)  # Output: (512, 4, 4)
        x = self.dr6(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Output: (batch_size, 512 * 4 * 4)
        
        # Fully connected layers
        x = self.dr7(F.relu(self.fc1(x)))
        embeddings = self.dr8(F.relu(self.fc2(x)))
        x = self.fc3(embeddings)
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
        self.conv6.weight = copy.deepcopy(prevModel.conv6.weight)
        self.conv6.bias = copy.deepcopy(prevModel.conv6.bias)
        self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
        self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
        self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-globals.OOD_CLASS] = copy.deepcopy(prevModel.fc2.bias)
        
        self.fisher_information = {}
        self.estimated_means = {}
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

        for name, param in self.named_parameters():
            if prevModel.fisher_information[name].shape == param.shape:
                # Directly copy Fisher Information and means for matching dimensions
                self.fisher_information[name] = prevModel.fisher_information[name].clone().to(globals.DEVICE)
                self.estimated_means[name] = prevModel.estimated_means[name].clone().to(globals.DEVICE)
            else:
                # Initialize new parameter with expanded dimensions
                new_fisher = torch.zeros_like(param)
                new_means = torch.zeros_like(param)  # Start with current values for new parameters

                # Determine matching dimensions dynamically
                matching_slices = tuple(slice(0, min(dim_new, dim_old)) 
                                        for dim_new, dim_old in zip(param.shape, prevModel.fisher_information[name].shape))

                # Copy over existing values for matching dimensions
                new_fisher[matching_slices] = prevModel.fisher_information[name][matching_slices]
                new_means[matching_slices] = prevModel.estimated_means[name][matching_slices]

                self.fisher_information[name] = new_fisher.to(globals.DEVICE)
                self.estimated_means[name] = new_means.to(globals.DEVICE)