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
    
class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.imgdr = RandomSquareDropout(8)
        #self.imgdr = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        #self.dr1 = nn.Dropout(0.2)
        self.dr1 = nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #self.dr2 = nn.Dropout(0.2)
        self.dr2 = nn.Identity()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dr3 = nn.Dropout(0.5)
        self.n_embeddings = 128
        self.fc2 = nn.Linear(self.n_embeddings, n_classes)
        self.fisher_information = {}
        self.estimated_means = {}
        self.prev_embedding_centers = []
        for name, param in self.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param).to(globals.DEVICE)
            self.estimated_means[name] = torch.zeros_like(param).to(globals.DEVICE)

    def forward(self, x):
        #x = self.imgdr(x)
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
    
    def get_embeddings(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = self.dr1(x)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dr2(x)
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
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
    
    def freeze_prev_tasks_in_last_layer(self):
        self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-1].grad = torch.zeros_like(self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-1])
        self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-1].grad = torch.zeros_like(self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-1])

    def copyPrev(self, prevModel):
        self.conv1.weight = copy.deepcopy(prevModel.conv1.weight)
        self.conv1.bias = copy.deepcopy(prevModel.conv1.bias)
        self.conv2.weight = copy.deepcopy(prevModel.conv2.weight)
        self.conv2.bias = copy.deepcopy(prevModel.conv2.bias)
        self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
        self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
        self.fc2.weight[:self.n_classes - globals.CLASSES_PER_ITER-1] = copy.deepcopy(prevModel.fc2.weight)
        self.fc2.bias[:self.n_classes - globals.CLASSES_PER_ITER-1] = copy.deepcopy(prevModel.fc2.bias)
        self.fisher_information = {}
        self.estimated_means = {}
        self.prev_embedding_centers = prevModel.prev_embedding_centers
        self.n_embeddings = prevModel.n_embeddings

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