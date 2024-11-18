import torch.nn as nn
import torch.nn.functional as F
import torch
from .CLModel import CLModel
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
    
class VenEtAlMLP(CLModel):
    def __init__(self, input_size, n_classes):
        super().__init__(n_classes)
        self.imgdr = RandomSquareDropout(8)
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 400)
        self.dr1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(400, 400)
        self.dr2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(400, n_classes)

    def forward(self, x):
        x = self.imgdr(x)
        x = x.view(-1, self.input_size)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        x = self.fc3(x)
        return x
    
    def CLCopy(self, prevModel, task):
        with torch.no_grad():
            self.fc1.weight = copy.deepcopy(prevModel.fc1.weight)
            self.fc1.bias = copy.deepcopy(prevModel.fc1.bias)
            self.fc2.weight = copy.deepcopy(prevModel.fc2.weight)
            self.fc2.bias = copy.deepcopy(prevModel.fc2.bias)
            self.fc3.weight[:task*globals.CLASSES_PER_ITER] = prevModel.fc3.weight.clone()
            self.fc3.bias[:task*globals.CLASSES_PER_ITER] = prevModel.fc3.bias.clone()

class VenEtAlMLPPerTask(CLModel):
    def __init__(self, input_size, n_classes):
        super().__init__(n_classes)
        self.imgdr = RandomSquareDropout(8)
        self.input_size = input_size
        self.classes_per_iter = globals.CLASSES_PER_ITER
        self.heads = int(n_classes / self.classes_per_iter)
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)

        self.fc_extra_layers = nn.ModuleList([nn.Linear(400, 8) for _ in range(self.heads)])
        self.fc3 = nn.ModuleList([nn.Linear(8, self.classes_per_iter) for _ in range(self.heads)])

    def forward(self, x):
        x = self.imgdr(x)
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        task_outputs = []
        for task in range(self.heads):
            task_x = F.relu(self.fc_extra_layers[task](x))
            task_outputs.append(self.fc3[task](task_x))

        return torch.cat(task_outputs, dim=1)

    def CLCopy(self, prevModel, task):
        with torch.no_grad():
            self.fc1.weight.copy_(prevModel.fc1.weight)
            self.fc1.bias.copy_(prevModel.fc1.bias)
            self.fc2.weight.copy_(prevModel.fc2.weight)
            self.fc2.bias.copy_(prevModel.fc2.bias)

            for t in range(task):
                self.fc_extra_layers[t].weight.copy_(prevModel.fc_extra_layers[t].weight)
                self.fc_extra_layers[t].bias.copy_(prevModel.fc_extra_layers[t].bias)
                
                self.fc3[t].weight.copy_(prevModel.fc3[t].weight)
                self.fc3[t].bias.copy_(prevModel.fc3[t].bias)