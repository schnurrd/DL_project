import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import globals
from image_utils import randomize

class AugmentedOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, alpha=0.5, save_path = "ood_data"):
        self.num_ood_samples = num_ood_samples
        self.ood_label = (iteration+1)*globals.CLASSES_PER_ITER
        self.alpha = alpha
        self.iteration = iteration
        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]
        classes = list(range(self.iteration * globals.CLASSES_PER_ITER, (self.iteration+1) * globals.CLASSES_PER_ITER))
        self.indices_per_class = []
        # Select random images from the chosen class
        for c in classes:
            self.indices_per_class.append([i for i in self.indices if globals.full_trainset.targets[i] == c])
    
    def _generate_ood_sample(self):
        c1 = random.choice(range(len(self.indices_per_class)))
        ind1 = self.indices_per_class[c1][random.choice(range(len(self.indices_per_class[c1])))]
        # Choose one random image from the selected class
        img1, _ = globals.full_trainset[ind1]
        return randomize(img1), self.ood_label
        c2 = c1
        while c2 == c1:
            c2 = random.choice(range(len(self.indices_per_class)))
        ind2 = self.indices_per_class[c2][random.choice(range(len(self.indices_per_class[c2])))]
        img2, _ = globals.full_trainset[ind2]
        choice = random.random()
        if choice < 1/4:
            #show_image(img1[0])
            #show_image(img2[0])
            ood_image = self._cutmix(img1, img2)
        elif choice < 2/4:
            #show_image(img1[0])
            #show_image(img2[0])
            ood_image = self._mixup(img1, img2)
        elif choice < 3/4:
            #show_image(img1[0])
            ood_image = self._randomize(img1)
        else:
            ood_image = self._cutmix(self.random_image_like(img1), self.random_image_like(img1))
        return (ood_image.to(dtype=torch.float32), self.ood_label)
        #show_image(ood_image[0])
        #print('\n\n\n')
        return ood_image.to(dtype=torch.float32), self.ood_label

    def random_image_like(self, img):
        choice = random.random()
        total_choices = 4
        if choice < 1/total_choices:
            return self.random_noise(img)
        elif choice < 2/total_choices:
            return self.wavy_pattern(img.shape)
        elif choice < 3/total_choices:
            return self.checkerboard_pattern(img.shape)
        else:
            return self.rotated_sinusoidal_grid(img.shape)
        
    def wavy_pattern(self, img_shape, min_frequency=0.1, max_frequency=1.0):
        #   Generate a random frequency between the specified min and max
        frequency = np.random.uniform(min_frequency, max_frequency)
        angle = np.random.uniform(0, np.pi)
        
        # Generate x and y coordinates based on the image shape
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Calculate the wave pattern based on the frequency and angle
        wave_pattern = 0.5 * (1 + np.sin(frequency * (X * np.cos(angle) + Y * np.sin(angle))))
        
        return torch.tensor(wave_pattern).unsqueeze(0)
    
    def checkerboard_pattern(self, img_shape):
        # Define a base size for squares with variability
        base_size_x = np.random.randint(min(img_shape[1:]) // 8, min(img_shape[1:]) // 4)
        base_size_y = np.random.randint(min(img_shape[1:]) // 8, min(img_shape[1:]) // 4)

        # Generate x and y coordinates
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)

        # Generate random square sizes
        square_size_x = base_size_x + np.random.randint(-base_size_x // 2, base_size_x // 2)
        square_size_y = base_size_y + np.random.randint(-base_size_y // 2, base_size_y // 2)

        # Rotate coordinates
        angle = np.random.uniform(0, np.pi)
        X_centered = X - img_shape[2] / 2
        Y_centered = Y - img_shape[1] / 2
        X_rotated = X_centered * np.cos(angle) - Y_centered * np.sin(angle)
        Y_rotated = X_centered * np.sin(angle) + Y_centered * np.cos(angle)

        # Calculate the sine wave to create smooth edges
        X_wave = 0.5 * (1 + np.cos((2 * np.pi * X_rotated / square_size_x) % (2 * np.pi)))
        Y_wave = 0.5 * (1 + np.cos((2 * np.pi * Y_rotated / square_size_y) % (2 * np.pi)))

        # Combine the waves to get a smooth checkerboard pattern
        checkerboard = X_wave * Y_wave

        return torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0)

    def rotated_sinusoidal_grid(self, img_shape, min_frequency=0.1, max_frequency=1.0):
        # Randomly choose frequencies and rotation angle
        frequency = np.random.uniform(min_frequency, max_frequency)
        angle = np.random.uniform(0, np.pi)  # Angle in radians between 0 and π

        # Generate x and y coordinates
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates by the chosen angle
        X_centered = X - img_shape[2] / 2
        Y_centered = Y - img_shape[1] / 2
        X_rotated = X_centered * np.cos(angle) - Y_centered * np.sin(angle)
        Y_rotated = X_centered * np.sin(angle) + Y_centered * np.cos(angle)

        # Apply sinusoidal pattern to the rotated coordinates
        grid_pattern = 0.5 * (1 + np.sin(frequency * X_rotated) * np.sin(frequency * Y_rotated))

        return torch.tensor(grid_pattern).unsqueeze(0)

    def _cutmix(self, img1, img2):
        def map_value(x, a, b, c, d):
            return c + (x - a) * (d - c) / (b - a)
        transition_width = 4
        _, H, W = img1.shape

        # Randomly choose whether to split into two or four sections
        split_type = random.choice(['two', 'four'])
        ood_image = img1.clone()

        if split_type == 'two':
            # Decide on vertical or horizontal split
            split_direction = random.choice(['vertical', 'horizontal'])
            
            # Choose a cutoff between 1/3 and 2/3 of the way across the chosen dimension
            cutoff_ratio = random.uniform(1/3, 2/3)

            if split_direction == 'vertical':
                cutoff = int(W * cutoff_ratio)
                for i in range(W):
                    alpha = np.clip((i - cutoff + transition_width) / (2 * transition_width), 0, 1)
                    ood_image[:, :, i] = (1 - alpha) * img1[:, :, i] + alpha * img2[:, :, i]
            else:  # horizontal
                cutoff = int(H * cutoff_ratio)
                for i in range(H):
                    alpha = np.clip((i - cutoff + transition_width) / (2 * transition_width), 0, 1)
                    ood_image[:, i, :] = (1 - alpha) * img1[:, i, :] + alpha * img2[:, i, :]

        if split_type == 'four':
            # Choose cutoffs for both vertical and horizontal splits
            cutoff_w = int(W * random.uniform(1/3, 2/3))
            cutoff_h = int(H * random.uniform(1/3, 2/3))

            # Assign each quadrant
            ood_image[:, :cutoff_h - transition_width, :cutoff_w - transition_width] = img1[:, :cutoff_h - transition_width, :cutoff_w - transition_width]  # Top-left
            ood_image[:, :cutoff_h - transition_width, cutoff_w + transition_width:] = img2[:, :cutoff_h - transition_width, cutoff_w + transition_width:]  # Top-right
            ood_image[:, cutoff_h + transition_width:, :cutoff_w - transition_width] = img2[:, cutoff_h + transition_width:, :cutoff_w - transition_width]  # Bottom-left
            ood_image[:, cutoff_h + transition_width:, cutoff_w + transition_width:] = img1[:, cutoff_h + transition_width:, cutoff_w + transition_width:]  # Bottom-right

            # Apply smooth transition for vertical blending region
            for i in range(cutoff_w - transition_width, cutoff_w + transition_width):
                alpha = map_value(i, cutoff_w - transition_width, cutoff_w + transition_width - 1, 0, 1)
                ood_image[:, :cutoff_h - transition_width, i] = (
                    alpha * img2[:, :cutoff_h - transition_width, i] + (1 - alpha) * img1[:, :cutoff_h - transition_width, i]
                )
                ood_image[:, cutoff_h + transition_width:, i] = (
                    alpha * img1[:, cutoff_h + transition_width:, i] + (1 - alpha) * img2[:, cutoff_h + transition_width:, i]
                )

            # Apply smooth transition for horizontal blending region
            for j in range(cutoff_h - transition_width, cutoff_h + transition_width):
                alpha = map_value(j, cutoff_h - transition_width, cutoff_h + transition_width - 1, 0, 1)
                ood_image[:, j, :cutoff_w - transition_width] = (
                    alpha * img2[:, j, :cutoff_w - transition_width] + (1 - alpha) * img1[:, j, :cutoff_w - transition_width]
                )
                ood_image[:, j, cutoff_w + transition_width:] = (
                    alpha * img1[:, j, cutoff_w + transition_width:] + (1 - alpha) * img2[:, j, cutoff_w + transition_width:]
                )

            # Apply blending for the central region where all four quadrants meet
            for i in range(cutoff_w - transition_width, cutoff_w + transition_width):
                for j in range(cutoff_h - transition_width, cutoff_h + transition_width):
                    alpha_w = map_value(i, cutoff_w - transition_width, cutoff_w + transition_width - 1, 0, 1)
                    alpha_h = map_value(j, cutoff_h - transition_width, cutoff_h + transition_width - 1, 0, 1)

                    # Compute the weighted blend of all four quadrants
                    top_left = (1 - alpha_w) * (1 - alpha_h) * img1[:, j, i]
                    top_right = alpha_w * (1 - alpha_h) * img2[:, j, i]
                    bottom_left = (1 - alpha_w) * alpha_h * img2[:, j, i]
                    bottom_right = alpha_w * alpha_h * img1[:, j, i]

                    # Set the central region pixel to the blended value
                    ood_image[:, j, i] = top_left + top_right + bottom_left + bottom_right

        return ood_image

    def _mixup(self, img1, img2):
        ood_image = self.alpha * img1 + (1 - self.alpha) * img2
        ood_image = torch.clamp(ood_image, 0, 1)
        return ood_image

    def _save_ood_data(self):
        torch.save(self.ood_data, self.save_path)
        print(f"OOD data saved to {self.save_path}")

    def _randomize(self, img):
        return randomize(img)

    def random_noise(self, img):
        # Generate random values between 0 and 1 with the same shape as img
        noise = torch.tensor(np.random.rand(*img.shape))
        return noise

    def __len__(self):
        # Original dataset length + OOD samples
        return self.original_length# + self.num_ood_samples

    def __getitem__(self, idx):
        # Return original dataset sample or OOD sample
        if torch.rand(1).item() < 1/(globals.CLASSES_PER_ITER+1):
            return randomize(globals.full_trainset[self.indices[idx]][0]), self.ood_label
        else:
            return globals.full_trainset[self.indices[idx]]
        if idx < self.original_length:
            return globals.full_trainset[self.indices[idx]]
        else:
            return self._generate_ood_sample()

'''
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

def get_base_dataset(dataset):
    """Recursively retrieve the base dataset from any nested Subset objects."""
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset

class AugmentedOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, alpha=0.5, save_path = "ood_data"):
        self.num_ood_samples = num_ood_samples
        self.ood_label = (iteration+1)*globals.CLASSES_PER_ITER
        self.alpha = alpha
        self.iteration = iteration
        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]
        classes = list(range(self.iteration * globals.CLASSES_PER_ITER, (self.iteration+1) * globals.CLASSES_PER_ITER))
        self.indices_per_class = []
        # Select random images from the chosen class
        for c in classes:
            self.indices_per_class.append([i for i in self.indices if globals.full_trainset.targets[i] == c])
    
    def _generate_ood_sample(self):
        c1 = random.choice(range(len(self.indices_per_class)))
        ind1 = self.indices_per_class[c1][random.choice(range(len(self.indices_per_class[c1])))]
        # Choose one random image from the selected class
        img1, _ = globals.full_trainset[ind1]
        # Apply augmentation randomly
        #return randomize(img1), self.ood_label
        c2 = c1
        while c2 == c1:
            c2 = random.choice(range(len(self.indices_per_class)))
        ind2 = self.indices_per_class[c2][random.choice(range(len(self.indices_per_class[c2])))]
        img2, _ = globals.full_trainset[ind2]
        choice = random.random()
        if choice < 1/4:
            #show_image(img1[0])
            #show_image(img2[0])
            ood_image = self._cutmix(img1, img2)
        elif choice < 2/4:
            #show_image(img1[0])
            #show_image(img2[0])
            ood_image = self._mixup(img1, img2)
        elif choice < 3/4:
            #show_image(img1[0])
            ood_image = self._randomize(img1)
        else:
            ood_image = self._cutmix(self.random_image_like(img1), self.random_image_like(img1))
        return (ood_image.to(dtype=torch.float32), self.ood_label)
        #show_image(ood_image[0])
        #print('\n\n\n')
        return ood_image.to(dtype=torch.float32), self.ood_label

    def random_image_like(self, img):
        choice = random.random()
        total_choices = 4
        if choice < 1/total_choices:
            return self.random_noise(img)
        elif choice < 2/total_choices:
            return self.wavy_pattern(img.shape)
        elif choice < 3/total_choices:
            return self.checkerboard_pattern(img.shape)
        else:
            return self.rotated_sinusoidal_grid(img.shape)
        
    def wavy_pattern(self, img_shape, min_frequency=0.1, max_frequency=1.0):
        #   Generate a random frequency between the specified min and max
        frequency = np.random.uniform(min_frequency, max_frequency)
        angle = np.random.uniform(0, np.pi)
        
        # Generate x and y coordinates based on the image shape
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Calculate the wave pattern based on the frequency and angle
        wave_pattern = 0.5 * (1 + np.sin(frequency * (X * np.cos(angle) + Y * np.sin(angle))))
        
        return torch.tensor(wave_pattern).unsqueeze(0)
    
    def checkerboard_pattern(self, img_shape):
        # Define a base size for squares with variability
        base_size_x = np.random.randint(min(img_shape[1:]) // 8, min(img_shape[1:]) // 4)
        base_size_y = np.random.randint(min(img_shape[1:]) // 8, min(img_shape[1:]) // 4)

        # Generate x and y coordinates
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)

        # Generate random square sizes
        square_size_x = base_size_x + np.random.randint(-base_size_x // 2, base_size_x // 2)
        square_size_y = base_size_y + np.random.randint(-base_size_y // 2, base_size_y // 2)

        # Rotate coordinates
        angle = np.random.uniform(0, np.pi)
        X_centered = X - img_shape[2] / 2
        Y_centered = Y - img_shape[1] / 2
        X_rotated = X_centered * np.cos(angle) - Y_centered * np.sin(angle)
        Y_rotated = X_centered * np.sin(angle) + Y_centered * np.cos(angle)

        # Calculate the sine wave to create smooth edges
        X_wave = 0.5 * (1 + np.cos((2 * np.pi * X_rotated / square_size_x) % (2 * np.pi)))
        Y_wave = 0.5 * (1 + np.cos((2 * np.pi * Y_rotated / square_size_y) % (2 * np.pi)))

        # Combine the waves to get a smooth checkerboard pattern
        checkerboard = X_wave * Y_wave

        return torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0)

    def rotated_sinusoidal_grid(self, img_shape, min_frequency=0.1, max_frequency=1.0):
        # Randomly choose frequencies and rotation angle
        frequency = np.random.uniform(min_frequency, max_frequency)
        angle = np.random.uniform(0, np.pi)  # Angle in radians between 0 and π

        # Generate x and y coordinates
        x = np.arange(img_shape[2])
        y = np.arange(img_shape[1])
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates by the chosen angle
        X_centered = X - img_shape[2] / 2
        Y_centered = Y - img_shape[1] / 2
        X_rotated = X_centered * np.cos(angle) - Y_centered * np.sin(angle)
        Y_rotated = X_centered * np.sin(angle) + Y_centered * np.cos(angle)

        # Apply sinusoidal pattern to the rotated coordinates
        grid_pattern = 0.5 * (1 + np.sin(frequency * X_rotated) * np.sin(frequency * Y_rotated))

        return torch.tensor(grid_pattern).unsqueeze(0)

    def _cutmix(self, img1, img2):
        def map_value(x, a, b, c, d):
            return c + (x - a) * (d - c) / (b - a)
        transition_width = 4
        _, H, W = img1.shape

        # Randomly choose whether to split into two or four sections
        split_type = random.choice(['two', 'four'])
        ood_image = img1.clone()

        if split_type == 'two':
            # Decide on vertical or horizontal split
            split_direction = random.choice(['vertical', 'horizontal'])
            
            # Choose a cutoff between 1/3 and 2/3 of the way across the chosen dimension
            cutoff_ratio = random.uniform(1/3, 2/3)

            if split_direction == 'vertical':
                cutoff = int(W * cutoff_ratio)
                for i in range(W):
                    alpha = np.clip((i - cutoff + transition_width) / (2 * transition_width), 0, 1)
                    ood_image[:, :, i] = (1 - alpha) * img1[:, :, i] + alpha * img2[:, :, i]
            else:  # horizontal
                cutoff = int(H * cutoff_ratio)
                for i in range(H):
                    alpha = np.clip((i - cutoff + transition_width) / (2 * transition_width), 0, 1)
                    ood_image[:, i, :] = (1 - alpha) * img1[:, i, :] + alpha * img2[:, i, :]

        if split_type == 'four':
            # Choose cutoffs for both vertical and horizontal splits
            cutoff_w = int(W * random.uniform(1/3, 2/3))
            cutoff_h = int(H * random.uniform(1/3, 2/3))

            # Assign each quadrant
            ood_image[:, :cutoff_h - transition_width, :cutoff_w - transition_width] = img1[:, :cutoff_h - transition_width, :cutoff_w - transition_width]  # Top-left
            ood_image[:, :cutoff_h - transition_width, cutoff_w + transition_width:] = img2[:, :cutoff_h - transition_width, cutoff_w + transition_width:]  # Top-right
            ood_image[:, cutoff_h + transition_width:, :cutoff_w - transition_width] = img2[:, cutoff_h + transition_width:, :cutoff_w - transition_width]  # Bottom-left
            ood_image[:, cutoff_h + transition_width:, cutoff_w + transition_width:] = img1[:, cutoff_h + transition_width:, cutoff_w + transition_width:]  # Bottom-right

            # Apply smooth transition for vertical blending region
            for i in range(cutoff_w - transition_width, cutoff_w + transition_width):
                alpha = map_value(i, cutoff_w - transition_width, cutoff_w + transition_width - 1, 0, 1)
                ood_image[:, :cutoff_h - transition_width, i] = (
                    alpha * img2[:, :cutoff_h - transition_width, i] + (1 - alpha) * img1[:, :cutoff_h - transition_width, i]
                )
                ood_image[:, cutoff_h + transition_width:, i] = (
                    alpha * img1[:, cutoff_h + transition_width:, i] + (1 - alpha) * img2[:, cutoff_h + transition_width:, i]
                )

            # Apply smooth transition for horizontal blending region
            for j in range(cutoff_h - transition_width, cutoff_h + transition_width):
                alpha = map_value(j, cutoff_h - transition_width, cutoff_h + transition_width - 1, 0, 1)
                ood_image[:, j, :cutoff_w - transition_width] = (
                    alpha * img2[:, j, :cutoff_w - transition_width] + (1 - alpha) * img1[:, j, :cutoff_w - transition_width]
                )
                ood_image[:, j, cutoff_w + transition_width:] = (
                    alpha * img1[:, j, cutoff_w + transition_width:] + (1 - alpha) * img2[:, j, cutoff_w + transition_width:]
                )

            # Apply blending for the central region where all four quadrants meet
            for i in range(cutoff_w - transition_width, cutoff_w + transition_width):
                for j in range(cutoff_h - transition_width, cutoff_h + transition_width):
                    alpha_w = map_value(i, cutoff_w - transition_width, cutoff_w + transition_width - 1, 0, 1)
                    alpha_h = map_value(j, cutoff_h - transition_width, cutoff_h + transition_width - 1, 0, 1)

                    # Compute the weighted blend of all four quadrants
                    top_left = (1 - alpha_w) * (1 - alpha_h) * img1[:, j, i]
                    top_right = alpha_w * (1 - alpha_h) * img2[:, j, i]
                    bottom_left = (1 - alpha_w) * alpha_h * img2[:, j, i]
                    bottom_right = alpha_w * alpha_h * img1[:, j, i]

                    # Set the central region pixel to the blended value
                    ood_image[:, j, i] = top_left + top_right + bottom_left + bottom_right

        return ood_image

    def _mixup(self, img1, img2):
        ood_image = self.alpha * img1 + (1 - self.alpha) * img2
        ood_image = torch.clamp(ood_image, 0, 1)
        return ood_image

    def _save_ood_data(self):
        torch.save(self.ood_data, self.save_path)
        print(f"OOD data saved to {self.save_path}")

    def _randomize(self, img):
        return randomize(img)

    def random_noise(self, img):
        # Generate random values between 0 and 1 with the same shape as img
        noise = torch.tensor(np.random.rand(*img.shape))
        return noise

    def __len__(self):
        # Original dataset length + OOD samples
        return self.original_length# + self.num_ood_samples

    def __getitem__(self, idx):
        # Return original dataset sample or OOD sample
        if torch.rand(1).item() < 1/3:
            return self._generate_ood_sample()
        else:
            return globals.full_trainset[self.indices[idx]]
        if idx < self.original_length:
            return globals.full_trainset[self.indices[idx]]
        else:
            return self._generate_ood_sample()
'''
