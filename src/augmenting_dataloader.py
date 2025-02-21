import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
import math
import numpy as np
import globals
from image_utils import randomize
import matplotlib.pyplot as plt
from scipy.stats import beta

class CutMixOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, centered = False):
        self.itetarion = iteration
        self.num_ood_samples = num_ood_samples
        self.centered = centered
        self.ood_label = (iteration+1)*globals.CLASSES_PER_ITER
        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.trainloader = globals.trainloaders[iteration].dataset
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]

        self.class_groups = self.__group_by_class()
        #self.ood_samples = self.__generate_cutmix()
        #self.combined_data = self.__combine_and_shuffle()

    def __group_by_class(self):
        class_groups = {} 
        for image, label in self.trainloader:
            if not label in class_groups:
                class_groups[label] = [image]
                continue

            class_groups[label].append(image)
        return class_groups

    def __cutmix(self, img1, img2, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        
        # Get dimensions of the image
        H, W = img1.shape[1], img1.shape[2]

        # Randomly generate the bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        if self.centered:
            cut_w = max(cut_w, W // 6)
            cut_h = max(cut_h, H // 6)
        
        # Uniformly sample the center of the rectangle
        if self.centered:
            cx = np.random.randint(W // 2) + W // 4 
            cy = np.random.randint(H // 2) + H // 4 
        else:
            cx = np.random.randint(W)
            cy = np.random.randint(H)

        # Calculate the bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        img1[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]

        return img1
    
    def __random_cutmix(self):
        label1, label2 = random.sample(list(self.class_groups.keys()), 2)

        image1 = self.class_groups[label1][random.randint(0, len(self.class_groups[label1]) - 1)]
        image2 = self.class_groups[label2][random.randint(0, len(self.class_groups[label2]) - 1)]

        return self.__cutmix(image1, image2)
    
    def display_ood_samples(self, num_samples=20, filename="cutmix_ood_samples.png"):
        """
        Save the first num_samples OOD samples to a file.
        """
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        for i in range(num_samples):
            if num_samples == 1:
                ax = axes
            else:
                ax = axes[i]
            image = self.__random_cutmix()
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # Adjust as necessary for your tensor format
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory

    def __len__(self):
        return self.num_ood_samples + self.original_length

    def __getitem__(self, index):
        if index < self.original_length:
            return globals.full_trainset[self.indices[index]]
        else:
            ood_image = self.__random_cutmix()
            return ood_image, self.ood_label

class JigsawOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, num_tiles = 4, random_patches = False):
        """
        Args:
            iteration (int): The current iteration index.
            num_ood_samples (int): Number of Jigsaw OOD samples to generate.
            num_tiles (int): The number of tiles to divide each row/column.
            random_patches: If true, the number of patches that the image is divided in will be randomly chosen
        """
        self.iteration = iteration
        self.num_ood_samples = num_ood_samples
        self.num_tiles = num_tiles 

        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.trainloader = globals.trainloaders[iteration].dataset
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]
        self.ood_label = (iteration + 1) * globals.CLASSES_PER_ITER
        self.shape = self.trainloader[0][0].shape[1:]
        self.random_patches = random_patches

        assert self.shape[0] / self.num_tiles == self.shape[0] // self.num_tiles 
        assert self.shape[1] / self.num_tiles == self.shape[1] // self.num_tiles 

        self.num_tiles_random = []

        if random_patches:
            for tiles in [x for x in range(2, int(np.sqrt(min(self.shape[0], self.shape[1]))))]:
                if self.shape[0] / tiles == self.shape[0] // tiles and self.shape[1] / tiles == self.shape[1] // tiles:
                    self.num_tiles_random.append(tiles)


    def __generate_jigsaw(self, image):
        image = np.array(image)

        num_tiles = self.num_tiles
        if self.random_patches:
            num_tiles = random.choice(self.num_tiles_random)

        _, H, W = image.shape
        tile_h, tile_w = H // num_tiles, W // num_tiles
        tiles = []

        for i in range(num_tiles):
            for j in range(num_tiles):
                tile = image[:, i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                tiles.append(tile)

        identity_permutation = np.arange(len(tiles))
        while True:
            permutation = np.random.permutation(len(tiles))
            if not np.array_equal(permutation, identity_permutation):
                break
        permutation = np.random.permutation(len(tiles))
        permuted_tiles = [tiles[permutation[t]] for t in range(len(tiles))]
        permuted_image = np.block([[permuted_tiles[i * num_tiles + j] for j in range(num_tiles)]
                                    for i in range(num_tiles)])

        return torch.tensor(permuted_image)
    
    def __random_jigsaw(self):
        image, _ = self.trainloader[random.randint(0, self.num_ood_samples - 1)]
        return self.__generate_jigsaw(image)

    def display_ood_samples(self, num_samples=20, filename="jigsaw_samples.png"):
        """
        Save the first num_samples OOD samples to a file.
        """
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        for i in range(num_samples):
            if num_samples == 1:
                ax = axes
            else:
                ax = axes[i]

            image = self.__random_jigsaw()
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # Adjust as necessary for your tensor format
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory

    def __len__(self):
        return self.num_ood_samples + self.original_length

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        if index < self.original_length:
            return globals.full_trainset[self.indices[index]]
        else:
            return self.__random_jigsaw(), self.ood_label

class FMixOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, alpha=1, decay_power=2, max_soft=0.0):
        """
        FMix Dataset for Out-of-Distribution Data Augmentation.

        Args:
            iteration (int): Current iteration.
            num_ood_samples (int): Number of OOD samples to generate.
            alpha (float): Alpha value for Beta distribution.
            decay_power (float): Decay power for frequency decay.
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
        """
        self.iteration = iteration
        self.num_ood_samples = num_ood_samples
        self.alpha = alpha
        self.decay_power = decay_power
        self.max_soft = max_soft

        self.ood_label = (iteration + 1) * globals.CLASSES_PER_ITER
        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.trainloader = globals.trainloaders[iteration].dataset
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]
        self.shape = self.trainloader[0][0].shape[1:]
        self.class_groups = self.__group_by_class()

    def __group_by_class(self):
        class_groups = {}
        for image, label in self.trainloader:
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(image)
        return class_groups

    def __random_fmix(self):
        # Sample two random classes
        label1, label2 = random.sample(list(self.class_groups.keys()), 2)
        # Randomly select one image from each class
        image1 = self.class_groups[label1][random.randint(0, len(self.class_groups[label1]) - 1)]
        image2 = self.class_groups[label2][random.randint(0, len(self.class_groups[label2]) - 1)]

        # Generate FMix mask and mixed image
        _, mask = self.__sample_mask()
        fmix_image = image1 * mask + image2 * (1 - mask)
        return fmix_image, mask

    def __sample_mask(self):
        lam = self.__sample_lam()
        mask = self.__make_low_freq_mask(lam)
        return lam, mask

    def __sample_lam(self):
        return max(0.2, min(0.8, beta.rvs(self.alpha, self.alpha)))
        #return beta.rvs(self.alpha, self.alpha)

    def __make_low_freq_mask(self, lam):
        freqs = self.__fftfreqnd(*self.shape)
        spectrum = self.__get_spectrum(freqs)
        spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
        mask = np.real(np.fft.irfftn(spectrum, self.shape))

        if len(self.shape) == 1:
            mask = mask[:1, :self.shape[0]]
        if len(self.shape) == 2:
            mask = mask[:1, :self.shape[0], :self.shape[1]]
        if len(self.shape) == 3:
            mask = mask[:1, :self.shape[0], :self.shape[1], :self.shape[2]]

        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        #mask = (mask - mask.min()) / mask.max()
        mask = self.__binarize_mask(mask, lam)
        return torch.tensor(mask, dtype=torch.float32)

    def __fftfreqnd(self, h, w, z=None):
        """ Get bin values for discrete fourier transform of size (h, w, z)

        :param h: Required, first dimension size
        :param w: Optional, second dimension size
        :param z: Optional, third dimension size
        """
        fz = fx = 0
        fy = np.fft.fftfreq(h)

        if w is not None:
            fy = np.expand_dims(fy, -1)

            if w % 2 == 1:
                fx = np.fft.fftfreq(w)[: w // 2 + 2]
            else:
                fx = np.fft.fftfreq(w)[: w // 2 + 1]

        if z is not None:
            fy = np.expand_dims(fy, -1)
            if z % 2 == 1:
                fz = np.fft.fftfreq(z)[:, None]
            else:
                fz = np.fft.fftfreq(z)[:, None]

        return np.sqrt(fx * fx + fy * fy + fz * fz)

    def __get_spectrum(self, freqs):
        scale = np.ones(1) / (np.maximum(freqs, 1. / max(self.shape)) ** self.decay_power)
        param_size = [1] + list(freqs.shape) + [2]
        param = np.random.randn(*param_size)
        scale = np.expand_dims(scale, -1)[None, :]
        return scale * param

    def __binarize_mask(self, mask, lam):
        """ Binarizes a given low frequency image such that it has mean lambda.

        Args:
            mask (ndarray): Low frequency image, usually the result of `make_low_freq_image`.
            lam (float): Mean value of the final mask.
        
        Returns:
            torch.Tensor: Binarized mask reshaped to the required dimensions.
        """
        idx = mask.reshape(-1).argsort()[::-1]  # Sort indices in descending order of mask values
        mask = mask.reshape(-1)
        
        # Determine the number of "1s" in the mask based on lambda
        num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)
        
        # Compute the effective softening
        eff_soft = self.max_soft
        if self.max_soft > lam or self.max_soft > (1 - lam):
            eff_soft = min(lam, 1 - lam)

        # Soft boundaries around the binarization
        soft = int(mask.size * eff_soft)
        num_low = num - soft
        num_high = num + soft

        # Set values in the mask
        mask[idx[:num_high]] = 1
        mask[idx[num_low:]] = 0
        if num_high > num_low:
            mask[idx[num_low:num_high]] = np.linspace(1, 0, num_high - num_low)

        # Reshape back to the original shape
        return mask.reshape((1, *self.shape))

    def display_ood_samples(self, num_samples=20, filename="fmix_ood_samples_with_masks.png"):
        """
        Save the first num_samples OOD samples along with their masks to a file.
        """
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))  # Two rows: images and masks
        for i in range(num_samples):
            if num_samples == 1:
                ax_img = axes[0]
                ax_mask = axes[1]
            else:
                ax_img = axes[0, i]
                ax_mask = axes[1, i]

            # Generate an FMix image and mask
            fmix_image, mask = self.__random_fmix()

            # Display FMix image
            ax_img.imshow(fmix_image.permute(1, 2, 0).cpu().numpy())  # Adjust for tensor format
            ax_img.axis('off')
            ax_img.set_title("FMix Image")

            # Display FMix mask
            ax_mask.imshow(mask.squeeze().cpu().numpy(), cmap='gray')  # Grayscale mask
            ax_mask.axis('off')
            ax_mask.set_title("FMix Mask")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def __len__(self):
        return self.num_ood_samples + self.original_length

    def __getitem__(self, index):
        if index < self.original_length:
            return globals.full_trainset[self.indices[index]]
        else:
            ood_image, _ = self.__random_fmix()
            return ood_image, self.ood_label

class SmoothMixOODTrainset(Dataset):
    def __init__(self, iteration, num_ood_samples, mask_type='S', centered = False, max_soft=1.0):
        """
        SmoothMix Dataset for Out-of-Distribution Data Augmentation.

        Args:
            iteration (int): Current iteration.
            num_ood_samples (int): Number of OOD samples to generate.
            mask_type (str): 'S' for square mask, 'C' for circular mask.
            max_soft (float): Softening value between 0 and 0.5 for mask edges.
        """
        self.iteration = iteration
        self.num_ood_samples = num_ood_samples
        self.mask_type = mask_type
        self.max_soft = max_soft
        self.centered = centered

        self.ood_label = (iteration + 1) * globals.CLASSES_PER_ITER
        self.original_length = len(globals.trainloaders[iteration].dataset)
        self.trainloader = globals.trainloaders[iteration].dataset
        self.indices = [globals.trainset.indices[i] for i in globals.trainloaders[iteration].dataset.indices]
        self.shape = self.trainloader[0][0].shape[1:]  # Infer size from the dataset images
        self.class_groups = self.__group_by_class()

    def __group_by_class(self):
        class_groups = {}
        for image, label in self.trainloader:
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(image)
        return class_groups

    def __random_smoothmix(self):
        label1, label2 = random.sample(list(self.class_groups.keys()), 2)
        image1 = self.class_groups[label1][random.randint(0, len(self.class_groups[label1]) - 1)]
        image2 = self.class_groups[label2][random.randint(0, len(self.class_groups[label2]) - 1)]

        # Generate SmoothMix mask and mixed image
        mask = self.__generate_mask()
        smoothmix_image = image1 * mask + image2 * (1 - mask)
        return smoothmix_image, mask

    def __generate_mask(self):
        if self.mask_type == 'S':
            return self.__generate_square_mask()
        elif self.mask_type == 'C':
            return self.__generate_circular_mask()
        else:
            raise ValueError("Invalid mask_type. Choose 'S' for square or 'C' for circular.")

    def __generate_square_mask(self):
        H, W = self.shape

        if self.centered:
            center_x = random.randint(0, W // 2) + W // 4
            center_y = random.randint(0, H // 2) + H // 4
        else:
            center_x = random.randint(0, W)
            center_y = random.randint(0, H)
        width = random.uniform(0.2, 0.5) * W  # Random square width
        height = random.uniform(0.2, 0.5) * H  # Random square height

        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)

        dist_x = np.maximum(0, np.abs(xx - center_x) - width / 2)
        dist_y = np.maximum(0, np.abs(yy - center_y) - height / 2)

        mask = np.maximum(0, 1 - (dist_x + dist_y) / self.max_soft)

        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def __generate_circular_mask(self):
        H, W = self.shape

        if self.centered:
            center_x = random.randint(0, W // 2) + W // 4
            center_y = random.randint(0, H // 2) + H // 4
        else:
            center_x = random.randint(0, W)
            center_y = random.randint(0, H)

        sigma = random.uniform(0.25, 0.4) * max(H, W)

        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)

        dist_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2

        mask = np.exp(-dist_squared / (2 * sigma ** 2))

        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def display_ood_samples(self, num_samples=20, filename="smoothmix_ood_samples.png"):
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))  # Two rows: images and masks
        for i in range(num_samples):
            if num_samples == 1:
                ax_img = axes[0]
                ax_mask = axes[1]
            else:
                ax_img = axes[0, i]
                ax_mask = axes[1, i]

            # Generate SmoothMix image and mask
            smoothmix_image, mask = self.__random_smoothmix()

            # Display SmoothMix image
            ax_img.imshow(smoothmix_image.permute(1, 2, 0).cpu().numpy(), cmap = 'gray')  # Adjust for tensor format
            ax_img.axis('off')
            ax_img.set_title("SmoothMix Image")

            # Display SmoothMix mask
            ax_mask.imshow(mask.squeeze().cpu().numpy(), cmap='gray')  # Grayscale mask
            ax_mask.axis('off')
            ax_mask.set_title("SmoothMix Mask")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def __len__(self):
        return self.num_ood_samples + self.original_length

    def __getitem__(self, index):
        if index < self.original_length:
            return globals.full_trainset[self.indices[index]]
        else:
            ood_image, _ = self.__random_smoothmix()
            return ood_image, self.ood_label


class JointTrainingNoOODTrainset(Dataset):
    def __init__(self, classes_per_iter):
        self.length = len(globals.trainset)
        self.classes_per_iter = classes_per_iter

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        feat, lab = globals.trainset[index]
        return feat, lab + lab//globals.CLASSES_PER_ITER