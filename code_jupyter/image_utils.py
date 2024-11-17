import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import random

# Define a function to visualize a single image
def show_image(img):
    plt.imshow(img.cpu(), cmap="gray")
    plt.axis("off")
    plt.show()

def random_square_dropout(img, square_size=8):
        channels, height, width = img.size()

        mask = torch.ones_like(img)
        y = torch.randint(0, height - square_size + 1, (1,)).item()
        x = torch.randint(0, width - square_size + 1, (1,)).item()

        # Set pixels within the square to 0 (assuming 0 is the lowest value in the image)
        mask[:,y:y + square_size, x:x + square_size] = 0
        img = img * mask
        return img

def flip_top_half(image_tensor):
    """
    Flips the top half of an image tensor vertically.

    Parameters:
    - image_tensor (torch.Tensor): Image tensor of shape (C, H, W)

    Returns:
    - torch.Tensor: Image tensor with the top half flipped
    """
    # Get the height of the image
    _, height, _ = image_tensor.shape
    
    # Calculate the midpoint of the height
    mid = height // 2
    
    # Flip the top half of the image vertically
    top_half_flipped = torch.flip(image_tensor[:, :mid, :], dims=[1])
    
    # Concatenate the flipped top half with the original bottom half
    return torch.cat((top_half_flipped, image_tensor[:, mid:, :]), dim=1)
    
def flip_bottom_half(image_tensor):
    """
    Flips the bottom half of an image tensor vertically.

    Parameters:
    - image_tensor (torch.Tensor): Image tensor of shape (C, H, W)

    Returns:
    - torch.Tensor: Image tensor with the bottom half flipped
    """
    # Get the height of the image
    _, height, _ = image_tensor.shape
    
    # Calculate the midpoint of the height
    mid = height // 2
    
    # Flip the bottom half of the image vertically
    bottom_half_flipped = torch.flip(image_tensor[:, mid:, :], dims=[1])
    
    # Concatenate the original top half with the flipped bottom half
    flipped_image = torch.cat((image_tensor[:, :mid, :], bottom_half_flipped), dim=1)
    
    return flipped_image

def randomize(img):
        '''
        Attempt to randomize an image by breaking it down into chunks, applying random transformations and piecing them back together
        '''
        should_blacken = random.uniform(0,1)
        if should_blacken > 0.5:
                img = random_square_dropout(img, img.shape[1]//2)
        image_tensor = img
        C, H, W = image_tensor.shape
        
        # Calculate the midpoint for dividing the image into quarters
        mid_H = H // 2
        mid_W = W // 2
        
        # Extract image quarters
        top_left = image_tensor[:, :mid_H, :mid_W]
        top_right = image_tensor[:, :mid_H, mid_W:]
        bottom_left = image_tensor[:, mid_H:, :mid_W]
        bottom_right = image_tensor[:, mid_H:, mid_W:]
        
        # Randomly flip, rotate, or swap each quarter
        quarters = [top_left, top_right, bottom_left, bottom_right]
        transformed_quarters = []
        
        for quarter in quarters:
                # Apply random transformations to each quarter
                if random.random() > 0.5:
                        quarter = torch.flip(quarter, dims=[1])  # Vertical flip
                if random.random() > 0.5:
                        quarter = torch.flip(quarter, dims=[2])  # Horizontal flip
                if random.random() > 0.5:
                        angle = random.uniform(-180, 180)
                        quarter = transforms.functional.rotate(quarter, angle)
                
                # Add to transformed list
                transformed_quarters.append(quarter)
        
        # Randomly shuffle the transformed quarters
        random.shuffle(transformed_quarters)
        
        # Reconstruct the image by combining the transformed quarters
        top_half = torch.cat((transformed_quarters[0], transformed_quarters[1]), dim=2)
        bottom_half = torch.cat((transformed_quarters[2], transformed_quarters[3]), dim=2)
        transformed_image = torch.cat((top_half, bottom_half), dim=1)
        
        return transformed_image