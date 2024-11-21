import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
import globals

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Taken from:
    https://github.com/KaiyangZhou/pytorch-center-loss
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def compute_saliency_map(model, image, target_class):
    image.requires_grad = True
    model.zero_grad()
    output = model(image)
    target_score = output[0, target_class]
    target_score.backward()
    saliency_map = image.grad.data.abs()
    saliency_map, _ = torch.max(saliency_map, dim=1)
    model.zero_grad()
    return saliency_map.squeeze().detach()

def mask_high_saliency_region(image, saliency_map, threshold=0.5):
    # Apply threshold to saliency map to create mask
    high_saliency_mask = saliency_map > (saliency_map.max() * threshold)
    # Blacken out high-saliency regions
    masked_image = image.clone()
    masked_image[:, high_saliency_mask] = 0
    return masked_image

def mask_top_n_saliency_regions(image, saliency_map, top_n=5, patch_size=5):
    """
    Masks out the top N highest saliency regions in an image.
    
    Parameters:
    - image: The input image tensor.
    - saliency_map: The saliency map corresponding to the image.
    - top_n: Number of high-saliency points to mask.
    - patch_size: Size of the square region to mask around each high-saliency point.
    
    Returns:
    - masked_image: The image with high-saliency regions blacked out.
    """
    # Flatten the saliency map to find the top N saliency values
    _, top_indices = torch.topk(saliency_map.view(-1), k=top_n)
    
    # Convert flat indices back to 2D coordinates
    coords = [(idx // saliency_map.size(1), idx % saliency_map.size(1)) for idx in top_indices]
    
    masked_image = image.clone()
    h, w = saliency_map.size()
    max = builtins.max
    min = builtins.min
    # Mask out an MxM square around each high-saliency point
    for center_h, center_w in coords:
        h_start = max(center_h - patch_size // 2, 0)
        h_end = min(center_h + patch_size // 2, h)
        w_start = max(center_w - patch_size // 2, 0)
        w_end = min(center_w + patch_size // 2, w)
        # Black out the MxM region around the saliency point
        masked_image[:, h_start:h_end, w_start:w_end] = 0
    
    return masked_image

def batch_compute_saliency_maps(model, images, labels):
    """
    Computes saliency maps for a batch of images given their labels.
    """
    #images = images.clone().detach().requires_grad_(True)
    images.requires_grad = True
    model.zero_grad()
    
    # Forward pass to get scores
    outputs = model(images)
    target_scores = outputs.gather(1, labels.view(-1, 1)).squeeze()
    
    # Backward pass to compute gradients w.r.t. each image in the batch
    target_scores.sum().backward()
    saliency_maps = images.grad.data.abs().max(dim=1)[0]  # Get max saliency across color channels
    images.requires_grad = False
    return saliency_maps

def mask_region_with_low_saliency(model, images, labels, patch_size=8):
    """
    Masks regions with low saliency by keeping only a patch_size x patch_size square around the pixel with the highest saliency.
    """
    # Compute saliency maps for the batch
    saliency_maps = batch_compute_saliency_maps(model, images, labels)
    
    # Get the coordinates of the maximum saliency for each image
    max_saliency_indices = torch.argmax(saliency_maps.view(saliency_maps.size(0), -1), dim=1)
    max_y, max_x = max_saliency_indices // saliency_maps.size(2), max_saliency_indices % saliency_maps.size(2)
    
    # Create a grid for coordinates
    y_grid, x_grid = torch.meshgrid(
        torch.arange(images.size(2), device=images.device),
        torch.arange(images.size(3), device=images.device),
        indexing="ij"
    )
    
    # Calculate patch boundaries
    y_min = (max_y - patch_size // 2).clamp(0, images.size(2) - 1)
    y_max = (max_y + patch_size // 2).clamp(0, images.size(2) - 1)
    x_min = (max_x - patch_size // 2).clamp(0, images.size(3) - 1)
    x_max = (max_x + patch_size // 2).clamp(0, images.size(3) - 1)
    
    # Create a mask for each image
    mask = ((y_grid.unsqueeze(0) >= y_min.view(-1, 1, 1)) & 
            (y_grid.unsqueeze(0) < y_max.view(-1, 1, 1)) & 
            (x_grid.unsqueeze(0) >= x_min.view(-1, 1, 1)) & 
            (x_grid.unsqueeze(0) < x_max.view(-1, 1, 1)))
    
    # Apply the mask to keep only the patch with high saliency
    masked_images = images * mask.unsqueeze(1)
    
    return masked_images

def mask_top_n_saliency_regions_batch(images, saliency_maps, top_n=5, patch_size=5):
    """
    Masks high-saliency regions in a batch of images without explicit loops.
    
    Parameters:
    - images: Batch of input images (B, C, H, W).
    - saliency_maps: Saliency maps corresponding to the images (B, H, W).
    - top_n: Number of high-saliency points to mask.
    - patch_size: Size of the square region to mask around each high-saliency point.
    
    Returns:
    - masked_images: The batch of images with high-saliency regions blacked out.
    """
    B, C, H, W = images.shape
    masked_images = images.clone()

    # Get top N indices for each image in the batch
    _, top_indices = torch.topk(saliency_maps.view(B, -1), k=top_n, dim=1)
    
    # Convert flat indices to 2D coordinates
    top_coords = torch.stack([(top_indices // W), (top_indices % W)], dim=-1)  # Shape: (B, top_n, 2)
    
    # Generate a mask for each top saliency location in a batch-wise manner
    for i in range(top_n):
        # Get the center coordinates for each image in the batch
        centers_h = top_coords[:, i, 0]
        centers_w = top_coords[:, i, 1]
        
        # Create masks for each patch, applying broadcasting
        h_range = torch.arange(-patch_size // 2, patch_size // 2, device=images.device).view(1, -1)
        w_range = torch.arange(-patch_size // 2, patch_size // 2, device=images.device).view(-1, 1)
        
        mask_h = (centers_h.view(-1, 1, 1) + h_range).clamp(0, H - 1)
        mask_w = (centers_w.view(-1, 1, 1) + w_range).clamp(0, W - 1)
        
        # Broadcast mask indices to zero out corresponding regions in masked_images
        masked_images[torch.arange(B).view(-1, 1, 1), :, mask_h, mask_w] = 0
    
    return masked_images

def create_zero_mask(model, threshold = 1e-6):
    masks = {}
    for name, param in model.named_parameters():
        # Create a mask for each parameter where values above threshold are 1, others are 0
        masks[name] = (param.abs() < threshold).float()
    return masks

def create_nonzero_mask(model, threshold = 1e-6):
    masks = {}
    for name, param in model.named_parameters():
        # Create a mask for each parameter where values above threshold are 1, others are 0
        masks[name] = (param.abs() > threshold).float()
    return masks

def calculate_nonzero_percentage(model, threshold = 1e-6):
    with torch.no_grad():
        # Initialize counters
        total_params = 0
        nonzero_params = 0

        # Iterate over each parameter in the model
        for param in model.parameters():
            # Count total and non-zero elements in the parameter tensor
            total_params += param.numel()
            nonzero_params += torch.sum(torch.abs(param) > threshold).item()

        # Calculate the percentage of non-zero parameters
        return  nonzero_params / total_params

def apply_mask_to_gradients(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Only allow gradient updates where the mask is zero
                param.grad *= (1 - masks[name])
                
def update_EWC_data(model, dataset, iter, batch_size = 20):
    if iter == 0:
        raise Exception("Iteration must be greater than 0!")
    named_params = {name: param for name, param in model.named_parameters()} # convert iterator to dict
    model.eval()
    log_likelihoods = []
    embeddings_per_class = {}
    non_ood_classes = [(iter-1)*(globals.CLASSES_PER_ITER+globals.OOD_CLASS) + j for j in range(globals.CLASSES_PER_ITER)]
    for c in non_ood_classes:
        embeddings_per_class[c] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 0)
    for data, target in loader:
        data, target = data.to(globals.DEVICE), target.to(globals.DEVICE)
        if globals.OOD_CLASS:
            target += iter-1
            mask = target != (iter-1)*(globals.CLASSES_PER_ITER+globals.OOD_CLASS) + globals.CLASSES_PER_ITER
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
        model.zero_grad()
        output, embeddings = model.get_pred_and_embeddings(data)
        for c in non_ood_classes:
            embeddings_per_class[c].append(embeddings[target == c].detach())
        output, target = output[mask], target[mask]
        output = F.log_softmax(output, dim=1)
        log_likelihoods.append(output[range(len(target)), target])

    log_likelihood = torch.cat(log_likelihoods).mean()
    if len(model.prev_embedding_centers) < iter*globals.CLASSES_PER_ITER:
        for c in non_ood_classes:
            embeddings_per_class[c] = torch.cat(embeddings_per_class[c]).mean(dim=0)
            model.prev_embedding_centers.append(embeddings_per_class[c].to(globals.DEVICE))
    grad_log_likelihood = autograd.grad(log_likelihood, model.parameters())
    names = [name for name, _ in named_params.items()]
    for name, param in zip(names, grad_log_likelihood):
        model.fisher_information[name] = (model.fisher_information[name] + (param.data.clone() ** 2).to(globals.DEVICE))
        model.estimated_means[name] = (named_params[name].data.clone()).to(globals.DEVICE)

def calc_ewc_loss(model):
    loss = torch.tensor(0.0, requires_grad = True)
    for n, p in model.named_parameters():
        _loss = model.fisher_information[n] * (p - model.estimated_means[n])**2
        loss = loss + _loss.sum()
    return loss