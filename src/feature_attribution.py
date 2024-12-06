import captum
import math
import random
import copy
from pytorch_utils import get_features, get_labels
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

#[Denis] added class:
class Feature_Importance_Evaluations:
    def __init__(self,Test_Datasets, DEVICE, Attribution_Method="Gradients",background_samples=100):
        
        #print("HP1")

        self.supported_attribution_methods=["GradientShap","Gradients"]
        if Attribution_Method not in self.supported_attribution_methods:
            raise Exception("The following Attribution Method is not supported: "+Attribution_Method)
            
        self.Attribution_Method=Attribution_Method
        self.DEVICE=DEVICE
        self.Test_Datasets_Features=[]
        self.Test_Datasets_Labels=[]
        self.backgrounds=[]
        self.after_training_attributions=None

        
        samples_per_task= math.ceil(background_samples/len(Test_Datasets))
        for Task_Num in range(len(Test_Datasets)):
            self.Test_Datasets_Features.append(get_features(Test_Datasets[Task_Num]).to("cpu"))
            self.Test_Datasets_Labels.append(get_labels(Test_Datasets[Task_Num]).tolist())
            unique_labels = list(set(self.Test_Datasets_Labels[-1]))
            samples_per_label= math.ceil(samples_per_task/len(unique_labels))
            for ac_label in unique_labels:
                found_samples=0
                while found_samples<samples_per_label:
                    ac_point=random.randint(0, self.Test_Datasets_Features[-1].shape[0]-1)
                    if ac_label==self.Test_Datasets_Labels[-1][ac_point]:
                        self.backgrounds.append(self.Test_Datasets_Features[-1][ac_point])
                        found_samples+=1
                        #print(found_samples)
        self.backgrounds = torch.stack(self.backgrounds).to(self.DEVICE)
        

        
        self.Feature_Attributions = [None]*len(Test_Datasets)
        self.attribution_model=None
        #print(self.Test_Datasets_Features[0].shape)
        #print(self.Test_Datasets_Labels[0])
        #print(self.backgrounds.shape)
        #print("HP2")

    def _normalize_tensor(self,ac_ten,abs=True):#Get it into the range of 0 to 1 and summing up to 1
        #another version could be softmax
        ac_ten_abs=torch.abs(ac_ten)
        if abs:
            ac_ten=ac_ten_abs/ac_ten_abs.sum()
        else:
            ac_ten=ac_ten/ac_ten_abs.sum()
        return ac_ten
        
    def _preparations_model(self,CL_model):
        self.attribution_model=copy.deepcopy(CL_model).to(self.DEVICE)
        self.attribution_model.eval()

        if self.Attribution_Method=="GradientShap":
            self.attribution_model=captum.attr.GradientShap(self.attribution_model)
        elif self.Attribution_Method=="Gradients":
            self.attribution_model=captum.attr.Saliency(self.attribution_model)
            
    
    def _get_Feature_Attribution(self,Task_Num):
        ac_Test_Dataset_features=self.Test_Datasets_Features[Task_Num].to(self.DEVICE).requires_grad_()
        ac_Test_Dataset_labels=torch.tensor(self.Test_Datasets_Labels[Task_Num], device=self.DEVICE)
        if self.Attribution_Method=="GradientShap":
            attribution = self.attribution_model.attribute(inputs=ac_Test_Dataset_features, 
                                                           baselines=self.backgrounds,
                                                           target=ac_Test_Dataset_labels,
                                                           n_samples=10,
                                                          )
        elif self.Attribution_Method=="Gradients":
            attribution = self.attribution_model.attribute(inputs=ac_Test_Dataset_features, 
                                                           target=ac_Test_Dataset_labels,
                                                           abs=False
                                                          )
        return attribution.to(self.DEVICE)

    def _calculate_mse(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two 3D tensors.
    
        Args:
            tensor1 (torch.Tensor): The first 3D tensor.
            tensor2 (torch.Tensor): The second 3D tensor.
    
        Returns:
            float: The Mean Squared Error between the two tensors.
        """
        # Ensure both tensors have the same shape
        if tensor1.shape != tensor2.shape:
            raise ValueError("Input tensors must have the same shape.")
        
        # Compute the Mean Squared Error
        tensor1 = self._normalize_tensor(tensor1,abs=False)
        tensor2 = self._normalize_tensor(tensor2,abs=False)
        mse = torch.mean((tensor1 - tensor2) ** 2)
        return mse.item()
        
    def _compute_shapc_with_thresholds(self,attribution_1, attribution_2, keep_percentage):
        #https://openaccess.thecvf.com/content/CVPR2024W/JRDB/papers/Cai_Is_Our_Continual_Learner_Reliable_Investigating_Its_Decision_Attribution_Stability_CVPRW_2024_paper.pdf
        """
        Computes SHAPC between two tensors of SHAP values, determining thresholds
        to retain a given percentage of features, computed separately for each channel,
        and returns the average SHAPC value over all channels.
        
        Parameters:
            attribution_1 (torch.Tensor): Tensor of SHAP values for task t with shape (C, Z, Z).
            attribution_2 (torch.Tensor): Tensor of SHAP values for task τ with shape (C, Z, Z).
            keep_percentage (float): Percentage of features to retain (value between 0 and 1).
        
        Returns:
            float: The average SHAPC value across all channels.
        """
        # Get the number of channels (C), width (Z), and height (Z)
        if len(attribution_1.shape)==2:
            attribution_1=torch.unsqueeze(attribution_1, 0)
            attribution_2=torch.unsqueeze(attribution_2, 0)
        channels, width, height = attribution_1.shape
    
        # Initialize a list to store SHAPC values for each channel
        shapc_values = []
    
        for c in range(channels):
            # Get the SHAP values for the current channel
            attribution_1_channel = attribution_1[c]
            attribution_2_channel = attribution_2[c]
    
            # Flatten tensors for threshold calculation
            flat_1 = attribution_1_channel.flatten()
            flat_2 = attribution_2_channel.flatten()
    
            # Determine the quantile for the given percentage
            threshold_1 = torch.quantile(flat_1, 1 - keep_percentage)
            threshold_2 = torch.quantile(flat_2, 1 - keep_percentage)
    
            # Compute the binary masks p_t(x) and p_τ(x) based on the thresholds
            mask_1 = (attribution_1_channel >= threshold_1).int()
            mask_2 = (attribution_2_channel >= threshold_2).int()
    
            # Compute the intersection and union of the masks
            intersection = (mask_1 & mask_2).float()  # Element-wise AND
            union = (mask_1 | mask_2).float()  # Element-wise OR
    
            # Compute the element-wise differences of SHAP values
            diff = torch.abs(attribution_1_channel - attribution_2_channel)
    
            # Apply the exponential transformation to differences
            exp_diff = torch.exp(-diff)
    
            # Compute the numerator (sum over the intersection)
            numerator = torch.sum(exp_diff * intersection)
    
            # Compute the denominator (sum over the union)
            denominator = torch.sum(exp_diff * union)
    
            # Avoid division by zero
            if denominator != 0:
                shapc_value = numerator / denominator
            else:
                shapc_value = 0.0
            
            # Append the SHAPC value for the current channel
            shapc_values.append(shapc_value)
    
        # Compute the average SHAPC value across all channels
        avg_shapc = torch.mean(torch.tensor(shapc_values))
    
        return avg_shapc.item()

    def _compute_entropy(self, ac_tensor: torch.Tensor, eps = 1e-12):

        flat_tensor = ac_tensor.flatten()
        flat_tensor = self._normalize_tensor(flat_tensor)
        flat_tensor_clamp = flat_tensor.clamp(min=eps)  # Replace 0s with eps for numerical stability
        entropy = -(flat_tensor * torch.log(flat_tensor_clamp)).sum()
        return entropy.item()

    import torch

    def _compute_rectangle_spread_metric(self,saliency_map):
        """
        Compute the total area of the smallest bounding rectangles for each unique saliency value.
    
        Args:
            saliency_map (torch.Tensor): 2D tensor representing the saliency map.
    
        Returns:
            float: Total area of the bounding rectangles.
        """
        # Ensure the saliency map is 2D
        
    
        if len(saliency_map.shape)==3:
            saliency_map=saliency_map[0]
        saliency_map=self._normalize_tensor(saliency_map)
            
        assert len(saliency_map.shape) == 2, "Saliency map must be a 2D tensor"
        
        # Find unique saliency values and their pixel coordinates
        unique_values = torch.sort(saliency_map.flatten()).values.tolist()
        total_area = 0.0
        
        for value in unique_values:
            # Find indices where the saliency map equals the current value
            indices = torch.nonzero(saliency_map >= value, as_tuple=False)
            
            if indices.size(0) > 0:  # Only process if there are pixels with this value
                # Get min and max coordinates for bounding rectangle
                min_y, min_x = torch.min(indices, dim=0).values
                max_y, max_x = torch.max(indices, dim=0).values
                #print(indices)
                #print(min_y, min_x, max_y, max_x)
                
                # Compute the area of the bounding rectangle
                area = (max_y - min_y + 1) * (max_x - min_x + 1)
                total_area += area.item()
    
        return total_area/len(unique_values)


    def _compute_spatial_variance(self, saliency_map):
        """
        Compute the spatial variance of saliency values.
    
        Args:
            saliency_map (torch.Tensor): 2D tensor representing the saliency map.
    
        Returns:
            float: Spatial variance of the saliency values.
        """
        # Ensure the saliency map is 2D
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]
        normalized_saliency = self._normalize_tensor(saliency_map)
        
        assert len(normalized_saliency.shape) == 2, "Saliency map must be a 2D tensor"
    
        # Get the height and width of the saliency map
        height, width = normalized_saliency.shape
    
        # Create coordinate grids
        x_coords = torch.arange(width, dtype=torch.float32, device=normalized_saliency.device)
        y_coords = torch.arange(height, dtype=torch.float32, device=normalized_saliency.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
    
        # Compute weighted mean (centroid)
        mu_x = (normalized_saliency * grid_x).sum()
        mu_y = (normalized_saliency * grid_y).sum()
    
        # Compute weighted variance
        var_x = (normalized_saliency * (grid_x - mu_x) ** 2).sum()
        var_y = (normalized_saliency * (grid_y - mu_y) ** 2).sum()
    
        # Combine variances for total spatial variance
        spatial_variance = var_x + var_y
    
        return spatial_variance.item()
                
    def Task_Feature_Attribution(self,CL_model,Task_Num):
        #print("HP3")
        
        self._preparations_model(CL_model)
            
        self.Feature_Attributions[Task_Num]=self._get_Feature_Attribution(Task_Num)
        #print("HP4")

    def Save_Random_Picture_Salency(self,samples_per_label=1,plt_name="Salencymaps.png"):

        images=[]
        salency_map_before=[]
        salency_map_after=[]
        for Task_Num in range(len(self.Test_Datasets_Features)):
            unique_labels = list(set(self.Test_Datasets_Labels[Task_Num]))
            for ac_label in unique_labels:
                found_samples=0
                while found_samples<samples_per_label:
                    ac_point=random.randint(0, self.Test_Datasets_Features[Task_Num].shape[0]-1)
                    if ac_label==self.Test_Datasets_Labels[Task_Num][ac_point]:
                        images.append(self.Test_Datasets_Features[Task_Num][ac_point].detach().cpu().numpy())
                        salency_map_before.append(self.Feature_Attributions[Task_Num][ac_point].detach().cpu().numpy())
                        salency_map_after.append(self.after_training_attributions[Task_Num][ac_point].detach().cpu().numpy())
                        found_samples+=1
                        #print(found_samples)
        
        n_images = len(images)
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Blue -> Black -> Red
        cmap = LinearSegmentedColormap.from_list("BlueBlackRed", colors)
                
        fig, axs = plt.subplots(3, n_images, figsize=(25, 11))
        for i in range(n_images):
            if images[i].ndim == 2: 
                axs[0, i].imshow(images[i], cmap='gray') 
            elif images[i].shape[0]==1:
                axs[0, i].imshow(images[i][0], cmap='gray') 
            else:  
                axs[0, i].imshow(images[i]) 
            axs[0, i].axis('off')  
        for i in range(n_images):
            norm = TwoSlopeNorm(vmin=np.min(salency_map_after[i]), vcenter=0, vmax=np.max(salency_map_after[i]))
            if len(salency_map_after[i].shape)==3:
                salency_map_after[i]=salency_map_after[i][0]
            im = axs[1, i].imshow(salency_map_after[i], cmap=cmap, norm=norm) 
            axs[1, i].axis('off')
            # Add a colorbar
            fig.colorbar(im, ax=axs[1, i], orientation='vertical', fraction=0.046, pad=0.04)
        for i in range(n_images):
            norm = TwoSlopeNorm(vmin=np.min(salency_map_before[i]), vcenter=0, vmax=np.max(salency_map_before[i]))
            if len(salency_map_before[i].shape)==3:
                salency_map_before[i]=salency_map_before[i][0]
            im = axs[2, i].imshow(salency_map_before[i], cmap=cmap, norm=norm)  
            axs[2, i].axis('off')
            # Add a colorbar
            fig.colorbar(im, ax=axs[2, i], orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(plt_name)
        plt. close()

    def Get_Feature_Change_Score(self,CL_model,threshold=0.5):
        #print("HP5")
        
        for aap,ac_attribution in enumerate(self.Feature_Attributions):
            if ac_attribution is None:
                raise Exception("The following Task was not evaluated beforehand: "+str(aap))

        self._preparations_model(CL_model)

        after_training_attributions=[]
        attributions_differences=[]
        attributions_differences_mean=[]
        attributions_entropy=[]
        attributions_entropy_mean=[]
        attributions_spread=[]
        attributions_spread_mean=[]
        for ac_task_num in range(len(self.Feature_Attributions)):
            after_training_attributions.append(self._get_Feature_Attribution(ac_task_num))
            attributions_differences.append([])
            attributions_entropy.append([])
            attributions_spread.append([])
            for ac_image in range(after_training_attributions[-1].shape[0]):
                """
                attributions_differences[-1].append(
                    self._compute_shapc_with_thresholds(
                        self.Feature_Attributions[ac_task_num][ac_image],
                        after_training_attributions[ac_task_num][ac_image],
                        threshold
                    )
                )
                """
                attributions_differences[-1].append(
                    self._calculate_mse(
                        self.Feature_Attributions[ac_task_num][ac_image],
                        after_training_attributions[ac_task_num][ac_image]
                    )
                )
                attributions_entropy[-1].append(
                    self._compute_entropy(
                        after_training_attributions[ac_task_num][ac_image]
                    )
                )
                """
                attributions_spread[-1].append(
                    self._compute_rectangle_spread_metric(
                        after_training_attributions[ac_task_num][ac_image]
                    )
                )
                """
                attributions_spread[-1].append(
                    self._compute_spatial_variance(
                        after_training_attributions[ac_task_num][ac_image]
                    )
                )
            attributions_differences_mean.append(sum(attributions_differences[-1])/len(attributions_differences[-1]))
            attributions_entropy_mean.append(sum(attributions_entropy[-1])/len(attributions_entropy[-1]))
            attributions_spread_mean.append(sum(attributions_spread[-1])/len(attributions_spread[-1]))
        #print("HP6")
        attributions_differences_means_mean=sum(attributions_differences_mean)/len(attributions_differences_mean)
        attributions_entropy_means_mean=sum(attributions_entropy_mean)/len(attributions_entropy_mean)
        attributions_spread_means_mean=sum(attributions_spread_mean)/len(attributions_spread_mean)
        self.after_training_attributions=after_training_attributions
        return_list=[attributions_differences_means_mean,
                     attributions_differences_mean,
                     attributions_entropy_means_mean,
                     attributions_entropy_mean,
                     attributions_spread_means_mean,
                     attributions_spread_mean]
        return return_list
        
                
