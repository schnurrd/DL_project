import captum
import math
import random
import copy
from pytorch_utils import get_features, get_labels
import torch

#[Denis] added class:
class Feature_Importance_Evaluations:
    def __init__(self,Test_Datasets, DEVICE, Attribution_Method="GradientShap",background_samples=100):
        
        #print("HP1")

        self.supported_attribution_methods=["GradientShap"]
        if Attribution_Method not in self.supported_attribution_methods:
            raise Exception("The following Attribution Method is not supported: "+Attribution_Method)
            
        self.Attribution_Method=Attribution_Method
        self.DEVICE=DEVICE
        self.Test_Datasets_Features=[]
        self.Test_Datasets_Labels=[]
        self.backgrounds=[]

        
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
            
    def _preparations_model(self,CL_model):
        self.attribution_model=copy.deepcopy(CL_model).to(self.DEVICE)
        self.attribution_model.eval()

        if self.Attribution_Method=="GradientShap":
            self.attribution_model=captum.attr.GradientShap(self.attribution_model)
            
    
    def _get_Feature_Attribution(self,Task_Num):
        ac_Test_Dataset_features=self.Test_Datasets_Features[Task_Num].to(self.DEVICE).requires_grad_()
        ac_Test_Dataset_labels=torch.tensor(self.Test_Datasets_Labels[Task_Num], device=self.DEVICE)

        attribution = self.attribution_model.attribute(inputs=ac_Test_Dataset_features, 
                                                       baselines=self.backgrounds,
                                                       target=ac_Test_Dataset_labels,
                                                       n_samples=10,
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
        
    def Task_Feature_Attribution(self,CL_model,Task_Num):
        #print("HP3")
        
        self._preparations_model(CL_model)
            
        self.Feature_Attributions[Task_Num]=self._get_Feature_Attribution(Task_Num)
        #print("HP4")


    def Get_Feature_Change_Score(self,CL_model,threshold=0.5):
        #print("HP5")
        
        for aap,ac_attribution in enumerate(self.Feature_Attributions):
            if ac_attribution is None:
                raise Exception("The following Task was not evaluated beforehand: "+str(aap))

        self._preparations_model(CL_model)

        after_training_attributions=[]
        attributions_differences=[]
        attributions_differences_mean=[]
        for ac_task_num in range(len(self.Feature_Attributions)):
            after_training_attributions.append(self._get_Feature_Attribution(ac_task_num))
            attributions_differences.append([])
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
            attributions_differences_mean.append(sum(attributions_differences[-1])/len(attributions_differences[-1]))
        #print("HP6")
        return sum(attributions_differences_mean)/len(attributions_differences_mean),attributions_differences_mean
        
                
