import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES

@LOSSES.register_module(force=True)
class Class_CELoss(nn.Module):
    def __init__(self, loss_weight=1.0):  
        """  
        Custom CrossEntropyLoss class for one-hot encoded labels.  
        """  
        super(Class_CELoss, self).__init__()  
        self.criterion = nn.CrossEntropyLoss()  
        self.loss_weight = loss_weight

    def forward(self, preds, one_hot_labels):  
        """  
        Compute the CrossEntropyLoss between logits and one-hot labels.  

        Args:  
            preds (torch.Tensor): The predicted logits, shape (B, Nq, 18).  
            one_hot_labels (torch.Tensor): The one-hot encoded target labels, shape (B, Nq, 18).  
        
        Returns:  
            torch.Tensor: The computed cross-entropy loss.  
        """  
        # Convert one-hot labels to class indices (B, Nq)  
        target_indices = torch.argmax(one_hot_labels, dim=-1)  # Shape: (B, Nq)  
        loss = self.criterion(preds.view(-1, preds.size(-1)), target_indices.view(-1))  
        return loss * self.loss_weight 



    
@LOSSES.register_module(force=True)
class DiceLoss(nn.Module):  
    def __init__(self, num_classes=18, epsilon=1e-6, loss_weight=1.0):  
        """  
        num_classes: Nq, number of classes  
        epsilon: Small value to avoid division by zero  
        """  
        super(DiceLoss, self).__init__()  
        self.num_classes = num_classes  
        self.epsilon = epsilon  
        self.loss_weight = loss_weight

    def forward(self, masks_lst_Nq_i, one_hot_gt_j):  
        """  
        masks_lst_Nq: Predicted probabilities, shape (Nq, B * Dx * Dy * Dz)  
        voxel_semantics: Ground truth semantic labels, shape (B * Dx * Dy * Dz,)  
        mask_camera: Mask region, shape (B * Dx * Dy * Dz,), default None  
        use_mask: Whether to use mask for selecting valid region  
        """  

        # Calculate Dice loss  
        # Get predictions and ground truth for the current class  
        pred = masks_lst_Nq_i  # Predicted values for current (valid or all voxels)  
        target = one_hot_gt_j  # Ground truth values for current (valid or all voxels)  

        # Compute Dice coefficient  
        intersection = (pred * target).sum()  # Intersection part  
        union = pred.sum() + target.sum()     # Sum of predictions and targets  
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)  # Dice formula  

        # Compute Dice loss  
        dice_loss = 1 - dice  # Dice loss (1 - Dice)  

        # Return weighted loss  
        return dice_loss * self.loss_weight



@LOSSES.register_module(force=True)  
class FocalLoss(nn.Module):  
    def __init__(self, num_classes=18, alpha=0.65, gamma=3.5, loss_weight=1.0):  
        """  
        num_classes: Nq, number of classes  
        alpha: Weight factor for class balance  
        gamma: Exponent in Focal Loss for hard example mining  
        loss_weight: Loss scaling factor  
        """  
        super(FocalLoss, self).__init__()  
        self.num_classes = num_classes  
        self.alpha = alpha  
        self.gamma = gamma  
        self.loss_weight = loss_weight  

    def forward(self, masks_lst_Nq_i, one_hot_gt_j):  
        """  
        masks_lst_Nq: Predicted probabilities, shape (Nq, B * Dx * Dy * Dz)  
        voxel_semantics: Ground truth semantic labels, shape (B * Dx * Dy * Dz,)  
        mask_camera: Mask region, shape (B * Dx * Dy * Dz,), default None  
        use_mask: Whether to use mask for selecting valid region  
        """  

        pred = masks_lst_Nq_i   
        target = one_hot_gt_j 

        # BCE Loss for the class  
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  
        prob = torch.sigmoid(pred)  
        pt = prob * target + (1 - prob) * (1 - target)  # pt: probability aligned with label  
        
        # Focal Loss scaling  
        focal_loss = self.alpha * (1 - pt).pow(self.gamma) * bce_loss  
        focal_loss = focal_loss.mean()
        
        return focal_loss * self.loss_weight  
    

@LOSSES.register_module(force=True)  
class BinaryFocalLoss(nn.Module):  
    """  
    Binary Focal Loss class, optionally supports mask_camera region.  

    Args:  
    - alpha: Positive/negative sample weight factor, default 0.25  
    - gamma: Focusing parameter, default 2.0  
    - use_mask_camera: Whether to use mask_camera region, default False  
    """  
    def __init__(self, alpha=0.25, gamma=2.0):  
        super(BinaryFocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  

    def forward(self, mask_fo, mapped_mask_fo_voxels, mask_camera=None, use_mask_camera=False):  
        """  
        Forward computation of Binary Focal Loss.  
        
        Args:  
        - mask_fo: (B * Dx * Dy * Dz,) Predicted probabilities, in range [0, 1]  
        - mapped_mask_fo_voxels: (B * Dx * Dy * Dz,) Ground truth labels, values 0 or 1  
        - mask_camera: (B * Dx * Dy * Dz,) Mask region, values are True or False  
        
        Returns:  
        - focal_loss: scalar, mean value of Focal Loss  
        """  
        if use_mask_camera:  
            if mask_camera is None:  
                raise ValueError("mask_camera must be provided when use_mask_camera is True.")  
            # Select regions where mask_camera is True  
            mask_fo = mask_fo[mask_camera.bool()]  
            mapped_mask_fo_voxels = mapped_mask_fo_voxels[mask_camera.bool()]  
        # Compute p_t  
        p_t = mask_fo * mapped_mask_fo_voxels + (1 - mask_fo) * (1 - mapped_mask_fo_voxels)  
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # Add 1e-8 to avoid log(0)  

        return focal_loss.mean()  