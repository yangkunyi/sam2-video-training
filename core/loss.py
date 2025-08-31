"""
Simplified loss functions for SAM2 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from loguru import logger


class SAM2TrainingLoss(nn.Module):
    """Combined loss function for SAM2 memory module training."""
    
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, 
                 iou_weight: float = 0.5, temporal_weight: float = 0.1,
                 smooth: float = 1e-6):
        """
        Initialize loss function.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss  
            iou_weight: Weight for IoU loss
            temporal_weight: Weight for temporal consistency loss
            smooth: Smoothing factor for Dice/IoU calculations
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.temporal_weight = temporal_weight
        self.smooth = smooth
        
        logger.info(f"Loss function initialized - BCE: {bce_weight}, Dice: {dice_weight}, "
                   f"IoU: {iou_weight}, Temporal: {temporal_weight}")
    
    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor,
                valid_masks: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Any:
        """
        Compute combined loss for multiple objects.
        
        Args:
            pred_masks: Predicted masks [B, T, N, H, W] where N is number of objects
            target_masks: Target masks [B, T, N, H, W] 
            valid_masks: Optional valid mask indicators [B, T, N]
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss tensor or (total_loss, components) dict
        """
        # Ensure input shapes match
        assert pred_masks.shape == target_masks.shape, \
            f"Shape mismatch: pred {pred_masks.shape} vs target {target_masks.shape}"
        
        B, T, N, H, W = pred_masks.shape
        
        # Flatten for loss computation
        pred_flat = pred_masks.view(B * T * N, H, W)
        target_flat = target_masks.view(B * T * N, H, W)
        
        if valid_masks is not None:
            valid_flat = valid_masks.view(B * T * N)
        else:
            valid_flat = torch.ones(B * T * N, device=pred_masks.device)
        
        # Compute individual losses
        bce_loss = self._bce_loss(pred_flat, target_flat, valid_flat)
        dice_loss = self._dice_loss(pred_flat, target_flat, valid_flat)
        iou_loss = self._iou_loss(pred_flat, target_flat, valid_flat)
        
        # Temporal consistency loss (if multiple frames)
        temporal_loss = self._temporal_loss(pred_flat, target_flat, valid_flat, T)
        
        # Combine losses
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss + 
                     self.iou_weight * iou_loss +
                     self.temporal_weight * temporal_loss)
        
        if return_components:
            components = {
                "bce": bce_loss.item(),
                "dice": dice_loss.item(),
                "iou": iou_loss.item(),
                "temporal": temporal_loss.item(),
            }
            return total_loss, components
        else:
            return total_loss
    
    def _bce_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                  valid: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss."""
        # BCE with logits expects same device
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply valid masks
        loss = loss.mean(dim=[1, 2])  # Average over spatial dimensions
        loss = (loss * valid).sum() / (valid.sum() + self.smooth)
        
        return loss
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        pred_probs = torch.sigmoid(pred)
        
        # Flatten and compute Dice
        pred_flat = pred_probs.flatten(1)
        target_flat = target.flatten(1)
        
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        
        # Apply valid masks
        loss = (loss * valid).sum() / (valid.sum() + self.smooth)
        
        return loss
    
    def _iou_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss."""
        pred_probs = torch.sigmoid(pred)
        
        # Flatten and compute IoU
        pred_flat = pred_probs.flatten(1)
        target_flat = target.flatten(1)
        
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        
        # Apply valid masks
        loss = (loss * valid).sum() / (valid.sum() + self.smooth)
        
        return loss
    
    def _temporal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if num_frames <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Reshape back to [B, T, N, H, W] for temporal computation
        B, T, N = pred.shape[0] // (T * N), num_frames, pred.shape[0] // (T * N) // num_frames // B
        H, W = pred.shape[1], pred.shape[2]
        
        pred_reshaped = pred.view(B, T, N, H, W)
        target_reshaped = target.view(B, T, N, H, W)
        valid_reshaped = valid.view(B, T, N)
        
        # Compute frame-to-frame differences
        pred_diff = torch.abs(pred_reshaped[:, 1:] - pred_reshaped[:, :-1])
        target_diff = torch.abs(target_reshaped[:, 1:] - target_reshaped[:, :-1])
        
        # Temporal consistency loss
        temporal_loss = F.mse_loss(pred_diff, target_diff, reduction='none')
        temporal_loss = temporal_loss.mean(dim=[2, 3, 4])  # Average over spatial and object dims
        
        # Apply valid masks (only for frames that are valid)
        valid_pairs = valid_reshaped[:, 1:] * valid_reshaped[:, :-1]
        temporal_loss = (temporal_loss * valid_pairs).sum() / (valid_pairs.sum() + self.smooth)
        
        return temporal_loss