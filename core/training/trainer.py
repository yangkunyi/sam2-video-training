"""
Unified SAM2 Lightning trainer module.
This module provides PyTorch Lightning implementation for SAM2 training.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger
import json
import random
import numpy as np

from core.model.sam2 import SAM2Model
from core.data.dataset import collate_fn
from core.training.loss import SAM2TrainingLoss
from config import Config


# Import SAM2 data utilities for format conversion
try:
    from sam2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    # Define fallback classes if SAM2 is not available
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class BatchedVideoMetaData:
        unique_objects_identifier: torch.LongTensor
        frame_orig_size: torch.LongTensor
    
    @dataclass
    class BatchedVideoDatapoint:
        img_batch: torch.FloatTensor
        obj_to_frame_idx: torch.IntTensor
        masks: torch.BoolTensor
        metadata: BatchedVideoMetaData
        dict_key: str


class PromptGenerator:
    """Generate prompts from ground truth masks for SAM2-style processing."""
    
    def __init__(
        self, 
        prompt_types: List[str] = None, 
        number_of_points: Tuple[int, int] = (1, 3), 
        include_center: bool = False
    ):
        """Initialize prompt generator."""
        self.prompt_types = prompt_types or ["point", "bbox"]
        self.number_of_points = number_of_points
        self.include_center = include_center
    
    def generate_prompts(self, mask: torch.Tensor) -> Dict[str, Any]:
        """Generate random prompts from mask."""
        if mask.sum() == 0:
            return self._empty_prompts()
        
        # Convert to numpy for easier processing
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Select random prompt type
        prompt_type = random.choice(self.prompt_types)
        
        if prompt_type == "point":
            return self._generate_point_prompts(mask_np)
        elif prompt_type == "bbox":
            return self._generate_bbox_prompts(mask_np)
        else:
            return self._empty_prompts()
    
    def _generate_point_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate point prompts."""
        if mask.sum() == 0:
            return self._empty_prompts()
        
        pos_pixels = np.where(mask > 0)
        neg_pixels = np.where(mask == 0)
        
        # Number of points
        num_total = random.randint(self.number_of_points[0], self.number_of_points[1])
        
        # Include center point if requested
        center_included = False
        if self.include_center and len(pos_pixels[0]) > 0:
            center_row = int(np.mean(pos_pixels[0]))
            center_col = int(np.mean(pos_pixels[1]))
            if mask[center_row, center_col] > 0:
                center_included = True
                num_total = max(1, num_total - 1)
        
        num_pos = min(max(1, num_total // 2), len(pos_pixels[0]))
        num_neg = num_total - num_pos
        
        # Sample positive points
        pos_points_list = []
        if center_included:
            pos_points_list.append([center_col, center_row])
            num_pos = max(0, num_pos - 1)
        
        if num_pos > 0 and len(pos_pixels[0]) > 0:
            pos_indices = np.random.choice(len(pos_pixels[0]), min(num_pos, len(pos_pixels[0])), replace=False)
            random_pos_points = np.column_stack([pos_pixels[1][pos_indices], pos_pixels[0][pos_indices]])
            pos_points_list.append(random_pos_points)
        
        pos_points = np.vstack(pos_points_list) if pos_points_list else np.zeros((0, 2))
        
        # Sample negative points
        if num_neg > 0 and len(neg_pixels[0]) > 0:
            neg_indices = np.random.choice(len(neg_pixels[0]), min(num_neg, len(neg_pixels[0])), replace=False)
            neg_points = np.column_stack([neg_pixels[1][neg_indices], neg_pixels[0][neg_indices]])
        else:
            neg_points = np.zeros((0, 2))
        
        # Combine points
        all_points = np.vstack([pos_points, neg_points])
        pos_labels = np.ones(len(pos_points))
        neg_labels = np.zeros(len(neg_points))
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        return {
            "point_coords": torch.from_numpy(all_points).float(),
            "point_labels": torch.from_numpy(all_labels).long(),
        }
    
    def _generate_bbox_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return self._empty_prompts()
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        return {"boxes": bbox}
    
    def _empty_prompts(self) -> Dict[str, Any]:
        """Generate empty prompts."""
        return {
            "point_coords": torch.zeros(0, 2),
            "point_labels": torch.zeros(0),
            "boxes": torch.zeros(4),
            "masks": torch.zeros(512, 512),
        }


class SAM2LightningModule(L.LightningModule):
    """Lightning module for SAM2 video training."""
    
    def __init__(self, config: Config):
        """
        Initialize Lightning module.
        
        Args:
            config: Training configuration
        """
        super().__init__()
        
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self.model = None
        self.criterion = SAM2TrainingLoss(
            bce_weight=config.loss.bce_weight,
            dice_weight=config.loss.dice_weight,
            iou_weight=config.loss.iou_weight,
            temporal_weight=config.loss.temporal_weight,
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.val_step_outputs = []
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(
            prompt_types=config.dataset.prompt_types,
            number_of_points=config.dataset.num_points,
            include_center=config.dataset.include_center_point,
        )
        
    def setup(self, stage: str):
        """Setup model when stage starts."""
        if self.model is None:
            logger.info("Loading SAM2 model...")
            self.model = SAM2Model(
            checkpoint_path=self.config.model.checkpoint_path,
            config_path=self.config.model.config_path,
            trainable_modules=self.config.model.trainable_modules,
            device=str(self.device),
            image_size=self.config.model.image_size,
            num_maskmem=self.config.model.num_maskmem,
            max_objects=self.config.max_objects,
        )
        self.model.load(str(self.device))
        
        # Report model info
        model_info = self.model.get_info()
        logger.info(f"Model loaded: {model_info['total_parameters']:,} total params, "
                   f"{model_info['trainable_parameters']:,} trainable params")
    
    def _convert_batch_to_sam2_format(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert simple batch format to enhanced BatchedVideoDatapoint format with efficient indexing.
        
        Args:
            batch: Simple batch with 'images', 'masks', 'prompts' keys
            
        Returns:
            Enhanced BatchedVideoDatapoint: Data structure with efficient indexing mechanisms
        """
        images = batch["images"]  # [B, T, C, H, W]
        masks = batch["masks"]  # [B, T, N, H, W]
        prompts = batch.get("prompts", [])
        
        batch_size, num_frames, num_channels, height, width = images.shape
        _, _, max_objects, _, _ = masks.shape
        
        # Convert images from [B, T, C, H, W] to [T, B, C, H, W] for frame-sequential processing
        img_batch = images.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]
        
        # Create obj_to_frame_idx mapping [T, O, 2] with proper indexing
        # Maps each of O objects to their (frame_idx, video_idx)
        obj_to_frame_idx = torch.zeros(num_frames, max_objects, 2, dtype=torch.int32)
        
        # Implement the key indexing mechanism for efficient access
        # This allows accessing [T×B] image batches through a flattened [T*B] dimension
        for frame_idx in range(num_frames):
            for obj_idx in range(max_objects):
                obj_to_frame_idx[frame_idx, obj_idx, 0] = frame_idx  # frame index
                obj_to_frame_idx[frame_idx, obj_idx, 1] = 0  # batch index (simplified for single video)
        
        # Create masks in SAM2 format [T, O, H, W]
        # Rearrange masks from [B, T, N, H, W] to [T, O, H, W]
        # For multi-batch training, we handle each batch item separately
        # In this simplified version, we use the first batch item
        b_idx = 0
        sam2_masks = masks[b_idx].permute(1, 0, 2, 3)  # [T, N, H, W] -> [T, O, H, W]
        
        # Create flat indexing mechanism for efficient access
        flat_obj_to_img_idx = torch.arange(num_frames * batch_size).view(num_frames, batch_size)
        
        # Create enhanced BatchedVideoDatapoint structure
        batched_datapoint = {
            "img_batch": img_batch,              # [T, B, C, H, W]
            "masks": sam2_masks,                 # [T, O, H, W]
            "obj_to_frame_idx": obj_to_frame_idx, # [T, O, 2]
            "prompts": prompts,
            "batch_size": batch_size,
            "max_objects": max_objects,
            "num_frames": num_frames,
            "flat_obj_to_img_idx": flat_obj_to_img_idx,  # [T, B]
        }
        
        return batched_datapoint
    
    def _process_prompts_for_sam2(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process prompts from dataset format to SAM2-compatible format.
        
        Args:
            batch: Batch containing prompts for first frame
            
        Returns:
            Dict: Processed prompts compatible with SAM2 model
        """
        prompts = batch.get("prompts", [])
        if not prompts:
            # Generate empty prompts
            return self.prompt_generator._empty_prompts()
        
        # If prompts are provided in dataset format (list of prompt dicts),
        # convert them to SAM2 format
        sam2_prompts = {
            "point_coords": torch.zeros(0, 2),
            "point_labels": torch.zeros(0),
            "boxes": torch.zeros(4),
            "masks": torch.zeros(height, width),  # Will be set below
        }
        
        # Process prompts for each object in first frame
        masks = batch["masks"]  # [B, T, N, H, W]
        if masks.shape[0] > 0:  # Batch size > 0
            first_frame_masks = masks[0, 0]  # First frame, all objects [N, H, W]
            
            # Combine all prompts for multiple objects
            all_point_coords = []
            all_point_labels = []
            all_boxes = []
            all_masks = []
            
            for obj_idx in range(first_frame_masks.shape[0]):
                obj_mask = first_frame_masks[obj_idx]
                obj_prompt = self.prompt_generator.generate_prompts(obj_mask)
                
                # Collect all prompt types
                if "point_coords" in obj_prompt and obj_prompt["point_coords"].shape[0] > 0:
                    all_point_coords.append(obj_prompt["point_coords"])
                    all_point_labels.append(obj_prompt["point_labels"])
                
                if "boxes" in obj_prompt:
                    all_boxes.append(obj_prompt["boxes"])
                
                if "masks" in obj_prompt:
                    all_masks.append(obj_prompt["masks"])
            
            # Combine prompts from all objects
            if all_point_coords:
                sam2_prompts["point_coords"] = torch.cat(all_point_coords, dim=0)
                sam2_prompts["point_labels"] = torch.cat(all_point_labels, dim=0)
            
            if all_boxes:
                sam2_prompts["boxes"] = torch.stack(all_boxes, dim=0)
            
            if all_masks:
                # Combine masks - for simplicity, use a placeholder
                # In practice, SAM2 expects individual masks per object
                sam2_prompts["masks"] = all_masks[0]  # Use first object's mask as placeholder
        
        return sam2_prompts
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        if self.config.optimizer.type.lower() == "adamw":
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
            )
        else:  # Default to Adam
            optimizer = optim.Adam(
                trainable_params,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        
        # Setup scheduler
        scheduler = None
        scheduler_config = {}
        
        if self.config.scheduler.enabled:
            if self.config.scheduler.type == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.config.scheduler.T_max,
                    eta_min=self.config.scheduler.eta_min,
                )
                scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
            elif self.config.scheduler.type == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.config.scheduler.step_size,
                    gamma=self.config.scheduler.gamma,
                )
                scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        
        logger.info(f"Configured optimizer: {self.config.optimizer.type}, "
                   f"LR: {self.config.optimizer.lr}, Trainable params: {len(trainable_params):,}")
        
        if scheduler:
            return {"optimizer": optimizer, **scheduler_config}
        else:
            return {"optimizer": optimizer}
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None,
                prompts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for multiple objects with BatchedVideoDatapoint support."""
        # Try to convert to BatchedVideoDatapoint format for more efficient processing
        batch = {"images": images, "masks": masks, "prompts": prompts}
        try:
            batched_datapoint = self._convert_batch_to_sam2_format(batch)
            return self.model(batched_video_datapoint=batched_datapoint)
        except Exception as e:
            logger.warning(f"Failed to convert to BatchedVideoDatapoint format: {e}")
            # Fallback to simplified format
            return self.model(images=images, masks=masks, prompts=prompts)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for multiple objects with efficient batched processing."""
        images = batch["images"]  # [B, T, C, H, W]
        masks = batch["masks"]  # [B, T, N, H, W] where N is number of objects
        prompts = batch["prompts"]
        
        # Forward pass with BatchedVideoDatapoint for efficient processing
        outputs = self.forward(images, masks, prompts)
        pred_masks = outputs.get("pred_masks_high_res", outputs.get("pred_masks", torch.zeros_like(masks)))
        
        # Ensure mask dimensions match
        if pred_masks.shape != masks.shape:
            if pred_masks.dim() == masks.dim():
                # Handle multiple objects - ensure we don't exceed target dimensions
                pred_masks = pred_masks[:, :, :masks.shape[2], :masks.shape[-2], :masks.shape[-1]]
            else:
                # Add object dimension if missing
                pred_masks = pred_masks.unsqueeze(2)
                if pred_masks.shape[2] > masks.shape[2]:
                    pred_masks = pred_masks[:, :, :masks.shape[2]]
        
        # Compute loss with proper handling of multiple objects
        total_loss, loss_components = self.criterion(
            pred_masks=pred_masks,
            target_masks=masks,
            return_components=True,
        )
        
        # Log metrics
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True)
        for name, value in loss_components.items():
            self.log(f"train/{name}", value, logger=True)
        
        # Log learning rate
        sch = self.lr_schedulers()
        if sch is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/learning_rate", lr, prog_bar=True, logger=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step for multiple objects with efficient batched processing."""
        images = batch["images"]
        masks = batch["masks"]  # [B, T, N, H, W] where N is number of objects
        prompts = batch.get("prompts", None)
        
        # Forward pass with BatchedVideoDatapoint for efficient processing
        outputs = self.forward(images, masks, prompts)
        pred_masks = outputs.get("pred_masks_high_res", outputs.get("pred_masks", torch.zeros_like(masks)))
        
        # Ensure mask dimensions match
        if pred_masks.shape != masks.shape:
            if pred_masks.dim() == masks.dim():
                # Handle multiple objects - ensure we don't exceed target dimensions
                pred_masks = pred_masks[:, :, :masks.shape[2], :masks.shape[-2], :masks.shape[-1]]
            else:
                # Add object dimension if missing
                pred_masks = pred_masks.unsqueeze(2)
                if pred_masks.shape[2] > masks.shape[2]:
                    pred_masks = pred_masks[:, :, :masks.shape[2]]
        
        # Compute loss with proper handling of multiple objects
        total_loss, loss_components = self.criterion(
            pred_masks=pred_masks,
            target_masks=masks,
            return_components=True,
        )
        
        # Store for epoch end
        self.val_step_outputs.append({
            "val_loss": total_loss,
            **{f"val_{k}": v for k, v in loss_components.items()}
        })
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Validation epoch end."""
        if not self.val_step_outputs:
            return
        
        # Aggregate metrics
        metrics = {}
        for key in self.val_step_outputs[0].keys():
            values = [step[key] for step in self.val_step_outputs]
            metrics[key] = torch.stack(values).mean()
        
        # Log metrics
        for name, value in metrics.items():
            self.log(name, value, prog_bar=(name == "val_loss"), logger=True)
        
        # Check for best model
        val_loss = metrics["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_best_model()
        
        logger.info(f"Validation epoch {self.current_epoch + 1}: "
                   f"Loss = {val_loss:.4f}, Best = {self.best_val_loss:.4f}")
        
        # Clear outputs
        self.val_step_outputs.clear()
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        if self.trainer.is_global_zero:
            checkpoint = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config.to_dict(),
            }
            
            checkpoint_path = Path(self.config.output_dir) / "best_model.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best model saved: {checkpoint_path}")
    
    def on_train_end(self):
        """Training end callback."""
        logger.info(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        
        if self.trainer.is_global_zero:
            # Save final metrics
            final_metrics = {
                "best_val_loss": self.best_val_loss.item(),
                "total_epochs": self.current_epoch + 1,
                "config": self.config.to_dict(),
            }
            
            metrics_path = Path(self.config.output_dir) / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
            
            logger.info(f"Training metrics saved to {metrics_path}")


class SAM2LightningDataModule(L.LightningDataModule):
    """Lightning data module for SAM2 training."""
    
    def __init__(self, config: Config):
        """
        Initialize data module.
        
        Args:
            config: Configuration containing dataset settings
        """
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
    
    def setup(self, stage: str):
        """Setup datasets for multiple objects."""
        from core.data.dataset import create_dataloader  # Import here to avoid circular
        
        if stage == "fit":
            # Use same dataset for train/val for now (can be split differently)
            self.train_dataset = create_dataloader(
                dataset_type="video",
                dataset_path=self.config.dataset.data_path,
                batch_size=self.config.dataset.batch_size,
                num_workers=self.config.dataset.num_workers,
                shuffle=True,
                image_size=self.config.dataset.image_size,
                video_clip_length=self.config.dataset.video_clip_length,
                prompt_types=self.config.dataset.prompt_types,
                number_of_points=self.config.dataset.num_points,
                include_center=self.config.dataset.include_center_point,
            )
            
            # Create validation dataset (could be different split)
            self.val_dataset = create_dataloader(
                dataset_type="video", 
                dataset_path=self.config.dataset.data_path,
                batch_size=self.config.dataset.batch_size,
                num_workers=self.config.dataset.num_workers,
                shuffle=False,
                image_size=self.config.dataset.image_size,
                video_clip_length=self.config.dataset.video_clip_length,
                prompt_types=self.config.dataset.prompt_types,
                number_of_points=self.config.dataset.num_points,
                include_center=self.config.dataset.include_center_point,
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_dataset
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self.val_dataset