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
    from sam2.training.utils.data_utils import (
        BatchedVideoDatapoint,
        BatchedVideoMetaData,
    )

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
        self.best_val_loss = float("inf")
        self.val_step_outputs = []

    def setup(self, stage: str):
        """Setup model when stage starts."""
        if stage == "fit":
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
            logger.info(
                f"Model loaded: {model_info['total_parameters']:,} total params, "
                f"{model_info['trainable_parameters']:,} trainable params"
            )

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

        logger.info(
            f"Configured optimizer: {self.config.optimizer.type}, "
            f"LR: {self.config.optimizer.lr}, Trainable params: {len(trainable_params):,}"
        )

        if scheduler:
            return {"optimizer": optimizer, **scheduler_config}
        else:
            return {"optimizer": optimizer}

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multiple objects with BatchedVideoDatapoint support."""
        # Try to convert to BatchedVideoDatapoint format for more efficient processing
        return self.model(images=images, masks=masks, prompts=prompts)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for multiple objects with efficient batched processing."""
        images = batch["images"]  # [B, T, C, H, W]
        masks = batch["masks"]  # [B, T, N, H, W] where N is number of objects
        prompts = batch["prompts"]

        # Forward pass with BatchedVideoDatapoint for efficient processing
        outputs = self.forward(images, masks, prompts)
        pred_masks = outputs.get(
            "pred_masks_high_res", outputs.get("pred_masks", torch.zeros_like(masks))
        )

        # Ensure mask dimensions match
        if pred_masks.shape != masks.shape:
            if pred_masks.dim() == masks.dim():
                # Handle multiple objects - ensure we don't exceed target dimensions
                pred_masks = pred_masks[
                    :, :, : masks.shape[2], : masks.shape[-2], : masks.shape[-1]
                ]
            else:
                # Add object dimension if missing
                pred_masks = pred_masks.unsqueeze(2)
                if pred_masks.shape[2] > masks.shape[2]:
                    pred_masks = pred_masks[:, :, : masks.shape[2]]

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
        pred_masks = outputs.get(
            "pred_masks_high_res", outputs.get("pred_masks", torch.zeros_like(masks))
        )

        # Ensure mask dimensions match
        if pred_masks.shape != masks.shape:
            if pred_masks.dim() == masks.dim():
                # Handle multiple objects - ensure we don't exceed target dimensions
                pred_masks = pred_masks[
                    :, :, : masks.shape[2], : masks.shape[-2], : masks.shape[-1]
                ]
            else:
                # Add object dimension if missing
                pred_masks = pred_masks.unsqueeze(2)
                if pred_masks.shape[2] > masks.shape[2]:
                    pred_masks = pred_masks[:, :, : masks.shape[2]]

        # Compute loss with proper handling of multiple objects
        total_loss, loss_components = self.criterion(
            pred_masks=pred_masks,
            target_masks=masks,
            return_components=True,
        )

        # Store for epoch end
        self.val_step_outputs.append(
            {
                "val_loss": total_loss,
                **{f"val_{k}": v for k, v in loss_components.items()},
            }
        )

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

        logger.info(
            f"Validation epoch {self.current_epoch + 1}: "
            f"Loss = {val_loss:.4f}, Best = {self.best_val_loss:.4f}"
        )

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
        logger.info(
            f"Training completed! Best validation loss: {self.best_val_loss:.4f}"
        )

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
                dataset_type=self.config.dataset.dataset_type,
                dataset_path=self.config.dataset.data_path,
                batch_size=self.config.dataset.batch_size,
                num_workers=self.config.dataset.num_workers,
                shuffle=self.config.dataset.shuffle,
                image_size=self.config.dataset.image_size,
                video_clip_length=self.config.dataset.video_clip_length,
                prompt_types=self.config.dataset.prompt_types,
                num_of_pos_points=self.config.dataset.num_of_pos_points,
                num_of_neg_points=self.config.dataset.num_of_neg_points,
                include_center=self.config.dataset.include_center_point,
            )

            # Create validation dataset (could be different split)
            self.val_dataset = create_dataloader(
                dataset_type=self.config.valdataset.dataset_type,
                dataset_path=self.config.valdataset.data_path,
                batch_size=self.config.valdataset.batch_size,
                num_workers=self.config.valdataset.num_workers,
                shuffle=self.config.valdataset.shuffle,
                image_size=self.config.valdataset.image_size,
                video_clip_length=self.config.valdataset.video_clip_length,
                prompt_types=self.config.valdataset.prompt_types,
                num_of_pos_points=self.config.valdataset.num_of_pos_points,
                num_of_neg_points=self.config.valdataset.num_of_neg_points,
                include_center=self.config.valdataset.include_center_point,
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_dataset

    def val_dataloader(self):
        """Return validation dataloader."""
        return self.val_dataset
