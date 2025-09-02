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

from core.sam2model import SAM2Model
from core.dataset import sam2_collate_fn
from core.loss_fns import MultiStepMultiMasksAndIous, CORE_LOSS_KEY
from core.config import Config


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


from torch.utils.data import DataLoader


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
        # Build weight_dict from config
        weight_dict = {
            "loss_mask": config.loss.bce_weight,
            "loss_dice": config.loss.dice_weight,
            "loss_iou": config.loss.iou_weight,
            "loss_class": 0.0,
        }
        self.criterion = MultiStepMultiMasksAndIous(weight_dict=weight_dict)

        # Training state
        self.best_val_loss = float("inf")
        self.val_step_outputs = []

    def setup(self, stage: str):
        """Setup model when stage starts."""
        if stage == "fit":
            if self.model is None:
                logger.info("Loading SAM2 model...")
                self.model = SAM2Model(
                    model_config=self.config.model,
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

    def forward(self, batch: BatchedVideoDatapoint):
        """Forward pass returning per-frame outputs (list of dicts)."""
        return self.model(batch)

    def training_step(
        self, batch: BatchedVideoDatapoint, batch_idx: int
    ) -> torch.Tensor:
        """Training step with BatchedVideoDatapoint input."""
        # Forward pass - batch is already in BatchedVideoDatapoint format
        outs_per_frame = self.forward(batch)

        # Ground truth masks [T, N, H, W]
        target_masks = batch.masks

        # Compute multi-step loss across frames
        losses = self.criterion(outs_per_frame, target_masks)
        total_loss = losses[CORE_LOSS_KEY]

        # Log metrics
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True)
        self.log("train/loss_mask", losses["loss_mask"], logger=True)
        self.log("train/loss_dice", losses["loss_dice"], logger=True)
        self.log("train/loss_iou", losses["loss_iou"], logger=True)

        # Log learning rate
        sch = self.lr_schedulers()
        if sch is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/learning_rate", lr, prog_bar=True, logger=True)

        return total_loss

    def validation_step(
        self, batch: BatchedVideoDatapoint, batch_idx: int
    ) -> torch.Tensor:
        """Validation step with BatchedVideoDatapoint input."""
        # Forward pass - batch is already in BatchedVideoDatapoint format
        outs_per_frame = self.forward(batch)

        # Ground truth masks [T, N, H, W]
        target_masks = batch.masks

        # Compute multi-step loss across frames
        losses = self.criterion(outs_per_frame, target_masks)
        total_loss = losses[CORE_LOSS_KEY]
        

        # Store for epoch end
        self.val_step_outputs.append({f"val/{k}": v for k, v in losses.items()}),


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
            self.log(name, value, prog_bar=(name == "val/total_loss"), logger=True)

        # Check for best model
        val_loss = metrics["val/total_loss"]
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
        """Setup datasets for training and validation."""
        from core.dataset import COCODataset  # Import here to avoid circular

        if stage == "fit":
            # Training dataset
            self.train_dataset = COCODataset(
                config=self.config.data,
                coco_json_path=self.config.data.train_path,
            )

            # Validation dataset
            self.val_dataset = COCODataset(
                config=self.config.data,
                coco_json_path=self.config.data.val_path,
            )

    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=sam2_collate_fn,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=sam2_collate_fn,
        )
