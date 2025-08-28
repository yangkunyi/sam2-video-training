"""
Main training script for SAM2 video training.
This is the unified entry point for all SAM2 training functionality.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from dataclasses import asdict

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from icecream import ic

# ic.disabl()

from config import Config
from core.training.trainer import SAM2LightningModule, SAM2LightningDataModule


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )


def create_trainer(config: Config, callbacks: Optional[list] = None) -> L.Trainer:
    """Create Lightning trainer instance."""
    # Create callbacks
    if callbacks is None:
        callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.output_dir) / "checkpoints",
        filename="sam2-epoch{epoch:02d}-val_loss{val/total_loss:.4f}",
        save_top_k=3,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Early stopping (optional)
    if config.trainer.early_stopping_patience:
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            patience=config.trainer.early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Create logger
    pl_logger = None
    if config.use_wandb:
        pl_logger = SwanLabLogger(
            project=config.wandb_project,
            experiment_name=f"sam2-training-{Path(config.output_dir).name}",
            logdir=config.output_dir,
        )

        # Log hyperparameters
        pl_logger.log_hyperparams(asdict(config))
    else:
        pl_logger = TensorBoardLogger(
            save_dir=config.output_dir,
            name="sam2_training",
        )

    # Create trainer
    trainer = L.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        val_check_interval=config.trainer.val_check_interval,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_progress_bar=config.trainer.enable_progress_bar,
        logger=pl_logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        default_root_dir=config.output_dir,
    )

    return trainer


def train(config: Config):
    """Main training function."""
    logger.info("Starting SAM2 training...")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration using Hydra's native method
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config, config_path)
    logger.info(f"Configuration saved to {config_path}")

    # Convert DictConfig to Config dataclass for Lightning components

    lightning_module = SAM2LightningModule(config)
    data_module = SAM2LightningDataModule(config)
    trainer = create_trainer(config)

    # Start training
    logger.info("Starting training...")
    trainer.fit(lightning_module, data_module)

    logger.info("Training completed!")
    logger.info(f"Outputs saved to: {output_dir}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Main entry point using Hydra."""
    # Setup logging
    typed_cfg: Config = OmegaConf.to_object(cfg)
    setup_logging(cfg.log_level)

    # Handle config creation
    if hydra.core.hydra_config.HydraConfig.get().mode == hydra.types.RunMode.MULTIRUN:
        # In multirun mode, we don't want to create sample configs
        pass
    else:
        # Check if we're asked to create a sample config
        # This can be done by running with a special config or flag
        pass

    # Set random seed
    L.seed_everything(cfg.seed)

    # Start training
    train(typed_cfg)


# Keep the original main function for backward compatibility
def main():
    """Main entry point - kept for backward compatibility."""
    # For backward compatibility, we can still support command line args
    # but prefer using Hydra
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="SAM2 Video Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--create-config", action="store_true", help="Create sample config and exit"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Common overrides for quick testing
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--model-checkpoint", type=str, default=None, help="Model checkpoint path"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to training data"
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)

    # Handle config creation
    if args.create_config:
        # Create sample config using Hydra's defaults
        from omegaconf import OmegaConf

        sample_config = OmegaConf.load("configs/config.yaml")
        with open("sample_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(sample_config))
        logger.info("Sample configuration saved to sample_config.yaml")
        logger.info("Please update the paths in sample_config.yaml before training.")
        return

    # For backward compatibility, run with Hydra
    # But we need to pass the command line args to Hydra
    # This is a bit complex, so we'll just call the hydra main directly
    # and let the user use Hydra's command line syntax

    # If specific args are provided, we can handle them
    if any(
        [
            args.config,
            args.seed,
            args.output_dir,
            args.model_checkpoint,
            args.data_path,
            args.max_epochs,
            args.batch_size,
            args.lr,
            args.no_wandb,
        ]
    ):
        logger.warning(
            "Using legacy command line arguments. Consider using Hydra's command line overrides instead."
        )
        logger.warning(
            "Example: python train.py model.checkpoint_path=/path/to/checkpoint"
        )

    # Run with Hydra
    hydra_main()


if __name__ == "__main__":
    hydra_main()
