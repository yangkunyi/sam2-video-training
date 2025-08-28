"""
Main training script for SAM2 video training.
This is the unified entry point for all SAM2 training functionality.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

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
        pl_logger = WandbLogger(
            project=config.wandb_project,
            name=f"sam2-training-{Path(config.output_dir).name}",
            save_dir=config.output_dir,
        )
        
        # Log hyperparameters
        pl_logger.log_hyperparams(config.to_dict())
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
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    import yaml
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_path}")
    
    # Create Lightning components
    lightning_module = SAM2LightningModule(config)
    data_module = SAM2LightningDataModule(config)
    trainer = create_trainer(config)
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(lightning_module, data_module)
    
    logger.info("Training completed!")
    logger.info(f"Outputs saved to: {output_dir}")



def main():
    """Main entry point."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="SAM2 Video Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create sample config and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Common overrides for quick testing
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--data-path", type=str, default=None, help="Path to training data")
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
        config = Config()
        import yaml
        sample_path = Path("sample_config.yaml")
        with open(sample_path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        logger.info(f"Sample configuration saved to {sample_path}")
        logger.info("Please update the paths in sample_config.yaml before training.")
        return
    
    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = Config()
        
        # Update config from file
        for key, value in config_dict.items():
            if hasattr(config, key):
                if isinstance(getattr(config, key), dict):
                    # Handle nested configs
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
    else:
        config = Config()
    
    # Apply command line overrides
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.model_checkpoint is not None:
        config.model.checkpoint_path = args.model_checkpoint
    if args.data_path is not None:
        config.dataset.data_path = args.data_path
    if args.max_epochs is not None:
        config.trainer.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.no_wandb:
        config.use_wandb = False
    
    # Set random seed
    L.seed_everything(config.seed)
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()