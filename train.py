"""
Simplified SAM2 video training script.
Clean, direct implementation focused on core training functionality.
"""

import os
import sys
from pathlib import Path

import hydra
import hydra.core.global_hydra as ghd
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from swanlab.integration.pytorch_lightning import SwanLabLogger

from core.config import Config
from core.trainer import SAM2LightningDataModule, SAM2LightningModule

from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="config", version_base=None)
@logger.catch(onerror=lambda _: sys.exit(1))
def main(cfg: DictConfig) -> None:
    """Main entry point with consolidated training logic and Hydra configuration management."""

    OUTPUT_DIR = Path(HydraConfig.get().run.dir)
    config: Config = OmegaConf.to_object(cfg)
    # =====================================
    # SECTION 1: LOGGING SETUP
    # =====================================
    log_level = cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
    )
    logger.add(OUTPUT_DIR / "training.log", rotation="10 MB", retention="10 days")
    logger.info(f"Initializing SwanLab logger for project: {config.swanlab.project}")
    swanlab_logger = SwanLabLogger(
        project=config.swanlab.project,
        experiment_name=f"sam2-video-{config.model.prompt_type}",
        description="SAM2 Video Training with Multiple Prompts",
        config=cfg,
        logdir=os.path.join(str(OUTPUT_DIR), "logs"),
    )
    # =====================================
    # SECTION 2: CONFIGURATION & SETUP
    # =====================================
    # Convert to structured config

    # Set random seed
    L.seed_everything(config.seed)

    logger.info("Starting SAM2 training...")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    # =====================================
    # SECTION 3: DIRECTORY & CONFIG SETUP
    # =====================================
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = OUTPUT_DIR / "config.yaml"
    OmegaConf.save(config, config_path)
    logger.info(f"Configuration saved to {config_path}")

    # =====================================
    # SECTION 4: LIGHTNING COMPONENTS
    # =====================================
    lightning_module = SAM2LightningModule(config)
    data_module = SAM2LightningDataModule(config)

    # =====================================
    # SECTION 5: CALLBACK CONFIGURATION
    # =====================================
    callbacks = [
        ModelCheckpoint(
            dirpath=OUTPUT_DIR / "checkpoints",
            filename="sam2-epoch{epoch:02d}-val_loss{val/total_loss:.4f}",
            save_top_k=3,
            monitor="val/total_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optional early stopping
    if config.trainer.early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val/total_loss",
                patience=config.trainer.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    # =====================================
    # SECTION 6: LOGGER SETUP
    # =====================================

    # =====================================
    # SECTION 7: TRAINER CREATION & EXECUTION
    # =====================================
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
        logger=swanlab_logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        default_root_dir=OUTPUT_DIR,
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit(lightning_module, data_module)

    logger.info("Training completed!")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # pass
    main()
