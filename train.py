"""
Simplified SAM2 video training script.
Clean, direct implementation focused on core training functionality.
"""

import os
import sys
from pathlib import Path
import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
# SwanLab logger is instantiated via Hydra target defined in configs

from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="train_1", version_base=None)
@logger.catch(onerror=lambda _: sys.exit(1))
def main(cfg: DictConfig) -> None:
    """Main entry point with consolidated training logic and Hydra configuration management."""

    OUTPUT_DIR = Path(HydraConfig.get().run.dir)
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
    # Instantiate loggers based on config
    exp_name = (
        f"sam2-video-{cfg.model.trainable_modules}-gt_stride_{cfg.loss.gt_stride}"
    )

    # SwanLab (optional)
    if getattr(cfg.swanlab, "enabled", True):
        logger.info(f"Initializing SwanLab logger for project: {cfg.swanlab.project}")
        swanlab_logger = instantiate(
            cfg.swanlab,
            experiment_name=exp_name,
            description="SAM2 Video Training with Multiple Prompts",
            logdir=os.path.join(str(OUTPUT_DIR), "logs"),
        )
    
    logger.info("SwanLab logger disabled via config.swanlab.enabled=false")

# Weights & Biases (optional)
    logger.info(f"Initializing W&B logger for project: {cfg.wandb.project}")
    wandb_logger = instantiate(
        cfg.wandb,
        name=exp_name,
    )

    
    # =====================================
    # SECTION 2: CONFIGURATION & SETUP
    # =====================================
    # Convert to structured config

    # Set random seed
    L.seed_everything(cfg.seed)

    logger.info("Starting SAM2 training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # =====================================
    # SECTION 3: DIRECTORY & CONFIG SETUP
    # =====================================
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = OUTPUT_DIR / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to {config_path}")

    # =====================================
    # SECTION 4: LIGHTNING COMPONENTS
    # =====================================
    # Instantiate module and data module via Hydra targets, passing unpacked sections
    lightning_module = instantiate(cfg.module, _recursive_=False)
    data_module = instantiate(cfg.data_module, _recursive_=False)

    # =====================================
    # SECTION 5: CALLBACK CONFIGURATION
    # =====================================
    # Instantiate callbacks list defined in config
    callbacks = [instantiate(cb) for cb in cfg.callbacks]

    # =====================================
    # SECTION 6: LOGGER SETUP
    # =====================================

    # =====================================
    # SECTION 7: TRAINER CREATION & EXECUTION
    # =====================================
    # Decide logger argument for Trainer

    trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
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
