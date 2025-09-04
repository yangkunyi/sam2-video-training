"""
Simplified configuration system for SAM2 video training.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# Hydra imports
from omegaconf import DictConfig


@dataclass
class ModelConfig:
    """Configuration for SAM2 model."""

    checkpoint_path: str = ""
    config_path: str = "configs/sam2.1_hiera_t.yaml"
    trainable_modules: List[str] = field(
        default_factory=lambda: ["memory_attention", "memory_encoder"]
    )
    device: str = "cuda"
    # Minimal training parameters
    image_size: int = 512
    prompt_type: str = "point"  # "point", "box", "mask"
    forward_backbone_per_frame_for_eval: bool = False

    # Prompt generation parameters (moved from DataConfig)
    num_pos_points: int = 1  # For point prompt generation
    num_neg_points: int = 0  # For point prompt generation
    include_center: bool = True  # Whether to include center point


@dataclass
class SwanLabConfig:
    """Configuration for SwanLab logging."""

    project: str = "sam2-video-training"


@dataclass
class DataConfig:
    """Configuration for dataset."""

    # Paths to COCO annotation files
    train_path: str = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/cholecseg8k/coco_style/merged_gt_coco_annotations_train.json"
    val_path: str = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/cholecseg8k/coco_style/merged_gt_coco_annotations_test.json"
    image_size: int = 512
    video_clip_length: int = 8
    stride: int = 4

    num_workers: int = 4
    batch_size: int = 1  # Limited by GPU memory
    num_categories: Optional[int] = 13  # Number of categories to allocate masks for


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Type of loss to use: "multi_step" (default) or "bce"
    type: str = "multi_step"
    # Subsample ground-truth masks along time when computing loss.
    # If gt_stride=k, only frames [0, k, 2k, ...] contribute to the loss.
    gt_stride: int = 1

    # Weights for the original multi-step loss (kept for backward compatibility)
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    iou_weight: float = 0.5

    # BCE-only options (used when type == "bce")
    bce_pos_weight: Optional[list] = None  # Optional per-class positive weights
    bce_reduction: str = "mean"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    enabled: bool = True
    type: str = "CosineAnnealingLR"
    T_max: int = 10
    eta_min: float = 1e-6
    step_size: int = 5
    gamma: float = 0.1


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning trainer."""

    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str = "16-mixed"
    max_epochs: int = 50
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    val_check_interval: float = 1.0
    num_sanity_val_steps: int = 2
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    logger: bool = True
    log_every_n_steps: int = 10

    # Early stopping (optional)
    early_stopping_patience: Optional[int] = None


@dataclass
class VisualizationConfig:
    """Configuration for GIF visualization and SwanLab logging."""

    enabled: bool = True
    train_every_n_steps: int = 400  # 0 disables train GIFs
    val_first_batch_every_n_epochs: int = 1  # log only first batch per epoch
    max_length: int = 4
    stride: int = 1
    caption: str = "cholecseg8k"


@dataclass
class Config:
    """Root configuration containing all settings."""

    model: ModelConfig
    data: DataConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    swanlab: SwanLabConfig
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Global settings
    seed: int = 42
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation for logging/checkpointing."""
        from dataclasses import asdict

        return asdict(self)


# Note: ConfigStore registration removed to favor Hydra `_target_` instantiation.
# These dataclasses are retained only as documentation/types and are not registered.
