"""
Simplified configuration system for SAM2 video training.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# Hydra imports
from hydra.core.config_store import ConfigStore
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

    # Tracker parameters
    num_maskmem: int = 7
    image_size: int = 512
    backbone_stride: int = 16
    use_obj_ptrs_in_encoder: bool = True
    max_obj_ptrs_in_encoder: int = 16
    add_tpos_enc_to_obj_ptrs: bool = True
    proj_tpos_enc_in_obj_ptrs: bool = True
    use_signed_tpos_enc_to_obj_ptrs: bool = True
    multimask_output_in_sam: bool = True
    multimask_output_for_tracking: bool = True
    max_objects: int = 10  # Maximum number of objects to track simultaneously
    
    # SAM2Train specific parameters
    prob_to_use_pt_input_for_train: float = 0.0
    prob_to_use_pt_input_for_eval: float = 0.0
    prob_to_use_box_input_for_train: float = 0.0
    prob_to_use_box_input_for_eval: float = 0.0
    
    # Frame correction parameters
    num_frames_to_correct_for_train: int = 1
    num_frames_to_correct_for_eval: int = 1
    rand_frames_to_correct_for_train: bool = False
    rand_frames_to_correct_for_eval: bool = False
    
    # Initial conditioning frames
    num_init_cond_frames_for_train: int = 1
    num_init_cond_frames_for_eval: int = 1
    rand_init_cond_frames_for_train: bool = True
    rand_init_cond_frames_for_eval: bool = False
    
    # Point sampling parameters
    add_all_frames_to_correct_as_cond: bool = False
    num_correction_pt_per_frame: int = 7
    pt_sampling_for_eval: str = "center"
    pt_sampling_for_train: str = "uniform"
    prob_to_sample_from_gt_for_train: float = 0.0
    
    # Additional training parameters
    use_act_ckpt_iterative_pt_sampling: bool = False
    forward_backbone_per_frame_for_eval: bool = False  # Maximum number of objects to track simultaneously


@dataclass
class DatasetConfig:
    """Configuration for dataset."""

    dataset_type: str = "coco"
    data_path: str = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/cholecseg8k/coco_style/merged_gt_coco_annotations_train.json"
    image_size: Tuple[int, int] = (512, 512)
    video_clip_length: int = 5
    batch_size: int = 1
    num_workers: int = 16
    shuffle: bool = True

    # Prompt configuration
    prompt_types: List[str] = field(default_factory=lambda: ['point'])
    num_of_pos_points: int = 1
    num_of_neg_points: int = 0
    include_center_point: bool = True


@dataclass
class ValDatasetConfig:
    """Configuration for validation dataset."""

    dataset_type: str = "coco"
    data_path: str = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/cholecseg8k/coco_style/merged_gt_coco_annotations_test.json"
    image_size: Tuple[int, int] = (512, 512)
    video_clip_length: int = 5
    batch_size: int = 1
    num_workers: int = 16
    shuffle: bool = False

    # Prompt configuration
    prompt_types: List[str] = field(default_factory=lambda: ['point'])
    num_of_pos_points: int = 1
    num_of_neg_points: int = 0
    include_center_point: bool = True


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    bce_weight: float = 1.0
    dice_weight: float = 1.0
    iou_weight: float = 0.5
    temporal_weight: float = 0.1
    smooth: float = 1e-6


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
    log_every_n_steps: int = 50

    # Early stopping (optional)
    early_stopping_patience: Optional[int] = None


@dataclass
class Config:
    """Root configuration containing all settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    valdataset: ValDatasetConfig = field(default_factory=ValDatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Multi-object tracking parameters
    max_objects: int = 10  # Maximum number of objects to track simultaneously

    # Global settings
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "sam2-training"
    output_dir: str = "./outputs"

    # Logging
    log_level: str = "INFO"
    save_dir: str = "./checkpoints"


# Register configs with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="sam2", node=ModelConfig)
cs.store(group="dataset", name="coco", node=DatasetConfig)
cs.store(group="valdataset", name="coco", node=ValDatasetConfig)
cs.store(group="loss", name="default", node=LossConfig)
cs.store(group="optimizer", name="adamw", node=OptimizerConfig)
cs.store(group="scheduler", name="cosine", node=SchedulerConfig)
cs.store(group="trainer", name="default", node=TrainerConfig)

