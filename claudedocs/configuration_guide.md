# SAM2 Video Training - Configuration Guide

## Configuration System Overview

The SAM2 Video Training project uses a simplified dataclass configuration system with Hydra integration for flexible and organized parameter management. The configuration is organized into logical components that mirror the project's architecture.

## Configuration Components

### Root Configuration (Config)

The root `Config` class contains all settings for the training process:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | ModelConfig | ModelConfig() | SAM2 model configuration |
| dataset | DatasetConfig | DatasetConfig() | Training dataset configuration |
| valdataset | ValDatasetConfig | ValDatasetConfig() | Validation dataset configuration |
| loss | LossConfig | LossConfig() | Loss function configuration |
| optimizer | OptimizerConfig | OptimizerConfig() | Optimizer configuration |
| scheduler | SchedulerConfig | SchedulerConfig() | Learning rate scheduler configuration |
| trainer | TrainerConfig | TrainerConfig() | PyTorch Lightning trainer configuration |
| max_objects | int | 10 | Maximum number of objects to track simultaneously |
| seed | int | 42 | Random seed for reproducibility |
| use_wandb | bool | True | Enable Wandb logging |
| wandb_project | str | "sam2-training" | Wandb project name |
| output_dir | str | "./outputs" | Output directory for logs and checkpoints |
| log_level | str | "INFO" | Logging level |
| save_dir | str | "./checkpoints" | Checkpoint save directory |

### Model Configuration (ModelConfig)

Configuration for the SAM2 model loading and tracking parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| checkpoint_path | str | "" | Path to SAM2 checkpoint file |
| config_path | str | "configs/sam2.1_hiera_t.yaml" | Path to SAM2 configuration file |
| trainable_modules | List[str] | ["memory_attention", "memory_encoder"] | List of modules to train |
| device | str | "cuda" | Device for model loading |
| num_maskmem | int | 7 | Number of mask memories |
| image_size | int | 512 | Image size for processing |
| backbone_stride | int | 16 | Backbone stride |
| use_obj_ptrs_in_encoder | bool | True | Use object pointers in encoder |
| max_obj_ptrs_in_encoder | int | 16 | Maximum object pointers in encoder |
| add_tpos_enc_to_obj_ptrs | bool | True | Add positional encoding to object pointers |
| proj_tpos_enc_in_obj_ptrs | bool | True | Project temporal position encoding |
| use_signed_tpos_enc_to_obj_ptrs | bool | True | Use signed temporal position encoding |
| multimask_output_in_sam | bool | True | Multi-mask output in SAM |
| multimask_output_for_tracking | bool | True | Multi-mask output for tracking |
| max_objects | int | 10 | Maximum objects for tracking |

### Dataset Configuration (DatasetConfig)

Configuration for training dataset loading:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_type | str | "coco" | Dataset type ("video" or "coco") |
| data_path | str | "/path/to/data.json" | Path to dataset |
| image_size | Tuple[int, int] | (512, 512) | Target image size |
| video_clip_length | int | 5 | Number of frames per video clip |
| batch_size | int | 1 | Training batch size |
| num_workers | int | 16 | Number of data loading workers |
| shuffle | bool | True | Shuffle dataset |
| prompt_types | List[str] | ['point'] | Types of prompts to generate |
| num_of_pos_points | int | 1 | Number of positive points |
| num_of_neg_points | int | 0 | Number of negative points |
| include_center_point | bool | True | Include center point in prompts |

### Validation Dataset Configuration (ValDatasetConfig)

Configuration for validation dataset loading (same parameters as DatasetConfig):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_type | str | "coco" | Dataset type ("video" or "coco") |
| data_path | str | "/path/to/test.json" | Path to validation dataset |
| image_size | Tuple[int, int] | (512, 512) | Target image size |
| video_clip_length | int | 5 | Number of frames per video clip |
| batch_size | int | 1 | Validation batch size |
| num_workers | int | 16 | Number of data loading workers |
| shuffle | bool | False | Shuffle validation dataset |
| prompt_types | List[str] | ['point'] | Types of prompts to generate |
| num_of_pos_points | int | 1 | Number of positive points |
| num_of_neg_points | int | 0 | Number of negative points |
| include_center_point | bool | True | Include center point in prompts |

### Loss Configuration (LossConfig)

Configuration for loss function weights and parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| bce_weight | float | 1.0 | Weight for BCE loss |
| dice_weight | float | 1.0 | Weight for Dice loss |
| iou_weight | float | 0.5 | Weight for IoU loss |
| temporal_weight | float | 0.1 | Weight for temporal consistency loss |
| smooth | float | 1e-6 | Smoothing factor for calculations |

### Optimizer Configuration (OptimizerConfig)

Configuration for optimizer settings:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | str | "AdamW" | Optimizer type |
| lr | float | 1e-4 | Learning rate |
| weight_decay | float | 1e-4 | Weight decay |
| betas | Tuple[float, float] | (0.9, 0.999) | Adam optimizer betas |
| eps | float | 1e-8 | Adam optimizer epsilon |

### Scheduler Configuration (SchedulerConfig)

Configuration for learning rate scheduler:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | True | Enable learning rate scheduler |
| type | str | "CosineAnnealingLR" | Scheduler type |
| T_max | int | 10 | Cosine annealing T_max |
| eta_min | float | 1e-6 | Cosine annealing eta_min |
| step_size | int | 5 | StepLR step size |
| gamma | float | 0.1 | StepLR gamma |

### Trainer Configuration (TrainerConfig)

Configuration for PyTorch Lightning trainer:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| accelerator | str | "auto" | Training accelerator |
| devices | Any | "auto" | Number of devices |
| precision | str | "16-mixed" | Training precision |
| max_epochs | int | 50 | Maximum training epochs |
| gradient_clip_val | float | 1.0 | Gradient clipping value |
| accumulate_grad_batches | int | 1 | Gradient accumulation steps |
| limit_train_batches | float | 1.0 | Limit training batches |
| limit_val_batches | float | 1.0 | Limit validation batches |
| val_check_interval | float | 1.0 | Validation check interval |
| num_sanity_val_steps | int | 2 | Sanity validation steps |
| enable_checkpointing | bool | True | Enable checkpointing |
| enable_progress_bar | bool | True | Enable progress bar |
| logger | bool | True | Enable logging |
| log_every_n_steps | int | 50 | Log every n steps |
| early_stopping_patience | Optional[int] | None | Early stopping patience |

## Configuration Files

### Main Configuration (`configs/config.yaml`)

The main configuration file uses Hydra's defaults system to organize component configurations:

```yaml
defaults:
  - model: sam2
  - dataset: coco
  - valdataset: coco
  - loss: default
  - optimizer: adamw
  - scheduler: cosine
  - trainer: default
  - _self_

# Global settings
max_objects: 10
seed: 42
use_wandb: true
wandb_project: sam2-training
output_dir: ./outputs
log_level: INFO
save_dir: ./checkpoints

hydra:
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

model:
  checkpoint_path: /path/to/checkpoint.pt
  config_path: sam2/sam2.1_hiera_t.yaml
```

### Component Configurations

Each component has its own configuration file in the respective directory:

1. `configs/model/sam2.yaml` - Model configuration
2. `configs/dataset/coco.yaml` - Training dataset configuration
3. `configs/valdataset/coco.yaml` - Validation dataset configuration
4. `configs/loss/default.yaml` - Loss configuration
5. `configs/optimizer/adamw.yaml` - Optimizer configuration
6. `configs/scheduler/cosine.yaml` - Scheduler configuration
7. `configs/trainer/default.yaml` - Trainer configuration

## Configuration Usage

### Command Line Overrides

Hydra allows for flexible command line overrides:

```bash
# Override learning rate
python train.py optimizer.lr=1e-5

# Override multiple parameters
python train.py optimizer.lr=1e-5 trainer.max_epochs=100

# Override dataset path
python train.py dataset.data_path=/new/path/to/data.json

# Override with new configuration file
python train.py --config-name new_config
```

### Programmatic Configuration

Configuration can also be modified programmatically:

```python
from config import Config
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("configs/config.yaml")

# Modify parameters
cfg.optimizer.lr = 1e-5
cfg.trainer.max_epochs = 100

# Convert to dataclass
config = OmegaConf.to_object(cfg)
```

## Training Pipeline Configuration

### Key Training Parameters

1. **Trainable Modules**: Configure via `model.trainable_modules` 
   - Options: `["memory_attention", "memory_encoder"]` (default)
   - Can include other SAM2 components as needed

2. **Video Clip Length**: Configure via `dataset.video_clip_length`
   - Default: 5 frames
   - Affects memory usage and temporal context

3. **Image Size**: Configure via `dataset.image_size`
   - Default: (512, 512)
   - Affects memory usage and processing speed

4. **Batch Size**: Configure via `dataset.batch_size`
   - Default: 1 (recommended for memory training)
   - Increase based on available GPU memory

5. **Learning Rate**: Configure via `optimizer.lr`
   - Default: 1e-4 (recommended for memory training)
   - Adjust based on training stability

### Configuration Pattern

The configuration system uses straightforward dataclass instantiation:

```python
# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
    config = Config()

# Update config from file
for key, value in config_dict.items():
    if hasattr(config, key):
        setattr(config, key, value)
```

## Best Practices

### Configuration Management

1. **Version Control**: Keep configuration files in version control
2. **Documentation**: Document parameter choices and their rationale
3. **Validation**: Validate configuration parameters before training
4. **Reproducibility**: Save configuration with training outputs

### Parameter Tuning

1. **Start Conservative**: Begin with smaller learning rates and batch sizes
2. **Monitor Progress**: Use logging to track training metrics
3. **Early Stopping**: Configure early stopping to prevent overfitting
4. **Learning Rate Scheduling**: Use schedulers to adapt learning rate during training

### Multi-Environment Configuration

1. **Environment-Specific Settings**: Use different configs for development, testing, and production
2. **Path Management**: Use relative paths or environment variables for data paths
3. **Resource Allocation**: Adjust batch size and precision based on available hardware