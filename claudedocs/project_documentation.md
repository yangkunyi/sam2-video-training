# SAM2 Video Training - Project Documentation

## Overview

This project provides a simplified, clean implementation for training SAM2 memory modules on video data with multi-object tracking capabilities. It uses PyTorch Lightning for streamlined training workflows and a simplified YAML-based configuration system.

## Architecture Overview

### Core Components Hierarchy
```
sam2-video-training/
├── train.py               # Main training script (Lightning-based)
├── config.py              # Simplified configuration system
├── core/                  # Core modules
│   ├── model/
│   │   └── sam2.py       # Unified SAM2 model and tracker
│   ├── data/
│   │   └── dataset.py    # Dataset implementations  
│   └── training/
│   │   ├── trainer.py    # Lightning trainer
│   │   └── loss.py       # Loss functions
├── configs/              # Configuration files
│   ├── config.yaml       # Main configuration
│   └── ...               # Component-specific configs
├── requirements.txt      # Python dependencies
└── README.md            # Project overview
```

### Training Pipeline Flow
```
Configuration → Lightning Components → Training Execution
     ↓                ↓                  ↓
Dataclass configs → Instantiation → Model training
                          ↓                  ↓
                    Lightning trainer   → Model updates & checkpoints
```

## Main Components

### 1. Training Script (`train.py`)

The main entry point for all SAM2 training functionality that supports both Hydra and traditional argument parsing.

**Key Features:**
- Unified entry point for all training workflows using PyTorch Lightning
- Dual configuration support (Hydra and traditional argparse)
- Automatic checkpointing with ModelCheckpoint callback
- Learning rate monitoring with LearningRateMonitor
- Optional early stopping support
- Wandb/TensorBoard logging integration

**Command Line Usage:**
```bash
# Generate sample configuration
python train.py --create-config

# Train with configuration file
python train.py --config config.yaml

# Override parameters from command line
python train.py --config config.yaml --lr 1e-4 --max-epochs 100

# Training without wandb
python train.py --config config.yaml --no-wandb

# Quick test with debug logging
python train.py --config config.yaml --debug --max-epochs 2
```

**Advanced Training:**
```bash
# Multi-GPU training
python train.py --config config.yaml --accelerator gpu --devices 4

# Mixed precision training
python train.py --config config.yaml --precision 16-mixed
```

### 2. Configuration System (`config.py`)

A simplified dataclass configuration system that organizes training settings into logical groups.

**Configuration Components:**
- `Config`: Root configuration containing all settings
- `ModelConfig`: SAM2 model loading and tracking parameters
- `DatasetConfig`: Training dataset configuration
- `ValDatasetConfig`: Validation dataset configuration
- `LossConfig`: Loss function weights and parameters
- `OptimizerConfig`: Optimizer settings
- `SchedulerConfig`: Learning rate scheduler configuration
- `TrainerConfig`: PyTorch Lightning trainer settings

**Key Parameters:**
- **trainable_modules**: Configure via model.trainable_modules (list of SAM2 components)
- **video clip length**: Configure via dataset.video_clip_length (default 5 frames)
- **image size**: Configure via dataset.image_size (default 512x512)
- **batch size**: Configure via dataset.batch_size (start with 1 for memory training)
- **learning rate**: Configure via optimizer.lr (start with 1e-4 for memory training)

### 3. Model (`core/model/sam2.py`)

A unified SAM2 model class that handles both loading and tracking functionality in a single module.

**Key Classes:**
- `SAM2Model`: Unified SAM2 model for loading and video tracking
- `create_sam2_model`: Factory function for model creation

**Key Features:**
- Selective module training configuration
- Multi-object tracking with frame-sequential, object-parallel approach
- Memory-aware processing for video sequences
- Parameter counting and training configuration reporting
- Support for various prompt types (points, bounding boxes, masks)

### 4. Dataset (`core/data/dataset.py`)

Dataset implementations for video training with multi-object support.

**Key Classes:**
- `VideoDataset`: Basic video dataset with synthetic mask generation
- `COCODataset`: COCO format dataset for video training with ground truth masks
- `PromptGenerator`: Generates random prompts from ground truth masks
- `create_dataloader`: Factory function for dataloader creation

**Supported Dataset Formats:**
- Video Directory Structure: Organized video frame directories
- COCO JSON Format: Annotations with video_id and order_in_video fields

### 5. Training (`core/training/trainer.py`)

PyTorch Lightning implementation for SAM2 training.

**Key Classes:**
- `SAM2LightningModule`: Lightning module for SAM2 video training
- `SAM2LightningDataModule`: Lightning data module for SAM2 training

**Key Features:**
- Automatic optimizer and scheduler configuration
- Training and validation steps with comprehensive logging
- Best model checkpointing based on validation loss
- Integrated loss computation with multiple loss components

### 6. Loss Functions (`core/training/loss.py`)

Combined loss functions for SAM2 memory module training.

**Key Classes:**
- `SAM2TrainingLoss`: Combined loss function with configurable weights

**Loss Components:**
- BCE Loss (Binary Cross-Entropy): For pixel-wise mask prediction accuracy
- Dice Loss: For spatial overlap between predicted and ground truth masks  
- IoU Loss (Intersection over Union): For region-based segmentation accuracy
- Temporal Loss: For consistency between consecutive frames

## Data Flow

1. **Configuration Loading**: Config is loaded from YAML files or command line arguments
2. **Model Initialization**: SAM2 model is loaded with specified checkpoint and configuration
3. **Dataset Setup**: Training and validation datasets are created based on configuration
4. **Training Loop**: Lightning trainer orchestrates the training process:
   - Data loading with custom collate function
   - Forward pass through SAM2 model
   - Loss computation with multiple components
   - Backpropagation and parameter updates
   - Validation and checkpointing
5. **Output Generation**: Model checkpoints, logs, and metrics are saved to output directory

## Key Features

### Multi-Object Support
- Handle multiple objects in video sequences simultaneously
- Object-parallel processing within frames
- Frame-sequential processing across time steps
- Support for variable number of objects per video

### Flexible Datasets
- Support for both video directory and COCO format datasets
- Automatic prompt generation from ground truth masks
- Configurable prompt types (points, bounding boxes, masks)
- Synthetic mask generation for basic video datasets

### Modern Training Infrastructure
- PyTorch Lightning for simplified training workflows
- Mixed precision training for better performance
- Distributed training support for multi-GPU setups
- Automatic checkpointing with best model saving
- Comprehensive logging with Wandb/TensorBoard integration

### Selective Training
- Configure specific modules for training (memory_attention, memory_encoder, etc.)
- Parameter counting and training percentage reporting
- Frozen parameter management for efficient fine-tuning

## Configuration Architecture

The project uses a **simplified dataclass configuration system** with Hydra integration:

```python
# Root configuration with all settings
Config:
  model: ModelConfig           # SAM2 model loading
  dataset: DatasetConfig       # Data loading and prompts
  valdataset: ValDatasetConfig # Validation data loading and prompts
  loss: LossConfig           # Loss computation
  optimizer: OptimizerConfig  # Training optimization
  scheduler: SchedulerConfig  # Learning rate scheduling
  trainer: TrainerConfig       # Lightning trainer settings
```

## Logging & Monitoring

- Uses **loguru** for structured logging with file output
- **Wandb/SwanLab** integration via global use_wandb flag
- **Lightning logging**: Automatic tensorboard and wandb integration
- Outputs organized by timestamp: outputs/YYYY-MM-DD/HH-MM-SS/
- Key metrics: loss, loss_bce, loss_dice, loss_iou, learning_rate
- Configuration snapshots saved as config.yaml

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- PyTorch Lightning
- torch
- loguru
- PIL
- pycocotools
- Hydra
- Transformers (for some components)
- Weights & Biases/SwanLab (optional)