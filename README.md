# SAM2 Video Training

A simplified, clean implementation for training SAM2 memory modules on video data.

## Features

- **Unified Training**: Single entry point for all training workflows using PyTorch Lightning
- **Simplified Configuration**: Clean YAML-based configuration system
- **Multi-Object Support**: Handle multiple objects in video sequences
- **Flexible Datasets**: Support for both video directory and COCO format datasets
- **Modern Training**: Mixed precision, distributed training, automatic checkpointing
- **Comprehensive Logging**: Integration with Weights & Biases and TensorBoard

## Quick Start

### 1. Create a Configuration

Generate a sample configuration file:

```bash
python train.py --create-config
```

Update the paths in `sample_config.yaml`:

```yaml
model:
  checkpoint_path: "/path/to/sam2_hiera_tiny.pt"
  config_path: "/path/to/sam2_config.yaml"
  trainable_modules: ["memory_attention", "memory_encoder"]

dataset:
  data_path: "/path/to/video_data"
  image_size: [512, 512]
  video_clip_length: 5
  batch_size: 1

trainer:
  max_epochs: 50
  accelerator: "auto"
  precision: "16-mixed"

use_wandb: true
wandb_project: "sam2-training"
```

### 2. Start Training

```bash
# Basic training
python train.py --config sample_config.yaml

# Override parameters from command line
python train.py --config sample_config.yaml --lr 1e-4 --max-epochs 100

# Without wandb
python train.py --config sample_config.yaml --no-wandb

# Quick test with debug logging
python train.py --config sample_config.yaml --debug --max-epochs 2
```

## Directory Structure

```
sam2-video-training/
├── train.py                 # Main training script (single entry point)
├── config.py               # Simplified configuration system
├── core/                   # Core modules
│   ├── model/
│   │   └── sam2.py        # Unified SAM2 model and tracker
│   ├── data/
│   │   └── dataset.py     # Simplified dataset implementations
│   ├── training/
│   │   ├── trainer.py     # Lightning trainer module
│   │   └── loss.py        # Loss functions
│   └── utils/
│       └── helpers.py     # Utility functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Configuration

The configuration system uses simple dataclasses with YAML support. Key sections:

### Model Configuration
- `checkpoint_path`: Path to SAM2 checkpoint
- `config_path`: Path to SAM2 configuration file  
- `trainable_modules`: List of modules to train (e.g., ["memory_attention", "memory_encoder"])

### Dataset Configuration  
- `data_path`: Path to video data directory
- `image_size`: Target image size [height, width]
- `video_clip_length`: Number of frames per video clip
- `batch_size`: Training batch size

### Training Configuration
- `max_epochs`: Maximum training epochs
- `accelerator`: Training accelerator (auto, cpu, gpu)
- `precision`: Training precision (16-mixed, 32-true)
- `lr`: Learning rate

## Data Format

### Video Dataset Structure
Organize your video data as:
```
video_data/
├── video_001/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── video_002/
│   ├── frame_001.jpg
│   └── ...
```

### COCO Dataset Format
Provide a COCO JSON file with video annotations including video_id and order_in_video fields.

## Training Features

- **PyTorch Lightning**: Modern training framework with automatic optimization
- **Mixed Precision**: 16-bit training for better performance
- **Automatic Checkpointing**: Save best models and restore training
- **Multi-GPU**: Built-in distributed training support
- **Early Stopping**: Optional early stopping based on validation loss
- **Learning Rate Monitoring**: Track learning rate changes

## Loss Functions

The training uses a combined loss function with configurable weights:
- BCE Loss: Binary cross-entropy for mask prediction
- Dice Loss: Spatial overlap for segmentation
- IoU Loss: Intersection over Union loss  
- Temporal Loss: Consistency between consecutive frames

## Examples

### Basic Training
```bash
python train.py \
  --model-checkpoint /models/sam2_hiera_tiny.pt \
  --model-config /models/sam2_config.yaml \
  --data-path /data/videos \
  --lr 1e-4 \
  --max-epochs 50
```

### Debug Mode
```bash
python train.py \
  --config config.yaml \
  --debug \
  --no-wandb \
  --max-epochs 2
```

### Multi-GPU Training
```bash
python train.py \
  --config config.yaml \
  --accelerator gpu \
  --devices 4
```

## Output

Training outputs are saved to:
- `checkpoints/`: Model checkpoints (best_model.pth, epoch checkpoints)
- `config.yaml`: Training configuration used
- `training_metrics.json`: Final training metrics and statistics

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- PyTorch Lightning
- torch
- loguru
- PIL
- Transformers (for some components)
- Weights & Biases (optional)

## Migration from Old Structure

The old complex structure has been simplified:

- **Before**: Multiple entry points (`my_app.py`, `lightning_train.py`)
- **After**: Single `train.py` with clean command line interface

- **Before**: Complex configuration with multiple enums and inheritance
- **After**: Simple dataclass configuration

- **Before**: Duplicated training code in multiple files  
- **After**: Unified Lightning-based implementation

- **Before**: Over-engineered model loading and tracking separation
- **After**: Single unified `SAM2Model` class

The refactored code is approximately **70% smaller** and significantly easier to maintain and extend.