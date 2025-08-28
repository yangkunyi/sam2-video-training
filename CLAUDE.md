# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

### Setup & Installation
```bash
# Install dependencies (includes PyTorch Lightning)
pip install -r requirements.txt

# Verify Lightning installation
python -c "import lightning; print('Lightning available')"
```

### Development & Training

#### Training Commands
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

#### Advanced Training
```bash
# Multi-GPU training
python train.py --config config.yaml --accelerator gpu --devices 4

# Mixed precision training
python train.py --config config.yaml --precision 16-mixed
```

### Configuration Validation
```bash
# Validate configuration structure
python -c "from config import Config; print('Config schema valid')"

# Test available command line options
python train.py --help
```

### Training Monitoring
```bash
# Monitor latest training logs
tail -f outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/training.log

# Check configuration used in last run
cat outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/config.yaml

# Access training artifacts
ls outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/checkpoints/
```

## High-Level Architecture

### Core Components Hierarchy
```
sam2-video-training/
├── train.py               # Main training script (Lightning-based)
├── config.py              # Simplified configuration system
├── core/                  # Core modules
│   ├── model/
│   │   └── sam2.py       # SAM2 model and tracker
│   ├── data/
│   │   └── dataset.py    # Dataset implementations  
│   └── training/
│       ├── trainer.py    # Lightning trainer
│       └── loss.py       # Loss functions
├── requirements.txt       # Python dependencies
└── .gitignore           # Consolidated ignore patterns
```

### Training Pipeline Flow
```
Config Classes -> Lightning Components -> Training Execution
       ↓                ↓                  ↓
Dataclass configs -> Instantiation -> Model training
                          ↓                  ↓
                    Lightning trainer    -> Model updates & checkpoints
```

### Configuration Architecture
The project uses a **simplified dataclass configuration system**:

```python
# Root configuration with all settings
Config:
  model: ModelConfig           # SAM2 model loading
  dataset: DatasetConfig       # Data loading and prompts
  trainer: TrainerConfig       # Lightning trainer settings
  loss: LossConfig           # Loss computation
  optimizer: OptimizerConfig  # Training optimization
  scheduler: SchedulerConfig  # Learning rate scheduling
```

### Key Training Parameters
- **trainable_modules**: Configure via model.trainable_modules (list of SAM2 components)
- **video clip length**: Configure via dataset.video_clip_length (default 5 frames)
- **image size**: Configure via dataset.image_size (default 512x512)
- **batch size**: Configure via dataset.batch_size (start with 1 for memory training)
- **learning rate**: Configure via optimizer.lr (start with 1e-4 for memory training)

### Configuration Pattern
Uses straightforward dataclass instantiation:
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

### Component Pattern
All components follow a clean modular design:
```python
# Model: SAM2Model wrapper with Lightning integration
# Dataset: VideoDataset and COCODataset classes
# Trainer: SAM2LightningModule for training logic
# Loss: SAM2TrainingLoss with combined loss functions
```

### Lightning Benefits
1. **Simplified Training**: Automatic training loops, checkpointing, and logging
2. **Distributed Training**: Built-in multi-GPU and multi-node support
3. **Mixed Precision**: Automatic mixed precision training for better performance
4. **Advanced Callbacks**: Learning rate monitoring, early stopping, model checkpointing
5. **Gradient Accumulation**: Support for larger effective batch sizes
6. **Validation Integration**: Automatic validation with configurable frequency
7. **Model Summary**: Automatic model parameter and architecture summary

### Logging & Monitoring
- Uses **loguru** for structured logging with file output
- **wandb** integration via global use_wandb flag
- **Lightning logging**: Automatic tensorboard and wandb integration
- Outputs organized by timestamp: outputs/YYYY-MM-DD/HH-MM-SS/
- Key metrics: loss, loss_bce, loss_dice, loss_iou, learning_rate
- Configuration snapshots saved as config.yaml

## Project Cleanup Summary

This project has been cleaned up to remove unnecessary code and files while preserving core functionality:

### Removed Elements
- **Duplicate ignore files**: Consolidated .iflowignore and .rgignore into .gitignore
- **Empty init files**: Removed all __init__.py files from core modules
- **Unused helper module**: Removed core/utils/ and helpers.py (functions not used)
- **Outdated references**: Removed Hydra legacy commands from documentation
- **Redundant code**: Streamlined train.py argument parsing and configuration

### Preserved Elements
- **Core training functionality**: All SAM2 model training capabilities maintained
- **Lightning integration**: PyTorch Lightning workflow fully preserved
- **Configuration system**: Flexible dataclass configuration kept
- **Documentation**: Current architecture and commands documented
- **Import structure**: All core imports work without init files

### Current Project Stats
- **Total Files**: 12 files (33% reduction from original)
- **Core Modules**: 4 functional modules (model, data, training)
- **Documentation**: Updated to reflect current state
- **Dependencies**: All necessary packages maintained