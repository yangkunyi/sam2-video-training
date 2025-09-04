# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# NEVER MODIFY THE core/sam2train.py UNLESS I ASK YOU TO

## Quick Commands

### Setup & Installation
```bash
# Install dependencies (includes PyTorch Lightning)
pip install -r requirements.txt

# Install SAM2 from official repository
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Verify Lightning installation
python -c "import lightning; print('Lightning available')"

# Verify SAM2 installation  
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 available')"
```

### Development & Training

#### Training Commands
```bash
# Train with Hydra configuration (recommended)
python train.py

# Override specific parameters
python train.py trainer.max_epochs=100 optimizer.lr=1e-4

# Use different dataset
python train.py dataset=coco_val

# Training quickstart
python train.py

# Quick test with debug logging
python train.py trainer.max_epochs=2 trainer.limit_train_batches=5 trainer.limit_val_batches=2
```

#### Advanced Training
```bash
# Multi-GPU training
python train.py trainer.accelerator=gpu trainer.devices=4

# Mixed precision training
python train.py trainer.precision=16-mixed

# Resume from checkpoint
python train.py ckpt_path=/path/to/checkpoint.ckpt

# Validate only (no training)
python train.py trainer.limit_train_batches=0
```

### Configuration & Testing
```bash
# Show configuration structure and overrides
python train.py --help

# Validate specific configuration
python train.py --config-name=config --cfg job trainer.max_epochs=1

# (optional) add tests under test/ and run via pytest
```

### Training Monitoring  
```bash
# Monitor latest training logs (Hydra output structure)
tail -f outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/.hydra/hydra.log

# Check configuration used in last run  
cat outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/.hydra/config.yaml

# Access Lightning checkpoints
ls outputs/$(date +%Y-%m-%d)/$(ls outputs/$(date +%Y-%m-%d) | tail -1)/checkpoints/

# Monitor with TensorBoard
tensorboard --logdir=outputs/

# SwanLab logs are stored in the Hydra run dir under logs/
```

## High-Level Architecture

### Core Components Hierarchy
```
sam2-video-training/
├── train.py               # Main training script (Hydra + Lightning)
├── config.py              # Dataclass configuration system
├── sam2_video/           # Library package
│   ├── config.py         # Dataclass config system
│   ├── model/            # Model
│   │   ├── sam2model.py  # SAM2 model wrapper and tracker
│   │   └── losses.py     # Combined loss functions
│   ├── training/         # Training components
│   │   └── trainer.py    # Lightning trainer module
│   ├── data/             # Data components
│   │   ├── dataset.py    # COCO→video datasets and collate
│   │   └── data_utils.py # BatchedVideoDatapoint structures
│   └── utils/            # Utilities
│       ├── prompts.py    # Prompt utilities (points, boxes)
│       ├── masks.py      # Mask utilities and merging
│       ├── viz.py        # Visualization helpers (GIF)
│       ├── model_utils.py# Parameter counting, trainable controls
│       └── __init__.py   # Aggregator (re-exports)
├── configs/              # Hydra configuration files
│   ├── config.yaml       # Main config with defaults
│   ├── dataset/          # Dataset-specific configs (coco, coco_val)
│   ├── model/            # Model configs
│   ├── trainer/          # Lightning trainer configs
│   └── ...              # Other component configs
├── test/                 # Test modules
├── requirements.txt      # Python dependencies
└── .gitignore           # Consolidated ignore patterns
```

### Training Pipeline Flow
```
Hydra Config Management -> PyTorch Lightning -> SAM2 Training
        ↓                       ↓                    ↓
YAML configs + overrides -> Lightning module -> Multi-object tracking
        ↓                       ↓                    ↓  
Dataclass instantiation -> Automatic optimization -> Memory module training
        ↓                       ↓                    ↓
Dataset + DataLoader -> Training/validation loops -> Checkpoints + metrics
```

### Configuration Architecture
The project uses **Hydra for configuration management** with dataclass configs:

```python
# Root configuration in configs/config.yaml
defaults:
  - dataset: coco             # Dataset configuration
  - model: sam2              # SAM2 model settings  
  - trainer: default         # Lightning trainer
  - loss: default           # Loss function weights
  - optimizer: adamw        # Optimizer settings
  - scheduler: cosine       # Learning rate scheduling

# Dataclass structure
Config:
  model: ModelConfig           # SAM2 model loading & trainable modules
  dataset: DatasetConfig       # Data paths, prompt generation, video clips
  trainer: TrainerConfig       # Lightning trainer (epochs, devices, precision)
  loss: LossConfig           # BCE, Dice, IoU loss weights  
  optimizer: OptimizerConfig  # Learning rate, weight decay
  scheduler: SchedulerConfig  # Cosine annealing, warmup
```

### Key Training Parameters
- **trainable_modules**: `model.trainable_modules` (e.g., ["memory_attention", "memory_encoder"])
- **video_clip_length**: `dataset.video_clip_length` (frames per sequence, default 5)  
- **image_size**: `dataset.image_size` (input resolution, default 512x512)
- **batch_size**: `dataset.batch_size` (start with 1 for memory constraints)
- **learning_rate**: `optimizer.lr` (start with 1e-4 for memory modules)
- **prompt_generation**: `model.num_pos_points`, `model.include_center` (point sampling)

### Multi-Object Tracking Features
- **Prompt Generation**: Configurable point/bbox/mask prompts from ground truth
- **COCO Integration**: Native COCO dataset support with RLE mask decoding
- **Flexible Sampling**: Point-based prompts with optional center point inclusion
- **Video Sequences**: Handle variable-length video clips with multiple objects per frame

### Configuration Pattern
Uses **Hydra's automatic configuration management**:
```python
# Hydra decorator handles all configuration loading
@hydra.main(version_base=None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Convert to structured config
    config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, cfg)
    
    # Override from command line automatically handled
    # python train.py optimizer.lr=1e-4 trainer.max_epochs=100
```

### Component Pattern
All components follow a clean modular design:
```python
# sam2_video/model/sam2model.py: SAM2Model wrapper with Lightning integration
# sam2_video/data/dataset.py: COCO->video datasets and collate  
# sam2_video/training/trainer.py: SAM2LightningModule with training/validation logic
# sam2_video/model/losses.py: SAM2 training losses (BCE, Dice, IoU)

# Key classes:
class SAM2Model:          # Wraps SAM2 predictor for training
# prompt generation is provided as functions in core/prompts.py
class VideoDataset:       # Loads video frames and annotations
class COCODataset:        # COCO format with RLE mask support
class SAM2LightningModule: # Lightning training module
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
- **Hydra output management**: Automatic timestamped directories in `outputs/`
- **loguru**: Structured logging with file output and console formatting
- **SwanLab logging**: via SwanLabLogger in `train.py`
- **Lightning logging**: Built-in metric logging
- **Output structure**: `outputs/YYYY-MM-DD/HH-MM-SS/` with checkpoints, logs, config
- **Key metrics**: Combined loss, BCE loss, Dice loss, IoU loss, learning rate, validation metrics
- **Configuration preservation**: Full config saved as `.hydra/config.yaml` in each run

### Development Workflow
1. **Modify configs**: Edit YAML files in `configs/` directory
2. **Override parameters**: Use command-line overrides: `python train.py optimizer.lr=1e-4`
3. **Monitor training**: SwanLab or log files in Hydra output directory
4. **Resume training**: Use `ckpt_path=path/to/checkpoint.ckpt` parameter

## Architecture Notes

### Design Principles
- **Hydra Configuration**: Centralized YAML-based config with command-line overrides
- **PyTorch Lightning**: Modern training framework with automatic optimization
- **Modular Components**: Clean separation between model, data, training, and loss
- **Simple package structure**: Minimal `__init__.py` where needed
- **Multi-Object Focus**: Built for video object tracking with multiple targets

### Key Implementation Details
- **Memory Module Training**: Focus on SAM2's memory attention and encoder components
- **Prompt-Based Learning**: Ground truth masks converted to point/bbox/mask prompts
- **Video Sequence Handling**: Configurable clip lengths
- **COCO Integration**: Native support for COCO dataset format with RLE decoding
- **Loss Combination**: Weighted BCE, Dice, IoU for training

### Testing & Validation
- **Integration Testing**: Lightning trainer validation loops for end-to-end sanity
- **Configuration Validation**: Hydra's config validation and override system
