# SAM2 Video Training

A simplified, clean implementation for training SAM2 memory modules on video data.

## Features

- Unified training via PyTorch Lightning
- Clean Hydra + dataclass configs
- Multi-object, per-category masks from COCO
- Mixed precision, checkpoints, LR monitor
- SwanLab visualizations (optional)

## Quick Start

### 1. Install dependencies

python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/segment-anything-2.git

### 2. Train

python train.py

Common overrides:

- Paths: `python train.py model.checkpoint_path=/path/sam2.pt model.config_path=/path/sam2.yaml`
- Data: `python train.py data.train_path=/data/train.json data.val_path=/data/val.json`
- Hardware: `python train.py trainer.accelerator=gpu trainer.devices=1 trainer.precision=16-mixed`

## Project Structure

- `train.py`: Single entry (Hydra + Lightning)
- `configs/`: Hydra configs grouped by domain
- `core/`: Training stack
  - `sam2model.py`: SAM2 wrapper and forward
  - `trainer.py`: Lightning modules
  - `dataset.py`: COCO→video clips + collate
  - `loss_fns.py`: Losses
  - `config.py`: Dataclass configs
  - `utils.py`: Prompts, visualization, helpers

## Configuration Notes

- `data.*`: set `train_path`, `val_path`, `image_size`, `video_clip_length`, `stride`, `batch_size`
- `model.*`: provide `checkpoint_path`, `config_path`; set prompt via `prompt_type` in {point, box, mask}
- `visualization.*`: enable/disable GIF logging; defaults are conservative

## Data Format

COCO-style JSON with `images`, `annotations`, `categories` and additional fields:

- `images[*].video_id`: integer grouping frames
- `images[*].order_in_video`: sorting within a video
 - Fail-fast required keys per image: `id`, `file_name`, `video_id`, `order_in_video`

`annotations[*].segmentation` must be RLE for efficiency. Category ids are remapped to contiguous indices.

## Training Features

- Lightning with mixed precision
- Checkpointing and LR monitor
- Optional early stopping

## Loss Functions

The training uses a combined loss function with configurable weights:
- BCE Loss: Binary cross-entropy for mask prediction
- Dice Loss: Spatial overlap for segmentation
- IoU Loss: Intersection over Union loss  
- Temporal Loss: Consistency between consecutive frames

## Examples

python train.py data.train_path=/data/train.json data.val_path=/data/val.json \
  model.checkpoint_path=/models/sam2.pt model.config_path=/models/sam2.yaml \
  trainer.accelerator=gpu trainer.devices=1

## Output

- `outputs/<date>/<time>/checkpoints`: checkpoint files
- `outputs/<date>/<time>/config.yaml`: resolved run config
- `outputs/<date>/<time>/training.log`: run logs

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

## Fail-Fast Behavior

- Errors in core methods are decorated with `@logger.catch(onerror=lambda _: sys.exit(1))` and will stop the run with a clear message.
- Dataset must include non-empty `categories` and valid `file_name` for each image.
- Prompt generation requires non-empty masks; otherwise it raises.
- Batch size is enforced to 1 in the collate for simpler tracking assumptions.

Validate your dataset before training:

python scripts/dataset_sanity_check.py /path/to/train.json --check-files


The refactored code is approximately **70% smaller** and significantly easier to maintain and extend.
