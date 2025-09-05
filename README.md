# SAM2 Video Training

A simplified, clean implementation for training SAM2 memory modules on video data.

## Features

- Unified training via PyTorch Lightning
- Clean Hydra with `_target_` instantiation
- Multi-object, per-category masks from COCO
- Mixed precision, checkpoints, LR monitor
- Weights & Biases logging (videos, metrics)

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
- `configs/`: Single flat config (`config.yaml`) plus SAM2 arch YAMLs under `configs/sam2/`
- `sam2_video/`: Library package
  - `config.py`: Types only (no ConfigStore)
  - `model/sam2model.py`: SAM2 wrapper and forward
  - `model/losses.py`: Losses
  - `training/trainer.py`: Lightning modules
  - `data/dataset.py`: COCO→video clips + collate
  - `data/data_utils.py`: BatchedVideoDatapoint structures
  - `utils/`: prompts.py, masks.py, viz.py, model_utils.py, `__init__.py` aggregator

## Configuration Notes

- All configuration now lives in `configs/config.yaml` for simplicity.
- `data.*`: set `train_path`, `val_path`, `image_size`, `video_clip_length`, `stride`, `batch_size`
- `model.*`: Hydra target for `sam2_video.model.sam2model.SAM2Model`; provide `checkpoint_path`, `config_path`; set prompt via `model.prompt_type` in {point, box, mask}
- `module`: Hydra target for LightningModule (`sam2_video.training.trainer.SAM2LightningModule`)
- `data_module`: Hydra target for LightningDataModule (`sam2_video.training.trainer.SAM2LightningDataModule`)
- `trainer`: Hydra target for `lightning.pytorch.trainer.trainer.Trainer`
- `callbacks`: list of Hydra targets (ModelCheckpoint, LearningRateMonitor)
- `visualization.*`: enable/disable GIF logging; defaults are conservative

### Learning Rate Schedule

- Scheduler is fixed to cosine decay with linear warmup using `transformers.get_cosine_schedule_with_warmup`.
- Step-based only; `num_training_steps` is derived from Lightning's `trainer.estimated_stepping_batches`.
- Config keys under `scheduler`:
  - `enabled`: turn scheduling on/off (default: true)
  - `warmup_steps`: linear warmup steps before cosine decay (default: 500)
  - `num_cycles`: cosine cycles over training (default: 0.5)

Examples:

```
python train.py scheduler.warmup_steps=1000
python train.py trainer.max_epochs=20 data.batch_size=2
```

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

The training uses a combined loss with configurable weights:
- BCE Loss: Binary cross-entropy for mask prediction
- Dice Loss: Spatial overlap for segmentation
- IoU Loss: Intersection over Union loss

Temporal subsampling for supervision is supported via `loss.gt_stride`.

- Set `loss.gt_stride=k` to compute the loss only on frames `[0, k, 2k, ...]` within each clip.
- Predictions and ground-truth are aligned on those frames only; forward still runs on all frames.
- Example: `loss.gt_stride=4` uses the 1st, 5th, 9th frames for loss if present.

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
- PyTorch Lightning, torch, torchvision, torchmetrics
- hydra-core, omegaconf
- numpy, Pillow, pycocotools, imageio
- loguru, wandb

## Migration Notes

The configs folder was simplified to reduce subdirectories:

- **Before**: Many Hydra groups under `configs/{model,data,trainer,optimizer,scheduler,module,data_module,callbacks,loss,visualization}`
- **After**: Single `configs/config.yaml` containing all sections. SAM2 arch YAML remains at `configs/sam2/sam2.1_hiera_t.yaml`.

Common overrides remain the same; you can now switch prompts via:

```
python train.py model.prompt_type=mask
python train.py model.prompt_type=box
```

## Fail-Fast Behavior

- Errors in core methods are decorated with `@logger.catch(onerror=lambda _: sys.exit(1))` and will stop the run with a clear message.
- Dataset must include non-empty `categories` and valid `file_name` for each image.
- Prompt generation requires non-empty masks; otherwise it raises.
- Batch size is enforced to 1 in the collate for simpler tracking assumptions.

The refactored code is approximately **70% smaller** and significantly easier to maintain and extend.
