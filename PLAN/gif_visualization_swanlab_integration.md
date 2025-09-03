# GIF Visualization and SwanLab Integration Plan

## Overview
Integrate existing visualization utilities with SwanLab video logging to create and log training/validation GIFs during SAM2 video training.

## Requirements
- **Goal**: Generate short GIFs from model predictions during train/val and log to SwanLab as videos
- **Constraints**: Keep overhead low (log infrequently), rank-0 only, batch_size=1 assumed by current dataset/visualization utilities, avoid extra dependencies/abstractions
- **Success Criteria**: GIFs render correctly and log to SwanLab without blocking training

## Architecture Design

### Core Principles (KISS, YAGNI, DRY)
- **KISS**: No new abstractions beyond tiny helpers in LightningModule; reuse existing utils
- **YAGNI**: No custom loggers or dataset hooks; minimal config knobs for frequency/length control
- **DRY**: Single `_log_gif` used by both training and validation; centralized scheduling logic

### Existing Assets to Leverage
1. `create_composite_visualization()` - creates single frame visualization with 2x2 layout
2. `create_visualization_gif()` - creates GIF from multiple frames and returns file path
3. SAM2LightningModule - existing Lightning trainer with train/validation steps
4. Hydra configuration system - for managing visualization settings

## Implementation Plan

### 1. Configuration Extension

#### File: `core/config.py`
**Add new dataclass for visualization settings:**
```python
@dataclass
class VisualizationConfig:
    enabled: bool = True
    train_every_n_steps: int = 0  # 0 disables train GIFs
    val_first_batch_every_n_epochs: int = 1  # log only first batch per epoch
    max_length: int = 4
    stride: int = 1
    caption: str = "这是一个测试视频"
```

**Update root Config class:**
```python
@dataclass
class Config:
    # ... existing fields ...
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
```

#### File: `configs/visualization/default.yaml`
**Create new Hydra configuration group:**
```yaml
# Conservative defaults to minimize training overhead
enabled: true
train_every_n_steps: 0  # Disabled by default
val_first_batch_every_n_epochs: 1
max_length: 4
stride: 1
caption: "这是一个测试视频"
```

#### File: `configs/config.yaml`
**Add visualization to defaults:**
```yaml
defaults:
  # ... existing defaults ...
  - visualization: default
```

### 2. Dependencies Update

#### File: `requirements.txt`
**Ensure SwanLab is declared:**
```
swanlab>=0.3.4
```

### 3. Trainer Enhancement

#### File: `core/trainer.py`

**Bug Fix: Fix validation forward unpacking**
```python
# In validation_step method:
# BEFORE: outs_per_frame = self.forward(batch)
# AFTER: outs_per_frame, obj_to_cat = self.forward(batch)
```

**Add helper methods to maintain DRY principle:**

```python
def _should_log_gif(self, split: str, batch_idx: int) -> bool:
    """Check if GIF should be logged for current step/epoch."""
    if not self.config.visualization.enabled or not self.trainer.is_global_zero:
        return False
    
    if split == "train":
        steps = self.config.visualization.train_every_n_steps
        return steps > 0 and self.global_step % steps == 0
    elif split == "val":
        epochs = self.config.visualization.val_first_batch_every_n_epochs
        return batch_idx == 0 and self.current_epoch % epochs == 0
    
    return False

def _log_gif(self, frames: torch.Tensor, gt_masks: torch.Tensor, 
             outs_per_frame: List[Dict], obj_to_cat: List[int], split: str) -> None:
    """Create and log GIF visualization."""
    try:
        from core.utils import create_visualization_gif
        import swanlab
        
        gif_path = create_visualization_gif(
            frames=frames,
            gt_masks=gt_masks,
            outs_per_frame=outs_per_frame,
            obj_to_cat=obj_to_cat,
            max_length=self.config.visualization.max_length,
            stride=self.config.visualization.stride,
        )
        
        if gif_path:
            caption = f"{self.config.visualization.caption} | {split} e{self.current_epoch} s{self.global_step}"
            self.logger.experiment.log({
                "video": swanlab.Video(gif_path, caption=caption)
            })
            logger.info(f"Logged {split} GIF: {gif_path}")
    except Exception as e:
        logger.warning(f"Failed to log {split} GIF: {e}")
```

**Integration in training_step:**
```python
def training_step(self, batch: BatchedVideoDatapoint, batch_idx: int) -> torch.Tensor:
    # ... existing forward pass and loss computation ...
    
    # GIF logging for training
    if self._should_log_gif("train", batch_idx):
        frames = batch.img_batch.squeeze(1)  # [T, C, H, W]
        self._log_gif(frames, batch.masks, outs_per_frame, obj_to_cat, "train")
    
    return total_loss
```

**Integration in validation_step:**
```python
def validation_step(self, batch: BatchedVideoDatapoint, batch_idx: int) -> torch.Tensor:
    # ... existing forward pass and loss computation ...
    
    # GIF logging for validation
    if self._should_log_gif("val", batch_idx):
        frames = batch.img_batch.squeeze(1)  # [T, C, H, W]
        self._log_gif(frames, batch.masks, outs_per_frame, obj_to_cat, "val")
    
    # ... rest of existing validation logic ...
```

### 4. Existing Files (No Changes Required)

#### File: `core/utils.py`
- **No modifications needed** - reuse existing `create_visualization_gif()` and `create_composite_visualization()` as-is

## Usage Examples

### Command Line Overrides
```bash
# Enable train GIFs every 500 steps
python train.py visualization.train_every_n_steps=500

# Log val first batch every 2 epochs
python train.py visualization.val_first_batch_every_n_epochs=2

# Shorter GIFs with faster playback
python train.py visualization.max_length=3 visualization.stride=1

# Custom caption
python train.py visualization.caption="训练/验证可视化"

# Disable visualization entirely
python train.py visualization.enabled=false
```

### Configuration File Customization
Create `configs/visualization/frequent.yaml`:
```yaml
enabled: true
train_every_n_steps: 100
val_first_batch_every_n_epochs: 1
max_length: 6
stride: 2
caption: "高频可视化"
```

Use with: `python train.py visualization=frequent`

## Implementation Safeguards

### 1. Performance Protection
- **Conservative defaults**: Train GIFs disabled by default (`train_every_n_steps=0`)
- **Minimal frequency**: Val GIFs only on first batch per epoch
- **Rank safety**: All logging guarded by `self.trainer.is_global_zero`

### 2. Error Handling
- **Graceful degradation**: Try/except around SwanLab import and logging
- **Non-blocking**: Failures logged as warnings, don't crash training
- **Path validation**: Check GIF path exists before logging

### 3. Assumptions & Limitations
- **Batch size = 1**: Current visualization utilities assume single batch item
- **Memory constraints**: GIF creation happens in memory, appropriate for short clips
- **Temporary files**: GIFs saved to `/tmp` directory by existing utility

## Validation Plan

### Unit Testing
1. **Config validation**: Ensure new VisualizationConfig loads correctly
2. **Helper methods**: Test `_should_log_gif` logic for different scenarios
3. **Error handling**: Verify graceful degradation when SwanLab unavailable

### Integration Testing
1. **Training run**: Verify GIFs generated and logged during training
2. **Validation run**: Confirm val GIFs appear in SwanLab interface  
3. **Performance impact**: Measure training speed with/without visualization
4. **Multi-GPU**: Test rank safety in distributed training setup

### Manual Verification
1. **SwanLab interface**: Check videos appear with correct captions
2. **GIF quality**: Verify 2x2 composite layout renders clearly
3. **Configuration override**: Test Hydra parameter changes take effect

## Next Steps

This plan provides a complete roadmap for implementing GIF visualization with SwanLab logging while maintaining the existing architecture's simplicity and performance characteristics. The implementation follows KISS/YAGNI/DRY principles by reusing existing utilities, adding minimal configuration, and centralizing logging logic.