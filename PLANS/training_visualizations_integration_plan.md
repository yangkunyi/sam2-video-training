# Training Visualizations Integration Plan

## Overview
This plan outlines the integration of the `log_training_visualizations` function from `core/utils.py` into the SAM2 video training pipeline. The function creates composite visualizations showing original images, ground truth masks, prompts, and predictions.

## Current State Analysis

### Issues with Existing Function
- **Missing PromptData class**: Function references undefined `PromptData` class
- **Incorrect data format**: Expects batch dict with keys like "frames", "gt_masks", "prompts"
- **Logger incompatibility**: Designed for direct SwanLab usage, but project uses SwanLabLogger
- **Not integrated**: Function exists but not called anywhere in training pipeline

### Current Training Architecture
- **Framework**: PyTorch Lightning with `SAM2LightningModule`
- **Data format**: `BatchedVideoDatapoint` objects from SAM2
- **Logger**: `SwanLabLogger` integrated with Lightning
- **Training location**: `training_step()` method in `SAM2LightningModule`

## Required Modifications

### 1. Data Structure Definitions
- **Add PromptData dataclass** in `core/utils.py`
- **Create conversion helpers** for BatchedVideoDatapoint format
- **Handle video sequence data** properly (multiple frames per batch)

### 2. Function Updates
- **Modify log_training_visualizations** to accept BatchedVideoDatapoint
- **Update create_composite_visualization** to handle PromptData properly
- **Add safe logging wrapper** for SwanLabLogger compatibility
- **Handle multi-category masks** and video sequences

### 3. Training Integration
- **Add visualization hook** in SAM2LightningModule
- **Create configuration options** for enabling/disabling visualizations
- **Integrate into training_step** and validation_step
- **Add performance controls** (frequency limits, batch sampling)

### 4. Configuration System
- **Add VisualizationsConfig** to core/config.py
- **Create Hydra config files** for visualization settings
- **Add command-line overrides** for runtime control

## Implementation Plan

### Phase 1: Core Infrastructure (core/utils.py)

#### 1.1 Add PromptData Dataclass
```python
@dataclass
class PromptData:
    prompt_type: str  # 'point', 'bbox', 'mask'
    obj_id: int
    points: Optional[torch.Tensor] = None  # [K, 2] for point prompts
    labels: Optional[torch.Tensor] = None  # [K] for point labels
    bbox: Optional[torch.Tensor] = None    # [4] for bbox prompts
    mask: Optional[torch.Tensor] = None    # [H, W] for mask prompts
```

#### 1.2 Update create_composite_visualization
- Relax prompts parameter to `Optional[List[PromptData]]`
- Handle None/empty prompts gracefully
- Fix nested List[List[...]] expectation
- Ensure proper mask channel handling

#### 1.3 Add Data Conversion Helpers
```python
def to_category_gt_masks(batch: BatchedVideoDatapoint, num_categories: int, video_idx: int) -> torch.Tensor:
    """Convert batch masks [T, B*N, H, W] to [T, C, H, W] format"""
    
def to_video_frames(batch: BatchedVideoDatapoint, video_idx: int) -> torch.Tensor:
    """Extract video frames from batch"""
    
def extract_category_preds(outs_per_frame: List[Dict], key='pred_masks_high_res') -> torch.Tensor:
    """Extract predictions from model outputs"""
    
def build_prompts_from_gt_first_frame(gt_masks_tchw: torch.Tensor, prompt_type: str) -> List[PromptData]:
    """Generate prompts from ground truth first frame"""
```

#### 1.4 Update log_training_visualizations
- New signature accepting BatchedVideoDatapoint and model outputs
- Add video sampling and frame stride controls
- Integrate with SwanLabLogger via safe wrapper
- Handle multi-video batches

#### 1.5 Add Safe Logging Wrapper
```python
def safe_log_image(pl_logger, key: str, image, step=None):
    """Safe image logging with multiple logger fallbacks"""
    # Try different SwanLabLogger patterns
    # Fallback to no-op if logging fails
```

### Phase 2: Configuration System

#### 2.1 Add VisualizationsConfig to core/config.py
```python
@dataclass
class VisualizationsConfig:
    enabled: bool = False
    every_n_steps: int = 200
    max_frames: int = 4
    max_videos: int = 1
    frame_stride: int = 1
    num_categories: Optional[int] = None
```

#### 2.2 Create Hydra Config Files
- `configs/visualizations/default.yaml` with default settings
- Update `configs/config.yaml` to include visualizations group

#### 2.3 Update Root Config
- Add visualizations field to main Config class
- Set default from data.num_categories if None

### Phase 3: Training Integration (core/trainer.py)

#### 3.1 Add Visualization Hook
```python
def maybe_log_visualizations(self, batch, outs_per_frame, batch_idx, stage):
    """Conditional visualization logging with performance controls"""
    # Check if enabled and right step interval
    # Call updated log_training_visualizations
    # Handle multi-GPU synchronization
```

#### 3.2 Integrate into Training Steps
- Add calls to maybe_log_visualizations in training_step
- Add calls to maybe_log_visualizations in validation_step
- Pass appropriate stage parameter ("train"/"val")

#### 3.3 Initialize Configuration
- Load visualizations config in __init__
- Store reference for easy access

## Configuration Options

### Enable/Disable
- `visualizations.enabled: bool` - Master switch for visualizations

### Frequency Controls
- `visualizations.every_n_steps: int` - Log every N steps (default: 200)
- `visualizations.frame_stride: int` - Sample every N frames (default: 1)

### Volume Controls
- `visualizations.max_frames: int` - Max frames per video (default: 4)
- `visualizations.max_videos: int` - Max videos per batch (default: 1)

### Category Configuration
- `visualizations.num_categories: int` - Override data.num_categories

### Usage Examples
```bash
# Enable visualizations
python train.py visualizations.enabled=true

# Custom frequency and volume
python train.py visualizations.enabled=true visualizations.every_n_steps=100 visualizations.max_frames=3

# Override categories
python train.py visualizations.enabled=true visualizations.num_categories=10
```

## Data Flow Integration

### Input Data Path
1. **BatchedVideoDatapoint** from dataloader
2. **Conversion helpers** extract frames, masks, prompts
3. **Model outputs** provide predictions
4. **Visualization function** creates composite images
5. **Safe logger** uploads to SwanLab

### Key Data Transformations
- **Images**: `batch.img_batch [T,B,C,H,W]` → `[T,C,H,W]` per video
- **GT Masks**: `batch.masks [T,B*N,H,W]` → `[T,C,H,W]` per video
- **Predictions**: `outs_per_frame[t]['pred_masks_high_res']` → sigmoid → `[C,H,W]`
- **Prompts**: Built from GT first frame, one per category

## Testing Strategy

### Unit Tests
- **test_viz_utils_shapes**: Verify proper tensor shapes and conversions
- **test_swanlab_fallback**: Test safe logging with different logger types
- **test_conversion_helpers**: Validate data format transformations

### Integration Tests
- **Smoke test**: Run single training step with visualizations enabled
- **Performance test**: Ensure logging doesn't significantly impact training speed
- **Multi-GPU test**: Verify proper synchronization across devices

### Validation Tests
- **Output verification**: Check that logged images contain expected content
- **Configuration test**: Verify all config options work as expected
- **Error handling**: Test graceful failure with edge cases

## Performance Considerations

### Optimization Strategies
- **Conditional execution**: Only run when enabled and at right interval
- **CPU offloading**: Move tensors to CPU for visualization creation
- **Frame limiting**: Cap number of frames and videos processed
- **Async logging**: Consider non-blocking logging calls

### Memory Management
- **Tensor cleanup**: Ensure proper cleanup of intermediate tensors
- **Figure management**: Close matplotlib figures after rendering
- **Batch sampling**: Process only subset of batch for visualization

## Risk Mitigation

### Data Format Mismatches
- **Validation**: Add assertions for expected tensor shapes
- **Fallback**: Skip visualization if data format is unexpected
- **Logging**: Add debug logging for troubleshooting

### Logger Compatibility
- **Wrapper pattern**: Use safe_log_image with multiple fallbacks
- **Graceful degradation**: Continue training if logging fails
- **Configuration**: Allow disabling visualizations if issues arise

### Performance Impact
- **Frequency control**: Default to conservative logging intervals
- **Batch limiting**: Process only first video in multi-video batches
- **Monitoring**: Add timing metrics for visualization overhead

## Implementation Checklist

### Core Infrastructure
- [ ] Add PromptData dataclass to core/utils.py
- [ ] Update create_composite_visualization signature and implementation
- [ ] Implement data conversion helpers
- [ ] Update log_training_visualizations with new signature
- [ ] Add safe_log_image wrapper function

### Configuration
- [ ] Add VisualizationsConfig to core/config.py
- [ ] Create configs/visualizations/default.yaml
- [ ] Update configs/config.yaml to include visualizations group
- [ ] Test configuration loading and overrides

### Training Integration
- [ ] Add maybe_log_visualizations method to SAM2LightningModule
- [ ] Integrate visualization calls into training_step
- [ ] Integrate visualization calls into validation_step
- [ ] Test integration with single and multi-GPU training

### Testing
- [ ] Create unit tests for visualization utilities
- [ ] Test integration with synthetic data
- [ ] Validate SwanLab logging functionality
- [ ] Performance benchmarking

### Documentation
- [ ] Update CLAUDE.md with visualization configuration
- [ ] Add usage examples to documentation
- [ ] Document configuration options and their effects

## Success Criteria

### Functional Requirements
- [ ] Visualizations can be enabled via configuration
- [ ] Composite images show image, GT masks, prompts, and predictions
- [ ] Logging works with SwanLabLogger without errors
- [ ] Performance impact is minimal (< 5% overhead)
- [ ] Configuration options work as expected

### Quality Requirements
- [ ] Code follows existing project conventions
- [ ] Proper error handling and graceful degradation
- [ ] Comprehensive test coverage
- [ ] Clear documentation and examples
- [ ] No breaking changes to existing functionality

## Rollout Plan

### Phase 1: Core Infrastructure
1. Implement PromptData and utility functions
2. Update visualization functions
3. Add unit tests
4. Validate with synthetic data

### Phase 2: Configuration
1. Add configuration classes
2. Create Hydra config files
3. Test configuration loading
4. Verify command-line overrides

### Phase 3: Integration
1. Add training hooks
2. Integrate with training steps
3. Test with real data
4. Performance optimization

### Phase 4: Testing and Documentation
1. Comprehensive testing
2. Documentation updates
3. User guide creation
4. Final validation

This plan provides a comprehensive approach to integrating training visualizations while maintaining code quality, performance, and usability.