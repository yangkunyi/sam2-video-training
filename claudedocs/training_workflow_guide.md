# SAM2 Video Training - Training Workflow and Loss Functions Guide

## Training Workflow Overview

The SAM2 Video Training project uses PyTorch Lightning to streamline the training process while maintaining flexibility and scalability. The training workflow follows a structured approach that integrates dataset loading, model training, validation, and checkpointing.

## Lightning Module Components

### SAM2LightningModule

The `SAM2LightningModule` class encapsulates the training logic for SAM2 video training:

**Constructor Parameters:**
- `config` (Config): Training configuration containing all settings

**Key Methods:**

1. **setup(stage)**: Setup model when stage starts
   - Initializes SAM2 model with configuration
   - Loads model checkpoint and config
   - Reports model information (total/trainable parameters)

2. **configure_optimizers()**: Configure optimizer and scheduler
   - Sets up optimizer based on configuration (AdamW, Adam)
   - Configures learning rate scheduler (CosineAnnealingLR, StepLR)
   - Reports optimizer and parameter information

3. **forward(images, masks, prompts)**: Forward pass through model
   - Delegates to SAM2 model forward method
   - Supports BatchedVideoDatapoint format

4. **training_step(batch, batch_idx)**: Training step implementation
   - Processes batch through forward pass
   - Computes loss with multiple components
   - Logs training metrics
   - Returns total loss for backpropagation

5. **validation_step(batch, batch_idx)**: Validation step implementation
   - Processes batch through forward pass
   - Computes validation loss
   - Stores outputs for epoch aggregation

6. **on_validation_epoch_end()**: Validation epoch end callback
   - Aggregates validation metrics
   - Logs validation results
   - Saves best model checkpoint
   - Clears validation outputs

7. **on_train_end()**: Training end callback
   - Saves final training metrics
   - Reports training completion

### SAM2LightningDataModule

The `SAM2LightningDataModule` class manages data loading for training:

**Constructor Parameters:**
- `config` (Config): Configuration containing dataset settings

**Key Methods:**

1. **setup(stage)**: Setup datasets for training/validation
   - Creates training dataset with configuration
   - Creates validation dataset with configuration

2. **train_dataloader()**: Return training dataloader
   - Returns configured training DataLoader

3. **val_dataloader()**: Return validation dataloader
   - Returns configured validation DataLoader

## Training Process Flow

### Initialization Phase
1. **Configuration Loading**: Load configuration from YAML or command line
2. **Model Setup**: Create and load SAM2 model with specified checkpoint
3. **Dataset Setup**: Create training and validation datasets
4. **Optimizer Configuration**: Set up optimizer and scheduler based on config
5. **Logger Setup**: Configure Wandb/TensorBoard logging

### Training Loop
1. **Epoch Start**: Begin new training epoch
2. **Training Steps**:
   - Load batch from training dataloader
   - Forward pass through model
   - Compute multi-component loss
   - Backpropagation and parameter updates
   - Log training metrics
3. **Validation**:
   - Run validation steps at configured intervals
   - Compute validation loss and metrics
   - Save best model checkpoint
4. **Epoch End**: Complete epoch and prepare for next

### Batch Processing Flow
1. **Data Loading**: Load batch from dataset with collate function
2. **Tensor Formatting**: Ensure proper tensor dimensions [B, T, N, H, W]
3. **Model Forward Pass**: Process through SAM2 model
4. **Loss Computation**: Calculate combined loss with components
5. **Backpropagation**: Compute gradients and update parameters
6. **Logging**: Record metrics and learning rate

## Loss Functions

### SAM2TrainingLoss

The `SAM2TrainingLoss` class implements a combined loss function for SAM2 training:

**Constructor Parameters:**
- `bce_weight` (float): Weight for BCE loss (default: 1.0)
- `dice_weight` (float): Weight for Dice loss (default: 1.0)
- `iou_weight` (float): Weight for IoU loss (default: 0.5)
- `temporal_weight` (float): Weight for temporal consistency loss (default: 0.1)
- `smooth` (float): Smoothing factor for calculations (default: 1e-6)

**Forward Method Parameters:**
- `pred_masks` (torch.Tensor): Predicted masks [B, T, N, H, W]
- `target_masks` (torch.Tensor): Target masks [B, T, N, H, W]
- `valid_masks` (Optional[torch.Tensor]): Valid mask indicators [B, T, N]
- `return_components` (bool): Whether to return individual loss components

### Loss Components

1. **BCE Loss (Binary Cross-Entropy)**
   - Measures pixel-wise prediction accuracy
   - Uses `F.binary_cross_entropy_with_logits`
   - Averaged over spatial dimensions
   - Weighted by valid masks

2. **Dice Loss**
   - Measures spatial overlap between predicted and target masks
   - Computed using Dice coefficient formula
   - Applies sigmoid to predictions
   - Weighted by valid masks

3. **IoU Loss (Intersection over Union)**
   - Measures region-based segmentation accuracy
   - Computed using IoU formula
   - Applies sigmoid to predictions
   - Weighted by valid masks

4. **Temporal Loss**
   - Ensures consistency between consecutive frames
   - Computes frame-to-frame differences
   - Uses MSE between predicted and target differences
   - Weighted by valid mask pairs

### Loss Combination

The total loss is computed as a weighted sum of all components:
```
total_loss = bce_weight * bce_loss + 
             dice_weight * dice_loss + 
             iou_weight * iou_loss +
             temporal_weight * temporal_loss
```

### Multi-Object Handling

The loss function handles multiple objects by:
1. **Flattening**: Reshape tensors to process all objects together
2. **Valid Masking**: Apply valid masks to ignore empty/padded objects
3. **Averaging**: Compute weighted averages across all valid predictions
4. **Temporal Consistency**: Ensure consistency across time steps for each object

## Training Features

### Automatic Checkpointing
- **Best Model Saving**: Save checkpoint when validation loss improves
- **Configuration Storage**: Save training configuration with checkpoints
- **Metric Tracking**: Store training metrics with final model

### Comprehensive Logging
- **Training Metrics**: Log loss components and learning rate
- **Validation Metrics**: Track validation performance
- **Wandb Integration**: Optional Wandb/SwanLab experiment tracking
- **TensorBoard Support**: Built-in TensorBoard logging

### Learning Rate Scheduling
- **Cosine Annealing**: Gradually decrease learning rate
- **Step Scheduling**: Reduce learning rate at specific intervals
- **Monitoring**: Track learning rate changes during training

### Early Stopping
- **Patience Configuration**: Set patience for early stopping
- **Validation Monitoring**: Monitor validation loss for improvement
- **Training Termination**: Stop training when no improvement

## Multi-Object Training

### Object-Parallel Processing
- **Batched Operations**: Process multiple objects simultaneously
- **Memory Efficiency**: Shared computations across objects
- **Temporal Consistency**: Maintain tracking across time steps

### Variable Object Handling
- **Padding Management**: Handle variable object counts per batch
- **Valid Masking**: Ignore padded/empty objects in loss computation
- **Dynamic Sizing**: Adapt to different object counts per video

### Prompt Integration
- **Multi-Object Prompts**: Handle prompts for multiple objects
- **First-Frame Initialization**: Initialize tracking with prompts
- **Prompt Types**: Support points, bounding boxes, and mask prompts

## Performance Optimization

### Mixed Precision Training
- **16-bit Operations**: Reduce memory usage and increase speed
- **Automatic Scaling**: Handle gradient scaling for stability
- **Precision Configuration**: Configure via trainer settings

### Distributed Training
- **Multi-GPU Support**: Train across multiple GPUs
- **Automatic Distribution**: Handle data/model parallelization
- **Device Configuration**: Configure via trainer settings

### Gradient Accumulation
- **Memory Management**: Handle larger effective batch sizes
- **Accumulation Steps**: Configure via trainer settings
- **Gradient Clipping**: Prevent gradient explosion

## Training Monitoring

### Metric Tracking
- **Loss Components**: Track BCE, Dice, IoU, and temporal losses
- **Learning Rate**: Monitor learning rate changes
- **Validation Performance**: Track validation loss improvements
- **Training Progress**: Monitor epoch and batch progress

### Model Information
- **Parameter Counting**: Report total and trainable parameters
- **Training Configuration**: Log configured trainable modules
- **Device Information**: Report model device placement

### Best Practices for Training

1. **Start Small**: Begin with smaller learning rates and batch sizes
2. **Monitor Metrics**: Watch loss components for convergence
3. **Validation Frequency**: Adjust validation check intervals
4. **Early Stopping**: Configure patience to prevent overfitting
5. **Checkpoint Management**: Regularly save best models
6. **Resource Monitoring**: Watch GPU memory and utilization

## Configuration Examples

### Basic Training Configuration
```yaml
trainer:
  accelerator: "auto"
  devices: "auto"
  precision: "16-mixed"
  max_epochs: 50
  gradient_clip_val: 1.0
  log_every_n_steps: 50
```

### Optimizer Configuration
```yaml
optimizer:
  type: "AdamW"
  lr: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]
```

### Scheduler Configuration
```yaml
scheduler:
  enabled: true
  type: "CosineAnnealingLR"
  T_max: 10
  eta_min: 1e-6
```

### Loss Configuration
```yaml
loss:
  bce_weight: 1.0
  dice_weight: 1.0
  iou_weight: 0.5
  temporal_weight: 0.1
```