# SAM2 Video Training - API Reference

## Main Modules

### train.py

Main training script for SAM2 video training.

#### Functions

##### setup_logging(log_level: str = "INFO")
Setup logging configuration.

**Parameters:**
- `log_level` (str): Logging level (default: "INFO")

##### create_trainer(config: Config, callbacks: Optional[list] = None) -> L.Trainer
Create Lightning trainer instance.

**Parameters:**
- `config` (Config): Training configuration
- `callbacks` (Optional[list]): Additional callbacks (default: None)

**Returns:**
- `L.Trainer`: Configured Lightning trainer

##### train(config: Config)
Main training function.

**Parameters:**
- `config` (Config): Training configuration

##### hydra_main(cfg: DictConfig) -> None
Main entry point using Hydra.

**Parameters:**
- `cfg` (DictConfig): Hydra configuration

##### main() -> None
Main entry point - kept for backward compatibility.

### config.py

Simplified configuration system for SAM2 video training.

#### Classes

##### ModelConfig
Configuration for SAM2 model.

**Attributes:**
- `checkpoint_path` (str): Path to SAM2 checkpoint
- `config_path` (str): Path to SAM2 config file
- `trainable_modules` (List[str]): List of modules to train
- `device` (str): Device for model loading
- `num_maskmem` (int): Number of mask memories
- `image_size` (int): Image size for processing
- `backbone_stride` (int): Backbone stride
- `use_obj_ptrs_in_encoder` (bool): Use object pointers in encoder
- `max_obj_ptrs_in_encoder` (int): Maximum object pointers in encoder
- `add_tpos_enc_to_obj_ptrs` (bool): Add positional encoding to object pointers
- `proj_tpos_enc_in_obj_ptrs` (bool): Project temporal position encoding
- `use_signed_tpos_enc_to_obj_ptrs` (bool): Use signed temporal position encoding
- `multimask_output_in_sam` (bool): Multi-mask output in SAM
- `multimask_output_for_tracking` (bool): Multi-mask output for tracking
- `max_objects` (int): Maximum number of objects to track

##### DatasetConfig
Configuration for training dataset.

**Attributes:**
- `dataset_type` (str): Dataset type ("video" or "coco")
- `data_path` (str): Path to dataset
- `image_size` (Tuple[int, int]): Target image size
- `video_clip_length` (int): Number of frames per video clip
- `batch_size` (int): Training batch size
- `num_workers` (int): Number of data loading workers
- `shuffle` (bool): Shuffle dataset
- `prompt_types` (List[str]): Types of prompts to generate
- `num_of_pos_points` (int): Number of positive points
- `num_of_neg_points` (int): Number of negative points
- `include_center_point` (bool): Include center point in prompts

##### ValDatasetConfig
Configuration for validation dataset.

**Attributes:**
- Same as DatasetConfig

##### LossConfig
Configuration for loss functions.

**Attributes:**
- `bce_weight` (float): Weight for BCE loss
- `dice_weight` (float): Weight for Dice loss
- `iou_weight` (float): Weight for IoU loss
- `temporal_weight` (float): Weight for temporal consistency loss
- `smooth` (float): Smoothing factor for calculations

##### OptimizerConfig
Configuration for optimizer.

**Attributes:**
- `type` (str): Optimizer type
- `lr` (float): Learning rate
- `weight_decay` (float): Weight decay
- `betas` (Tuple[float, float]): Adam optimizer betas
- `eps` (float): Adam optimizer epsilon

##### SchedulerConfig
Configuration for learning rate scheduler.

**Attributes:**
- `enabled` (bool): Enable learning rate scheduler
- `type` (str): Scheduler type
- `T_max` (int): Cosine annealing T_max
- `eta_min` (float): Cosine annealing eta_min
- `step_size` (int): StepLR step size
- `gamma` (float): StepLR gamma

##### TrainerConfig
Configuration for PyTorch Lightning trainer.

**Attributes:**
- `accelerator` (str): Training accelerator
- `devices` (Any): Number of devices
- `precision` (str): Training precision
- `max_epochs` (int): Maximum training epochs
- `gradient_clip_val` (float): Gradient clipping value
- `accumulate_grad_batches` (int): Gradient accumulation steps
- `limit_train_batches` (float): Limit training batches
- `limit_val_batches` (float): Limit validation batches
- `val_check_interval` (float): Validation check interval
- `num_sanity_val_steps` (int): Sanity validation steps
- `enable_checkpointing` (bool): Enable checkpointing
- `enable_progress_bar` (bool): Enable progress bar
- `logger` (bool): Enable logging
- `log_every_n_steps` (int): Log every n steps
- `early_stopping_patience` (Optional[int]): Early stopping patience

##### Config
Root configuration containing all settings.

**Attributes:**
- `model` (ModelConfig): SAM2 model configuration
- `dataset` (DatasetConfig): Training dataset configuration
- `valdataset` (ValDatasetConfig): Validation dataset configuration
- `loss` (LossConfig): Loss function configuration
- `optimizer` (OptimizerConfig): Optimizer configuration
- `scheduler` (SchedulerConfig): Learning rate scheduler configuration
- `trainer` (TrainerConfig): Lightning trainer configuration
- `max_objects` (int): Maximum number of objects to track
- `seed` (int): Random seed
- `use_wandb` (bool): Enable Wandb logging
- `wandb_project` (str): Wandb project name
- `output_dir` (str): Output directory
- `log_level` (str): Logging level
- `save_dir` (str): Checkpoint save directory

### core/model/sam2.py

Unified SAM2 model module for loading and video tracking.

#### Classes

##### SAM2Model(nn.Module)
Unified SAM2 model class that handles both loading and tracking functionality.

**Attributes:**
- `checkpoint_path` (str): Path to SAM2 checkpoint
- `config_path` (str): Path to SAM2 config file
- `trainable_modules` (List[str]): List of modules to train
- `device` (str): Device for model loading
- `image_size` (int): Image size for processing
- `num_maskmem` (int): Number of mask memories
- `loaded` (bool): Model loaded status

**Methods:**

###### __init__(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None, trainable_modules: Optional[List[str]] = None, device: str = "cuda", image_size: int = 512, num_maskmem: int = 7, **kwargs)
Initialize SAM2 model with configuration.

**Parameters:**
- `checkpoint_path` (Optional[str]): Path to SAM2 checkpoint
- `config_path` (Optional[str]): Path to SAM2 config file
- `trainable_modules` (Optional[List[str]]): List of modules to train
- `device` (str): Device for model loading (default: "cuda")
- `image_size` (int): Image size for processing (default: 512)
- `num_maskmem` (int): Number of mask memories (default: 7)
- `**kwargs`: Additional model parameters

###### load(self, device: str = None) -> "SAM2Model"
Load SAM2 model and set up for tracking.

**Parameters:**
- `device` (str): Device to load model on (overwrites initialization device if provided)

**Returns:**
- `SAM2Model`: Self for method chaining

###### configure_for_training(self, trainable_modules: List[str]) -> None
Configure model for selective module training.

**Parameters:**
- `trainable_modules` (List[str]): List of modules to train

###### freeze_all_parameters(self) -> None
Freeze all model parameters.

###### count_trainable_parameters(self) -> int
Count trainable parameters.

**Returns:**
- `int`: Number of trainable parameters

###### count_total_parameters(self) -> int
Count total parameters.

**Returns:**
- `int`: Total number of parameters

###### forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None, prompts: Optional[List[Dict[str, Any]]] = None, batched_video_datapoint: Optional[Any] = None) -> Dict[str, torch.Tensor | Dict]
Forward pass for video tracking with multiple objects.

**Parameters:**
- `images` (torch.Tensor): Video frames [B, T, C, H, W]
- `masks` (Optional[torch.Tensor]): Ground truth masks [B, T, N, H, W]
- `prompts` (Optional[List[Dict[str, Any]]]): Prompts for first frame
- `batched_video_datapoint` (Optional[Any]): SAM2 BatchedVideoDatapoint

**Returns:**
- `Dict[str, torch.Tensor | Dict]`: Dictionary with predicted masks

#### Functions

##### create_sam2_model(checkpoint_path: str, config_path: str, trainable_modules: List[str] = None, device: str = "cuda", **kwargs) -> SAM2Model
Factory function to create SAM2 model.

**Parameters:**
- `checkpoint_path` (str): Path to SAM2 checkpoint
- `config_path` (str): Path to SAM2 config file
- `trainable_modules` (List[str]): List of modules to train (default: None)
- `device` (str): Device to load model on (default: "cuda")
- `**kwargs`: Additional model parameters

**Returns:**
- `SAM2Model`: Loaded SAM2 model

### core/data/dataset.py

Simplified dataset module for SAM2 video training.

#### Classes

##### PromptGenerator
Generate random prompts from ground truth masks.

**Attributes:**
- `prompt_types` (List[str]): Types of prompts to generate
- `number_of_pos_points` (int): Number of positive points
- `include_center` (bool): Whether to include center point
- `num_of_neg_points` (int): Number of negative points

**Methods:**

###### __init__(self, prompt_types: List[str] = None, num_of_pos_points: int = 1, include_center: bool = False, num_of_neg_points: int = 0)
Initialize prompt generator.

**Parameters:**
- `prompt_types` (List[str]): Types of prompts to generate (default: None)
- `num_of_pos_points` (int): Number of positive points (default: 1)
- `include_center` (bool): Whether to include center point (default: False)
- `num_of_neg_points` (int): Number of negative points (default: 0)

###### generate_prompts(self, mask: np.ndarray) -> Dict[str, Any]
Generate random prompts from mask.

**Parameters:**
- `mask` (np.ndarray): Ground truth mask

**Returns:**
- `Dict[str, Any]`: Generated prompts

##### VideoDataset(Dataset)
Basic video dataset for training.

**Attributes:**
- `data_path` (Path): Path to video data directory
- `image_size` (Tuple[int, int]): Target image size
- `video_clip_length` (int): Number of frames per clip
- `prompt_generator` (PromptGenerator): Prompt generator instance

**Methods:**

###### __init__(self, data_path: str, image_size: Tuple[int, int] = (512, 512), video_clip_length: int = 5, prompt_types: List[str] = None, number_of_points: Tuple[int, int] = (1, 3), include_center: bool = False, num_of_neg_points: int = 0)
Initialize video dataset.

**Parameters:**
- `data_path` (str): Path to video data directory
- `image_size` (Tuple[int, int]): Target image size (default: (512, 512))
- `video_clip_length` (int): Number of frames per clip (default: 5)
- `prompt_types` (List[str]): Types of prompts to generate (default: None)
- `number_of_points` (Tuple[int, int]): Range of points to sample per mask (default: (1, 3))
- `include_center` (bool): Whether to include center point (default: False)
- `num_of_neg_points` (int): Number of negative points (default: 0)

###### __len__(self) -> int
Get dataset length.

**Returns:**
- `int`: Number of videos in dataset

###### __getitem__(self, idx) -> Dict[str, Any]
Get a video clip.

**Parameters:**
- `idx` (int): Index of video

**Returns:**
- `Dict[str, Any]`: Dictionary with images, masks, prompts, and metadata

##### COCODataset(Dataset)
COCO format dataset for video training.

**Attributes:**
- `coco_json_path` (Path): Path to COCO JSON annotation file
- `image_size` (Tuple[int, int]): Target image size
- `video_clip_length` (int): Number of frames per clip
- `prompt_generator` (PromptGenerator): Prompt generator instance
- `images` (List): COCO images
- `annotations` (List): COCO annotations
- `video_to_images` (Dict): Mapping from video ID to images
- `image_id_to_annotations` (Dict): Mapping from image ID to annotations

**Methods:**

###### __init__(self, coco_json_path: str, image_size: Tuple[int, int] = (512, 512), video_clip_length: int = 5, prompt_types: List[str] = None, num_of_pos_points: int = 1, include_center: bool = False, num_of_neg_points: int = 0)
Initialize COCO dataset.

**Parameters:**
- `coco_json_path` (str): Path to COCO JSON annotation file
- `image_size` (Tuple[int, int]): Target image size (default: (512, 512))
- `video_clip_length` (int): Number of frames per clip (default: 5)
- `prompt_types` (List[str]): Types of prompts to generate (default: None)
- `num_of_pos_points` (int): Number of positive points (default: 1)
- `include_center` (bool): Whether to include center point (default: False)
- `num_of_neg_points` (int): Number of negative points (default: 0)

###### __len__(self) -> int
Get dataset length.

**Returns:**
- `int`: Number of videos in dataset

###### __getitem__(self, idx) -> Dict[str, Any]
Get a video clip from COCO.

**Parameters:**
- `idx` (int): Index of video

**Returns:**
- `Dict[str, Any]`: Dictionary with images, masks, prompts, and metadata

#### Functions

##### collate_fn(batch) -> Dict[str, Any]
Collate function for video data.

**Parameters:**
- `batch`: Batch of dataset items

**Returns:**
- `Dict[str, Any]`: Collated batch

##### create_dataloader(dataset_type: str, dataset_path: str, prompt_types: List[str], batch_size: int = 1, num_workers: int = 4, shuffle: bool = True, num_of_pos_points: int = 1, include_center: bool = False, num_of_neg_points: int = 0, **kwargs)
Factory function to create dataloader for multiple objects.

**Parameters:**
- `dataset_type` (str): Type of dataset ("video" or "coco")
- `dataset_path` (str): Path to dataset
- `prompt_types` (List[str]): Types of prompts to generate
- `batch_size` (int): Batch size (default: 1)
- `num_workers` (int): Number of workers (default: 4)
- `shuffle` (bool): Whether to shuffle (default: True)
- `num_of_pos_points` (int): Number of positive points (default: 1)
- `include_center` (bool): Whether to include center point (default: False)
- `num_of_neg_points` (int): Number of negative points (default: 0)
- `**kwargs`: Additional arguments

**Returns:**
- `DataLoader`: Configured DataLoader

### core/training/trainer.py

Unified SAM2 Lightning trainer module.

#### Classes

##### SAM2LightningModule(L.LightningModule)
Lightning module for SAM2 video training.

**Attributes:**
- `config` (Config): Training configuration
- `model` (SAM2Model): SAM2 model instance
- `criterion` (SAM2TrainingLoss): Loss function
- `best_val_loss` (float): Best validation loss

**Methods:**

###### __init__(self, config: Config)
Initialize Lightning module.

**Parameters:**
- `config` (Config): Training configuration

###### setup(self, stage: str)
Setup model when stage starts.

**Parameters:**
- `stage` (str): Training stage

###### configure_optimizers(self) -> Dict[str, Any]
Configure optimizer and scheduler.

**Returns:**
- `Dict[str, Any]`: Optimizer and scheduler configuration

###### forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None, prompts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]
Forward pass for multiple objects.

**Parameters:**
- `images` (torch.Tensor): Input images
- `masks` (Optional[torch.Tensor]): Ground truth masks
- `prompts` (Optional[List[Dict[str, Any]]]): Prompts

**Returns:**
- `Dict[str, torch.Tensor]`: Model outputs

###### training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor
Training step for multiple objects.

**Parameters:**
- `batch` (Dict[str, Any]): Training batch
- `batch_idx` (int): Batch index

**Returns:**
- `torch.Tensor`: Training loss

###### validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor
Validation step for multiple objects.

**Parameters:**
- `batch` (Dict[str, Any]): Validation batch
- `batch_idx` (int): Batch index

**Returns:**
- `torch.Tensor`: Validation loss

###### on_validation_epoch_end(self)
Validation epoch end.

###### on_train_end(self)
Training end callback.

##### SAM2LightningDataModule(L.LightningDataModule)
Lightning data module for SAM2 training.

**Attributes:**
- `config` (Config): Configuration containing dataset settings
- `train_dataset`: Training dataset
- `val_dataset`: Validation dataset

**Methods:**

###### __init__(self, config: Config)
Initialize data module.

**Parameters:**
- `config` (Config): Configuration containing dataset settings

###### setup(self, stage: str)
Setup datasets for multiple objects.

**Parameters:**
- `stage` (str): Setup stage

###### train_dataloader(self)
Return training dataloader.

**Returns:**
- Training DataLoader

###### val_dataloader(self)
Return validation dataloader.

**Returns:**
- Validation DataLoader

### core/training/loss.py

Simplified loss functions for SAM2 training.

#### Classes

##### SAM2TrainingLoss(nn.Module)
Combined loss function for SAM2 memory module training.

**Attributes:**
- `bce_weight` (float): Weight for BCE loss
- `dice_weight` (float): Weight for Dice loss
- `iou_weight` (float): Weight for IoU loss
- `temporal_weight` (float): Weight for temporal consistency loss
- `smooth` (float): Smoothing factor

**Methods:**

###### __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, iou_weight: float = 0.5, temporal_weight: float = 0.1, smooth: float = 1e-6)
Initialize loss function.

**Parameters:**
- `bce_weight` (float): Weight for BCE loss (default: 1.0)
- `dice_weight` (float): Weight for Dice loss (default: 1.0)
- `iou_weight` (float): Weight for IoU loss (default: 0.5)
- `temporal_weight` (float): Weight for temporal consistency loss (default: 0.1)
- `smooth` (float): Smoothing factor (default: 1e-6)

###### forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor, valid_masks: Optional[torch.Tensor] = None, return_components: bool = False) -> Any
Compute combined loss for multiple objects.

**Parameters:**
- `pred_masks` (torch.Tensor): Predicted masks [B, T, N, H, W]
- `target_masks` (torch.Tensor): Target masks [B, T, N, H, W]
- `valid_masks` (Optional[torch.Tensor]): Valid mask indicators [B, T, N]
- `return_components` (bool): Whether to return individual loss components

**Returns:**
- Total loss tensor or (total_loss, components) dict