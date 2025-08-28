# SAM2 Video Training - Dataset and Model Components Guide

## Dataset Components

### VideoDataset

The `VideoDataset` class provides basic video dataset functionality for training with synthetic mask generation.

**Constructor Parameters:**
- `data_path` (str): Path to video data directory
- `image_size` (Tuple[int, int]): Target image size (default: (512, 512))
- `video_clip_length` (int): Number of frames per clip (default: 5)
- `prompt_types` (List[str]): Types of prompts to generate (default: None)
- `number_of_points` (Tuple[int, int]): Range of points to sample per mask (default: (1, 3))
- `include_center` (bool): Whether to include the center point of the mask (default: False)
- `num_of_neg_points` (int): Number of negative points to sample (default: 0)

**Key Features:**
- Loads video frames from directory structure
- Generates synthetic masks for multiple objects
- Creates random prompts from masks
- Supports variable number of objects per video clip
- Handles frame padding and random clip selection

**Data Format:**
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

### COCODataset

The `COCODataset` class provides COCO format dataset functionality for video training with ground truth masks.

**Constructor Parameters:**
- `coco_json_path` (str): Path to COCO JSON annotation file
- `image_size` (Tuple[int, int]): Target image size (default: (512, 512))
- `video_clip_length` (int): Number of frames per clip (default: 5)
- `prompt_types` (List[str]): Types of prompts to generate (default: None)
- `num_of_pos_points` (int): Number of positive points to sample (default: 1)
- `include_center` (bool): Whether to include the center point of the mask (default: False)
- `num_of_neg_points` (int): Number of negative points to sample (default: 0)

**Required COCO JSON Format:**
```json
{
  "images": [
    {
      "id": 1,
      "video_id": 1,
      "order_in_video": 0,
      "path": "/path/to/frame_001.jpg",
      "width": 512,
      "height": 512
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "segmentation": {
        "size": [512, 512],
        "counts": "..."
      }
    }
  ]
}
```

**Key Features:**
- Loads ground truth masks from COCO format annotations
- Supports RLE (Run-Length Encoding) mask decoding
- Caches decoded masks for performance
- Handles variable number of objects per frame
- Generates prompts from ground truth masks

### PromptGenerator

The `PromptGenerator` class generates random prompts from ground truth masks for model training.

**Constructor Parameters:**
- `prompt_types` (List[str]): Types of prompts to generate (default: ["point", "bbox", "mask"])
- `num_of_pos_points` (int): Number of positive points to sample (default: 1)
- `include_center` (bool): Whether to include the center point (default: False)
- `num_of_neg_points` (int): Number of negative points to sample (default: 0)

**Supported Prompt Types:**
1. **Point Prompts**: Random points inside/outside the mask
2. **Bounding Box Prompts**: Bounding box around the mask
3. **Mask Prompts**: Direct mask input

**Prompt Generation Process:**
1. Select random prompt type from available options
2. Generate appropriate prompts based on mask
3. Handle empty masks with empty prompts
4. Include center point if configured

### Data Loading Pipeline

The data loading pipeline uses PyTorch DataLoader with a custom collate function:

1. **Dataset Creation**: Create dataset instance with configuration
2. **DataLoader Setup**: Configure DataLoader with batch size and workers
3. **Collation**: Custom `collate_fn` handles variable sequence lengths and object counts
4. **Batch Formation**: Create properly structured batches for training

**Collate Function Features:**
- Handles variable sequence lengths by padding
- Manages variable object counts across frames
- Creates BatchedVideoDatapoint structures
- Supports efficient batched processing

## Model Components

### SAM2Model

The `SAM2Model` class is a unified SAM2 model that handles both loading and tracking functionality.

**Constructor Parameters:**
- `checkpoint_path` (str): Path to SAM2 checkpoint
- `config_path` (str): Path to SAM2 config file
- `trainable_modules` (List[str]): List of modules to train (default: ["memory_attention", "memory_encoder"])
- `device` (str): Device to load model on (default: "cuda")
- `image_size` (int): Image size for processing (default: 512)
- `num_maskmem` (int): Number of mask memories (default: 7)
- `**kwargs`: Additional model parameters

**Key Methods:**

1. **load(device)**: Load SAM2 model and set up for tracking
   - Loads base SAM2 model using build_sam2
   - Extracts essential components (image_encoder, memory_attention, memory_encoder)
   - Moves model to specified device
   - Configures for selective training

2. **configure_for_training(trainable_modules)**: Configure model for selective module training
   - Freezes all parameters
   - Unfreezes specified modules
   - Reports training configuration statistics

3. **forward(images, masks, prompts, batched_video_datapoint)**: Forward pass for video tracking
   - Handles both simplified batch format and SAM2 BatchedVideoDatapoint format
   - Processes frames sequentially with temporal consistency
   - Handles multiple objects with object-parallel processing
   - Returns predicted masks and tracking outputs

**Multi-Object Tracking Approach:**
- **Frame-Sequential Processing**: Process video frames in temporal order
- **Object-Parallel Processing**: Handle multiple objects simultaneously within each frame
- **Memory-Aware Tracking**: Utilize SAM2's memory mechanisms for temporal consistency
- **Batched Operations**: Efficient processing using batched tensor operations

**Training Configuration:**
- **Selective Training**: Configure specific modules for training
- **Parameter Management**: Freeze/unfreeze parameters as needed
- **Statistics Reporting**: Count and report trainable parameters

### Model Loading Process

The model loading process follows these steps:

1. **Validation**: Check checkpoint and config file paths
2. **Base Model Loading**: Use SAM2's build_sam2 function
3. **Component Extraction**: Extract essential components from base model
4. **Setup**: Copy SAM2 base functionality to model
5. **Device Placement**: Move model to target device
6. **Training Configuration**: Configure for selective training if specified

### Key Model Features

1. **Unified Interface**: Combines model loading and tracking in single class
2. **Selective Training**: Configure specific modules for training
3. **Multi-Object Support**: Handle multiple objects simultaneously
4. **Memory Integration**: Utilize SAM2's memory mechanisms
5. **Flexible Input**: Support multiple input formats
6. **Comprehensive Output**: Return detailed tracking information

## Component Integration

### Dataset-Model Integration

The dataset components provide data in a format compatible with the SAM2Model:

1. **Tensor Format**: Images [B, T, C, H, W], Masks [B, T, N, H, W]
2. **Prompt Format**: List of dictionaries with prompt information
3. **Batch Structure**: Properly structured batches for efficient processing

### Data Flow Through Components

1. **Data Loading**: Dataset loads and preprocesses video data
2. **Batching**: DataLoader creates batches with collate function
3. **Model Input**: Batches are fed to SAM2Model forward method
4. **Processing**: Model processes frames with memory and tracking
5. **Output**: Model returns predicted masks and tracking information

## Performance Considerations

### Dataset Optimization

1. **Caching**: Cache decoded masks in COCODataset
2. **Efficient Loading**: Use multiple workers for data loading
3. **Memory Management**: Properly size batches for available memory
4. **Preprocessing**: Apply transforms efficiently during loading

### Model Optimization

1. **Selective Training**: Only train necessary modules
2. **Batched Operations**: Process multiple objects/frames simultaneously
3. **Memory Management**: Efficient use of SAM2's memory mechanisms
4. **Device Placement**: Properly manage GPU/CPU memory

## Configuration Examples

### Video Dataset Configuration
```yaml
dataset:
  dataset_type: "video"
  data_path: "/path/to/video/data"
  image_size: [512, 512]
  video_clip_length: 5
  batch_size: 1
  num_workers: 16
  shuffle: True
  prompt_types: ["point"]
  num_of_pos_points: 1
  include_center_point: True
```

### COCO Dataset Configuration
```yaml
dataset:
  dataset_type: "coco"
  data_path: "/path/to/coco_annotations.json"
  image_size: [512, 512]
  video_clip_length: 5
  batch_size: 1
  num_workers: 16
  shuffle: True
  prompt_types: ["point", "bbox"]
  num_of_pos_points: 1
  include_center_point: True
```

### Model Configuration
```yaml
model:
  checkpoint_path: "/path/to/sam2_checkpoint.pt"
  config_path: "configs/sam2.1_hiera_t.yaml"
  trainable_modules: ["memory_attention", "memory_encoder"]
  image_size: 512
  num_maskmem: 7
```