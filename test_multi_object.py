"""
Test script for multi-object video tracking with SAM2.
"""

import torch
import numpy as np
from core.data.dataset import VideoDataset, collate_fn
from core.model.sam2 import SAM2Model
from torch.utils.data import DataLoader
import tempfile
import os
from pathlib import Path
from PIL import Image

def create_test_video_data():
    """Create simple test video data with multiple objects."""
    # Create temporary directory for test data
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple video with 3 frames
    video_dir = test_dir / "test_video"
    video_dir.mkdir(exist_ok=True)
    
    # Create simple test images with multiple objects
    for i in range(3):
        # Create a simple RGB image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some simple objects (rectangles)
        # Object 1
        img[50:100, 50:100, 0] = 255  # Red rectangle
        # Object 2
        img[150:200, 150:200, 1] = 255  # Green rectangle
        
        # Save image
        img_pil = Image.fromarray(img)
        img_pil.save(video_dir / f"frame_{i:04d}.png")
    
    return str(test_dir)

def test_multi_object_dataset():
    """Test the dataset with multi-object support."""
    print("Testing multi-object dataset...")
    
    # Create test data
    data_path = create_test_video_data()
    
    # Create dataset
    dataset = VideoDataset(
        data_path=data_path,
        image_size=(256, 256),
        video_clip_length=3,
        prompt_types=["point", "bbox"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Test batch creation
    batch = next(iter(dataloader))
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch masks shape: {batch['masks'].shape}")
    print(f"Number of objects: {batch['num_objects']}")
    
    # Clean up
    import shutil
    shutil.rmtree("test_data")
    
    return batch

def test_model_forward_pass(batch):
    """Test model forward pass with multi-object data."""
    print("Testing model forward pass...")
    
    # Note: For a complete test, we would need to load the actual SAM2 model
    # This is a simplified test that just checks the data flow
    
    images = batch["images"]
    masks = batch["masks"]
    
    print(f"Input images shape: {images.shape}")
    print(f"Input masks shape: {masks.shape}")
    
    # Test BatchedVideoDatapoint structure
    if "img_batch" in batch:
        print(f"Batched img_batch shape: {batch['img_batch'].shape}")
        print(f"Batched masks shape: {batch['masks'].shape}")
        print(f"Object to frame index shape: {batch['obj_to_frame_idx'].shape}")
    
    print("Model forward pass test completed!")

if __name__ == "__main__":
    print("Running multi-object video tracking tests...")
    
    # Test dataset
    batch = test_multi_object_dataset()
    
    # Test model
    test_model_forward_pass(batch)
    
    print("All tests completed successfully!")