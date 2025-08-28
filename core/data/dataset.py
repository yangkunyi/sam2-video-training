"""
Simplified dataset module for SAM2 video training.
This module provides video data loading and preprocessing.
"""

import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from loguru import logger

# Try to import pycocotools for RLE decoding, with fallback
from pycocotools import mask as mask_utils


class PromptGenerator:
    """Generate random prompts from ground truth masks."""
    
    def __init__(self, prompt_types: List[str] = None, num_points: Tuple[int, int] = (1, 3)):
        """Initialize prompt generator."""
        self.prompt_types = prompt_types or ["point", "bbox", "mask"]
        self.num_points = num_points
    
    def generate_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate random prompts from mask."""
        # Handle empty mask
        if mask.sum() == 0:
            return self._empty_prompts()
        
        # Select random prompt type
        prompt_type = random.choice(self.prompt_types)
        
        if prompt_type == "point":
            return self._generate_point_prompts(mask)
        elif prompt_type == "bbox":
            return self._generate_bbox_prompts(mask)
        elif prompt_type == "mask":
            return self._generate_mask_prompts(mask)
        else:
            return self._empty_prompts()
    
    def _generate_point_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate point prompts."""
        if mask.sum() == 0:
            return self._empty_prompts()
        
        pos_pixels = np.where(mask > 0)
        neg_pixels = np.where(mask == 0)
        
        # Number of points
        num_total = random.randint(self.num_points[0], self.num_points[1])
        num_pos = min(max(1, num_total // 2), len(pos_pixels[0]))
        num_neg = num_total - num_pos
        
        # Sample positive points
        pos_indices = np.random.choice(len(pos_pixels[0]), min(num_pos, len(pos_pixels[0])), replace=False)
        pos_points = np.column_stack([pos_pixels[1][pos_indices], pos_pixels[0][pos_indices]])
        
        # Sample negative points
        num_neg = min(num_neg, len(neg_pixels[0]))
        if num_neg > 0:
            neg_indices = np.random.choice(len(neg_pixels[0]), num_neg, replace=False)
            neg_points = np.column_stack([neg_pixels[1][neg_indices], neg_pixels[0][neg_indices]])
        else:
            neg_points = np.zeros((0, 2))
        
        # Combine points
        all_points = np.vstack([pos_points, neg_points])
        pos_labels = np.ones(len(pos_points))
        neg_labels = np.zeros(len(neg_points))
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        return {
            "point_coords": torch.from_numpy(all_points).float(),
            "point_labels": torch.from_numpy(all_labels).long(),
        }
    
    def _generate_bbox_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return self._empty_prompts()
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        return {"boxes": bbox}
    
    def _generate_mask_prompts(self, mask: np.ndarray) -> Dict[str, Any]:
        """Generate mask prompts."""
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return {"masks": mask_tensor}
    
    def _empty_prompts(self) -> Dict[str, Any]:
        """Generate empty prompts."""
        return {
            "point_coords": torch.zeros(0, 2),
            "point_labels": torch.zeros(0),
            "boxes": torch.zeros(4),
            "masks": torch.zeros(512, 512),
        }


class VideoDataset(Dataset):
    """Basic video dataset for training."""
    
    def __init__(self, data_path: str, image_size: Tuple[int, int] = (512, 512), 
                 video_clip_length: int = 5, prompt_types: List[str] = None):
        """
        Initialize video dataset.
        
        Args:
            data_path: Path to video data directory
            image_size: Target image size
            video_clip_length: Number of frames per clip
            prompt_types: Types of prompts to generate
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.video_clip_length = video_clip_length
        self.prompt_generator = PromptGenerator(prompt_types)
        
        # Default image transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Find video directories
        self.video_dirs = sorted([
            d for d in self.data_path.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ])
        
        logger.info(f"Found {len(self.video_dirs)} videos in {self.data_path}")
    
    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        """Get a video clip."""
        video_dir = self.video_dirs[idx]
        
        # Find frame files
        frame_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            frame_files.extend(glob.glob(str(video_dir / ext)))
        
        frame_files = sorted(frame_files)
        
        # Random clip selection
        if len(frame_files) < self.video_clip_length:
            # Pad with repeat last frame
            needed = self.video_clip_length - len(frame_files)
            frame_files.extend([frame_files[-1]] * needed)
        else:
            # Random start position
            max_start = len(frame_files) - self.video_clip_length
            start_idx = random.randint(0, max_start)
            frame_files = frame_files[start_idx:start_idx + self.video_clip_length]
        
        # Load frames and masks
        images = []
        masks = []
        prompts = []
        
        for i, frame_path in enumerate(frame_files):
            # Load image
            image = Image.open(frame_path).convert("RGB")
            image_tensor = self.transform(image)
            images.append(image_tensor)
            
            # Create synthetic masks for multiple objects (since no GT provided)
            h, w = self.image_size
            num_objects = random.randint(1, 3)  # 1-3 objects per frame
            mask = np.zeros((num_objects, h, w), dtype=np.float32)
            
            # Generate multiple random rectangle masks
            for obj_idx in range(num_objects):
                mask_h = random.randint(h // 8, h // 4)
                mask_w = random.randint(w // 8, w // 4)
                start_h = random.randint(0, h - mask_h)
                start_w = random.randint(0, w - mask_w)
                mask[obj_idx, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1.0
            
            masks.append(torch.from_numpy(mask).float())
            
            # Generate prompts only for first frame
            if i == 0:
                prompt_dict = self.prompt_generator.generate_prompts(mask)
                prompts.append(prompt_dict)
            else:
                prompts.append({})
        
        # Stack tensors
        images = torch.stack(images)  # [T, C, H, W]
        masks = torch.stack(masks)  # [T, N, H, W] where N is number of objects
        
        # Generate prompts for all objects in first frame
        first_frame_prompts = []
        for obj_idx in range(masks[0].shape[0]):  # For each object
            obj_mask = masks[0][obj_idx].numpy()
            prompt_dict = self.prompt_generator.generate_prompts(obj_mask)
            first_frame_prompts.append(prompt_dict)
        
        return {
            "images": images,
            "masks": masks,
            "prompts": first_frame_prompts,  # Prompts for all objects in first frame
            "video_path": str(video_dir),
            "num_objects": masks[0].shape[0],  # Number of objects
        }


class COCODataset(Dataset):
    """COCO format dataset for video training."""
    
    def __init__(self, coco_json_path: str, image_size: Tuple[int, int] = (512, 512),
                 video_clip_length: int = 5, prompt_types: List[str] = None):
        """
        Initialize COCO dataset.
        
        Args:
            coco_json_path: Path to COCO JSON annotation file
            image_size: Target image size  
            video_clip_length: Number of frames per clip
            prompt_types: Types of prompts to generate
        """
        self.coco_json_path = Path(coco_json_path)
        self.image_size = image_size
        self.video_clip_length = video_clip_length
        self.prompt_generator = PromptGenerator(prompt_types)
        
        # Validate input
        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON file not found: {self.coco_json_path}")
        
        # Load COCO annotations
        try:
            with open(self.coco_json_path, "r") as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in COCO file {self.coco_json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load COCO file {self.coco_json_path}: {e}")
        
        # Validate COCO format
        required_keys = ["images", "annotations"]
        for key in required_keys:
            if key not in coco_data:
                raise ValueError(f"Missing required key '{key}' in COCO JSON file")
        
        self.images = coco_data.get("images", [])
        self.annotations = coco_data.get("annotations", [])
        self.categories = coco_data.get("categories", [])
        
        if not self.images:
            raise ValueError("No images found in COCO dataset")
        if not self.annotations:
            logger.warning("No annotations found in COCO dataset")
        
        # Create image ID to image mapping
        self.image_id_to_image = {img["id"]: img for img in self.images}
        
        # Validate that all images have required fields
        for img in self.images:
            required_img_fields = ["id", "video_id", "order_in_video", "path"]
            for field in required_img_fields:
                if field not in img:
                    raise ValueError(f"Image missing required field '{field}': {img}")
            
            # Validate image path exists
            img_path = Path(img["path"])
            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
        
        # Group images by video
        self.video_to_images = {}
        for img in self.images:
            video_id = img.get("video_id", 0)
            if video_id not in self.video_to_images:
                self.video_to_images[video_id] = []
            self.video_to_images[video_id].append(img)
        
        # Sort by frame order
        for video_id in self.video_to_images:
            self.video_to_images[video_id].sort(key=lambda x: x.get("order_in_video", 0))
        
        self.video_ids = sorted(self.video_to_images.keys())
        
        # Create image ID to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
        
        # Default transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Cache for decoded masks
        self.mask_cache = {}
        
        logger.info(f"Loaded COCO dataset with {len(self.video_ids)} videos")
    
    def _decode_rle_mask(self, rle_data: Dict, height: int, width: int) -> np.ndarray:
        """
        Decode RLE mask to binary mask.
        
        Args:
            rle_data: RLE encoded mask data
            height: Height of the mask
            width: Width of the mask
            
        Returns:
            np.ndarray: Decoded binary mask
        """
        if not isinstance(rle_data, dict):
            logger.warning("Invalid RLE data format")
            return np.zeros((height, width), dtype=np.uint8)
        
        try:
            mask = mask_utils.decode(rle_data)
            return mask
        except Exception as e:
            logger.warning(f"Failed to decode RLE with pycocotools: {e}")
        # Return empty mask if decoding fails
        logger.warning("Failed to decode RLE mask, returning empty mask")
        return np.zeros((height, width), dtype=np.uint8)
    
    def _load_gt_masks_for_image(self, image_id: int, height: int, width: int) -> Dict[int, np.ndarray]:
        """
        Load ground truth masks for a given image.
        
        Args:
            image_id: ID of the image
            height: Height of the image
            width: Width of the image
            
        Returns:
            Dict[int, np.ndarray]: Mapping from object ID to mask
        """
        cache_key = (image_id, height, width)
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        annotations = self.image_id_to_annotations.get(image_id, [])
        masks = {}
        
        for ann in annotations:
            # Validate annotation
            if not isinstance(ann, dict):
                logger.warning(f"Invalid annotation format for image {image_id}")
                continue
                
            obj_id = ann.get("category_id")
            if obj_id is None:
                logger.warning(f"Annotation missing category_id for image {image_id}")
                continue
                
            segmentation = ann.get("segmentation")
            
            if segmentation is None:
                continue
                
            # Decode the segmentation mask
            try:
                if isinstance(segmentation, dict) and "counts" in segmentation:
                    # RLE format
                    mask = self._decode_rle_mask(segmentation, height, width)
                    # Only add non-empty masks
                    if mask.sum() > 0:
                        masks[obj_id] = mask.astype(np.float32)
                elif isinstance(segmentation, list):
                    # Polygon format - simplified handling
                    # In practice, you would need to properly rasterize polygons
                    logger.warning("Polygon segmentation not fully implemented, using placeholder")
                    mask = np.zeros((height, width), dtype=np.float32)
                    # Create a simple mask for now
                    mask[height//4:3*height//4, width//4:3*width//4] = 1.0
                    # Only add non-empty masks
                    if mask.sum() > 0:
                        masks[obj_id] = mask
            except Exception as e:
                logger.warning(f"Failed to decode segmentation for annotation {ann.get('id', 'unknown')}: {e}")
                continue
        
        # Cache the result
        self.mask_cache[cache_key] = masks
        return masks
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        """Get a video clip from COCO."""
        video_id = self.video_ids[idx]
        video_images = self.video_to_images[video_id]
        
        # Random clip selection
        if len(video_images) < self.video_clip_length:
            # Pad with repeat last frame
            needed = self.video_clip_length - len(video_images)
            video_images.extend([video_images[-1]] * needed)
        else:
            # Random start position
            max_start = len(video_images) - self.video_clip_length
            start_idx = random.randint(0, max_start)
            video_images = video_images[start_idx:start_idx + self.video_clip_length]
        
        # Load frames and masks
        images = []
        masks = []
        prompts = []
        height, width = self.image_size
        for i, img_info in enumerate(video_images):
            # Load image
            image_path = img_info["path"]
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            images.append(image_tensor)
            
            # Load ground truth masks
            image_id = img_info["id"]
            
            gt_masks = self._load_gt_masks_for_image(image_id, height, width)
            
            # Convert to tensor format [N, H, W] where N is number of objects
            if gt_masks:
                # Stack all masks
                mask_list = [torch.from_numpy(mask).float() for mask in gt_masks.values()]
                mask_tensor = torch.stack(mask_list, dim=0)  # [N, H, W]
            else:
                # No objects, create empty mask
                mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
            
            masks.append(mask_tensor)
            
            # Generate prompts only for first frame
            if i == 0:
                # For first frame, generate prompts for all objects
                frame_prompts = []
                for obj_id, mask in gt_masks.items():
                    prompt_dict = self.prompt_generator.generate_prompts(mask)
                    frame_prompts.append(prompt_dict)
                prompts.append(frame_prompts)
            else:
                prompts.append([])
        
        # Stack tensors
        images = torch.stack(images)  # [T, C, H, W]
        
        # For masks, we need to handle variable number of objects per frame
        # Find max number of objects across all frames in this clip
        max_objects = max(mask.shape[0] for mask in masks)
        
        # Pad masks to have the same number of objects
        padded_masks = []
        for mask in masks:
            if mask.shape[0] < max_objects:
                # Pad with zero masks
                padding = torch.zeros((max_objects - mask.shape[0], height, width), dtype=mask.dtype)
                mask = torch.cat([mask, padding], dim=0)
            padded_masks.append(mask)
        
        masks = torch.stack(padded_masks)  # [T, N, H, W] where N is max objects
        
        # Generate prompts for all objects in first frame
        first_frame_prompts = prompts[0] if prompts[0] else []
        
        return {
            "images": images,
            "masks": masks,
            "prompts": first_frame_prompts,  # Prompts for all objects in first frame
            "video_path": str(video_id),
            "num_objects": max_objects,  # Maximum number of objects in any frame
        }


def collate_fn(batch):
    """Collate function for video data."""
    # Find max sequence length
    max_len = max(len(item["images"]) for item in batch)
    
    batched_images = []
    batched_masks = []
    prompts = []
    video_paths = []
    num_objects_list = []
    
    for item in batch:
        images = item["images"]  # [T, C, H, W]
        masks = item["masks"]  # [T, N, H, W] where N is number of objects
        
        # Pad sequences
        curr_len = len(images)
        if curr_len < max_len:
            pad_len = max_len - curr_len
            last_image = images[-1].unsqueeze(0)
            last_mask = masks[-1].unsqueeze(0)
            
            pad_images = last_image.repeat(pad_len, 1, 1, 1)
            pad_masks = last_mask.repeat(pad_len, 1, 1, 1)
            
            images = torch.cat([images, pad_images])
            masks = torch.cat([masks, pad_masks])
        
        batched_images.append(images)
        batched_masks.append(masks)
        prompts.append(item["prompts"])
        video_paths.append(item["video_path"])
        num_objects_list.append(item.get("num_objects", 1))
    
    return {
        "images": torch.stack(batched_images),  # [B, T, C, H, W]
        "masks": torch.stack(batched_masks),  # [B, T, N, H, W]
        "prompts": prompts,
        "video_paths": video_paths,
        "num_objects": num_objects_list,
    }


def create_dataloader(dataset_type: str, dataset_path: str, batch_size: int = 1, 
                    num_workers: int = 4, shuffle: bool = True, max_objects: int = 10, **kwargs):
    """Factory function to create dataloader for multiple objects."""
    if dataset_type.lower() == "video":
        dataset = VideoDataset(data_path=dataset_path, **kwargs)
    elif dataset_type.lower() == "coco":
        dataset = COCODataset(coco_json_path=dataset_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )