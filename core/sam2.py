"""
SAM2 model module using SAM2Train for advanced video training.
This module wraps SAM2Train to maintain the trainable_modules interface.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import yaml
import numpy as np
from loguru import logger
from beartype import beartype
from icecream import ic

# Import SAM2Train and related components
from core.sam2train import SAM2Train
from sam2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from sam2.build_sam import build_sam2


@beartype
class SAM2Model(SAM2Train):
    """
    SAM2 model wrapper that uses SAM2Train while maintaining the trainable_modules interface.
    This provides compatibility with existing configuration while leveraging SAM2Train features.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        trainable_modules: Optional[List[str]] = None,
        device: str = "cuda",
        image_size: int = 512,
        num_maskmem: int = 7,
        max_objects: int = 10,
        # SAM2Train specific parameters
        prob_to_use_pt_input_for_train: float = 0.0,
        prob_to_use_pt_input_for_eval: float = 0.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        num_frames_to_correct_for_train: int = 1,
        num_frames_to_correct_for_eval: int = 1,
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        num_init_cond_frames_for_train: int = 1,
        num_init_cond_frames_for_eval: int = 1,
        rand_init_cond_frames_for_train: bool = True,
        rand_init_cond_frames_for_eval: bool = False,
        add_all_frames_to_correct_as_cond: bool = False,
        num_correction_pt_per_frame: int = 7,
        pt_sampling_for_eval: str = "center",
        pt_sampling_for_train: str = "uniform",
        prob_to_sample_from_gt_for_train: float = 0.0,
        use_act_ckpt_iterative_pt_sampling: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        **kwargs,
    ):
        """
        Initialize SAM2 model with SAM2Train backend.

        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config_path: Path to SAM2 config file
            trainable_modules: List of module names to train
            device: Device to load model on
            image_size: Image size for processing
            num_maskmem: Number of mask memories
            max_objects: Maximum number of objects to track
            **kwargs: Additional SAM2Train parameters
        """
        # Store configuration
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.image_size = image_size
        self.max_objects = max_objects
        
        # Convert trainable_modules to freeze parameters
        trainable_modules = trainable_modules or ["memory_attention", "memory_encoder"]
        
        # Map trainable_modules to freeze parameters
        freeze_image_encoder = "image_encoder" not in trainable_modules
        freeze_memory_attention = "memory_attention" not in trainable_modules
        freeze_memory_encoder = "memory_encoder" not in trainable_modules
        freeze_sam_prompt_encoder = "sam_prompt_encoder" not in trainable_modules
        freeze_sam_mask_decoder = "sam_mask_decoder" not in trainable_modules
        freeze_obj_ptr_proj = "obj_ptr_proj" not in trainable_modules
        freeze_obj_ptr_tpos_proj = "obj_ptr_tpos_proj" not in trainable_modules

        # Build the base SAM2 model first
        logger.info(f"Building SAM2 model from config: {config_path}")
        sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        
        # Initialize SAM2Train with the built model components
        super().__init__(
            image_encoder=sam2_model.image_encoder,
            memory_attention=sam2_model.memory_attention,
            memory_encoder=sam2_model.memory_encoder,
            sam_prompt_encoder=sam2_model.sam_prompt_encoder,
            sam_mask_decoder=sam2_model.sam_mask_decoder,
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=sam2_model.backbone_stride,
            sigmoid_scale_for_mem_enc=sam2_model.sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc=sam2_model.sigmoid_bias_for_mem_enc,
            use_mask_input_as_output_without_sam=sam2_model.use_mask_input_as_output_without_sam,
            directly_add_no_mem_embed=sam2_model.directly_add_no_mem_embed,
            use_high_res_features_in_sam=sam2_model.use_high_res_features_in_sam,
            multimask_output_in_sam=sam2_model.multimask_output_in_sam,
            multimask_min_pt_num=sam2_model.multimask_min_pt_num,
            multimask_max_pt_num=sam2_model.multimask_max_pt_num,
            use_multimask_token_for_obj_ptr=sam2_model.use_multimask_token_for_obj_ptr,
            compile_image_encoder=sam2_model.compile_image_encoder,
            # SAM2Train specific parameters
            prob_to_use_pt_input_for_train=prob_to_use_pt_input_for_train,
            prob_to_use_pt_input_for_eval=prob_to_use_pt_input_for_eval,
            prob_to_use_box_input_for_train=prob_to_use_box_input_for_train,
            prob_to_use_box_input_for_eval=prob_to_use_box_input_for_eval,
            num_frames_to_correct_for_train=num_frames_to_correct_for_train,
            num_frames_to_correct_for_eval=num_frames_to_correct_for_eval,
            rand_frames_to_correct_for_train=rand_frames_to_correct_for_train,
            rand_frames_to_correct_for_eval=rand_frames_to_correct_for_eval,
            num_init_cond_frames_for_train=num_init_cond_frames_for_train,
            num_init_cond_frames_for_eval=num_init_cond_frames_for_eval,
            rand_init_cond_frames_for_train=rand_init_cond_frames_for_train,
            rand_init_cond_frames_for_eval=rand_init_cond_frames_for_eval,
            add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
            num_correction_pt_per_frame=num_correction_pt_per_frame,
            pt_sampling_for_eval=pt_sampling_for_eval,
            pt_sampling_for_train=pt_sampling_for_train,
            prob_to_sample_from_gt_for_train=prob_to_sample_from_gt_for_train,
            use_act_ckpt_iterative_pt_sampling=use_act_ckpt_iterative_pt_sampling,
            forward_backbone_per_frame_for_eval=forward_backbone_per_frame_for_eval,
            # Freeze parameters
            freeze_image_encoder=freeze_image_encoder,
            freeze_memory_attention=freeze_memory_attention,
            freeze_memory_encoder=freeze_memory_encoder,
            freeze_sam_prompt_encoder=freeze_sam_prompt_encoder,
            freeze_sam_mask_decoder=freeze_sam_mask_decoder,
            freeze_obj_ptr_proj=freeze_obj_ptr_proj,
            freeze_obj_ptr_tpos_proj=freeze_obj_ptr_tpos_proj,
        )
        
        self.loaded = True
        logger.info("SAM2Train model initialized successfully")

    def load(self, device: str = None) -> "SAM2Model":
        """
        Compatibility method - model is already loaded during initialization.
        
        Args:
            device: Device to move model to
            
        Returns:
            Self for method chaining
        """
        if device and device != self.device:
            self.to(device)
            self.device = device
            logger.info(f"Moved model to device: {device}")
        
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass that converts simple format to BatchedVideoDatapoint.
        
        Args:
            batch: Dictionary with 'images' (B,T,C,H,W) and 'masks' (B,T,N,H,W)
            
        Returns:
            List of frame outputs compatible with existing loss functions
        """
        # Convert to BatchedVideoDatapoint format
        batched_input = self._create_batched_video_datapoint(batch)
        
        # Call parent's forward method
        outputs = super().forward(batched_input)
        
        return outputs

    def _create_batched_video_datapoint(self, batch: Dict[str, torch.Tensor]) -> BatchedVideoDatapoint:
        """
        Convert simple batch format to BatchedVideoDatapoint.
        
        Args:
            batch: Dictionary with 'images' and 'masks'
            
        Returns:
            BatchedVideoDatapoint for SAM2Train
        """
        images = batch["images"]  # [B, T, C, H, W]
        masks = batch["masks"]    # [B, T, N, H, W]
        
        B, T, C, H, W = images.shape
        N = masks.shape[2]  # Number of objects
        
        # Transpose to [T, B, C, H, W] as expected by BatchedVideoDatapoint
        img_batch = images.transpose(0, 1)
        
        # Create obj_to_frame_idx tensor [T, O, 2] where O is total objects
        obj_to_frame_idx = []
        masks_flat = []
        unique_objects_identifier = []
        
        for t in range(T):
            for b in range(B):
                for n in range(N):
                    # Check if mask has any content (not empty)
                    if masks[b, t, n].sum() > 0:
                        obj_to_frame_idx.append([t, b])
                        masks_flat.append(masks[b, t, n])
                        unique_objects_identifier.append([b, n, t])  # [video_id, obj_id, frame_id]
        
        if not obj_to_frame_idx:
            # Handle case with no objects - create dummy data
            obj_to_frame_idx = torch.zeros((T, 1, 2), dtype=torch.int)
            masks_tensor = torch.zeros((T, 1, H, W), dtype=torch.bool)
            unique_objects_identifier = torch.zeros((T, 1, 3), dtype=torch.long)
        else:
            # Convert to proper tensor format
            obj_to_frame_idx = torch.tensor(obj_to_frame_idx, dtype=torch.int)
            
            # Reshape obj_to_frame_idx to [T, O, 2] format
            objects_per_frame = len(obj_to_frame_idx) // T if len(obj_to_frame_idx) > 0 else 1
            if len(obj_to_frame_idx) % T != 0:
                # Pad to make it divisible by T
                pad_size = T - (len(obj_to_frame_idx) % T)
                pad_obj_to_frame = obj_to_frame_idx[-1:].repeat(pad_size, 1)
                obj_to_frame_idx = torch.cat([obj_to_frame_idx, pad_obj_to_frame])
                
                # Pad masks_flat too
                pad_masks = [masks_flat[-1]] * pad_size
                masks_flat.extend(pad_masks)
                
                # Pad unique_objects_identifier
                pad_unique = [unique_objects_identifier[-1]] * pad_size
                unique_objects_identifier.extend(pad_unique)
            
            objects_per_frame = len(obj_to_frame_idx) // T
            obj_to_frame_idx = obj_to_frame_idx.reshape(T, objects_per_frame, 2)
            
            masks_tensor = torch.stack(masks_flat).reshape(T, objects_per_frame, H, W).bool()
            unique_objects_identifier = torch.tensor(unique_objects_identifier, dtype=torch.long)
            unique_objects_identifier = unique_objects_identifier.reshape(T, objects_per_frame, 3)
        
        # Create metadata
        frame_orig_size = torch.tensor([[H, W]] * (T * objects_per_frame), dtype=torch.long)
        frame_orig_size = frame_orig_size.reshape(T, objects_per_frame, 2)
        
        metadata = BatchedVideoMetaData(
            unique_objects_identifier=unique_objects_identifier.reshape(-1, 3),
            frame_orig_size=frame_orig_size.reshape(-1, 2)
        )
        
        return BatchedVideoDatapoint(
            img_batch=img_batch,
            obj_to_frame_idx=obj_to_frame_idx,
            masks=masks_tensor,
            metadata=metadata,
            dict_key="video_batch"
        )

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter counts.
        
        Returns:
            Dictionary with model information
        """
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": self.device,
            "image_size": self.image_size,
            "max_objects": self.max_objects,
        }

    def save_config(self, path: str) -> None:
        """Save model configuration to file."""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": self.device,
            "image_size": self.image_size,
            "max_objects": self.max_objects,
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)