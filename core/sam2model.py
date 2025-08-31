"""
Unified SAM2 model for video training that combines configuration management
with core training functionality. Eliminates the wrapper pattern for better
maintainability.
"""

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra.core.global_hydra as ghd
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sam2.build_sam import build_sam2
# SAM2 core imports
from sam2.modeling.sam2_base import SAM2Base

from core import utils
# Local imports
from core.config import ModelConfig
from core.data_utils import BatchedVideoDatapoint


class SAM2Model(SAM2Base):
    """
    Unified SAM2 model for video training that combines configuration management
    with core training functionality. Eliminates the wrapper pattern for better
    maintainability.

    This class provides:
    - Simplified SAM2 training without iterative correction complexity
    - Configuration management and utility methods
    - Direct inheritance from SAM2Base for better performance
    - Sequential frame processing for video object tracking
    """

    def __init__(
        self,
        model_config: "ModelConfig",
    ):
        """
        Initialize unified SAM2 model with configuration management and training capabilities.

        Args:
            model_config: ModelConfig containing all model configuration
        """

        # 1. 先把配置存下来
        self.model_config = model_config
        self.checkpoint_path = model_config.checkpoint_path
        self.config_path = model_config.config_path
        self.image_size = model_config.image_size

        # 2. 一些训练相关的断言/开关
        self.prompt_type = model_config.prompt_type
        assert self.prompt_type in ["point", "box", "mask"], (
            f"prompt_type must be one of ['point', 'box', 'mask'], got {self.prompt_type}"
        )
        self.forward_backbone_per_frame_for_eval = (
            model_config.forward_backbone_per_frame_for_eval
        )

        # 3. 先构建一个完整的 SAM2 模型
        logger.info(f"Building SAM2 model from config: {self.config_path}")
        sam2_model = build_sam2(
            self.config_path, self.checkpoint_path, device=model_config.device
        )
        super().__init__(
            image_encoder=sam2_model.image_encoder,
            memory_attention=sam2_model.memory_attention,
            memory_encoder=sam2_model.memory_encoder,
        )

        for attr_name in dir(sam2_model):
            if attr_name.startswith("_"):  # 私有成员
                continue
            if hasattr(self, attr_name):  # SAM2Base 已经有的
                continue
            if attr_name in ["forward", "forward_image", "device"]:
                continue
            setattr(self, attr_name, getattr(sam2_model, attr_name))

        # 6. 按需冻结权重
        trainable_modules = model_config.trainable_modules or [
            "memory_attention",
            "memory_encoder",
        ]
        self._setup_trainable_modules(trainable_modules)

        self.loaded = True
        logger.info("Unified SAM2 model initialized successfully")

    def load(self, device: str = None) -> "SAM2Model":
        """
        Compatibility method - model is already loaded during initialization.

        Args:
            device: Device to move model to

        Returns:
            Self for method chaining
        """
        return self

    def forward(self, input: BatchedVideoDatapoint) -> List[Dict[str, torch.Tensor]]:
        """Forward pass through simplified SAM2 training."""
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}

        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Simplified prompt preparation - only adds prompt to first frame.

        Args:
            backbone_out: Backbone features dictionary
            input: BatchedVideoDatapoint with video data
            start_frame_idx: Starting frame index (default 0)

        Returns:
            backbone_out: Updated with prompt inputs for first frame only
        """
        # Load ground-truth masks for all frames
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = input.num_frames

        # Initialize prompt containers
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        # Generate prompt for first frame only
        first_frame_cat_mask = gt_masks_per_frame[start_frame_idx]
        first_frame_mask, obj_to_cat = utils.cat_to_obj_mask(first_frame_cat_mask)
        backbone_out["obj_to_cat"] = obj_to_cat

        if self.prompt_type == "mask":
            # Use GT mask directly as prompt
            backbone_out["mask_inputs_per_frame"][start_frame_idx] = first_frame_mask

        elif self.prompt_type == "box":
            # Generate box prompt from GT mask
            points, labels = utils.generate_box_prompt(first_frame_mask)
            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][start_frame_idx] = point_inputs

        elif self.prompt_type == "point":
            # Generate point prompt from GT mask
            points, labels = utils.generate_point_prompt(
                mask=first_frame_mask,
                num_pos_points=self.model_config.num_pos_points,
                num_neg_points=self.model_config.num_neg_points,
                include_center=self.model_config.include_center,
            )
            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][start_frame_idx] = point_inputs

        return backbone_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """
        Simplified forward video tracking - sequential processing without iterative correction.

        Args:
            backbone_out: Backbone features and prompt setup
            input: BatchedVideoDatapoint with video data
            return_dict: Whether to return dict format (default False)

        Returns:
            List of frame outputs for loss calculation
        """
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        num_frames = backbone_out["num_frames"]
        all_frame_outputs = []

        # Sequential processing: [0, 1, 2, ..., num_frames-1]
        for frame_idx in range(num_frames):
            # First frame is conditioning frame (has prompt), others are not
            is_init_cond_frame = frame_idx == 0

            # Get the image features for current frame
            img_ids = input.flat_obj_to_img_idx[frame_idx]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if already computed)
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute image features on the fly for the given img_ids
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(
                    frame_idx, None
                ),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(frame_idx, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(frame_idx, None),
                num_frames=num_frames,
            )

            all_frame_outputs.append(current_out)

        if return_dict:
            # Convert to dict format if requested
            output_dict = {
                "cond_frame_outputs": {0: all_frame_outputs[0]},  # Only first frame
                "non_cond_frame_outputs": {
                    i: all_frame_outputs[i] for i in range(1, num_frames)
                },
            }
            return output_dict

        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        gt_masks=None,
        output_dict=None,  # Kept for compatibility but not used in simple version
    ):
        """
        Simplified track step - single prediction without iterative correction.

        Args:
            frame_idx: Current frame index
            is_init_cond_frame: Whether this is the conditioning frame (first frame)
            current_vision_feats: Vision features for current frame
            current_vision_pos_embeds: Vision position embeddings
            feat_sizes: Feature sizes
            point_inputs: Point prompts (only for first frame)
            mask_inputs: Mask prompts (only for first frame)
            num_frames: Total number of frames
            track_in_reverse: Whether tracking in reverse order (default False)
            run_mem_encoder: Whether to run memory encoder (default True)
            prev_sam_mask_logits: Previous SAM mask logits (default None)
            gt_masks: Ground truth masks (for compatibility)
            output_dict: Output dictionary (for compatibility)

        Returns:
            current_out: Frame prediction output for loss calculation
        """
        if output_dict is None:
            output_dict = {}

        # Core tracking step - this calls the base class functionality
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        # Extract SAM outputs
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # Set up outputs (simplified - no iterative correction)
        # For the simple version, we only have single-step predictions
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Use final prediction for output (no correction needed)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Run memory encoder on the predicted mask to encode it into a new memory feature
        # (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return utils.count_trainable_parameters(self)

    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return utils.count_total_parameters(self)

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter counts.

        Returns:
            Dictionary with model information
        """
        return utils.get_model_info(
            self,
            self.checkpoint_path,
            self.config_path,
            self.device,
            self.image_size,
        )

    def _get_module_mapping(self) -> Dict[str, nn.Module]:
        """
        Get dictionary mapping module names to actual PyTorch modules.

        Returns:
            Dictionary mapping module names to modules
        """
        return {
            "image_encoder": self.image_encoder,
            "memory_attention": self.memory_attention,
            "memory_encoder": self.memory_encoder,
            "prompt_encoder": self.sam_prompt_encoder,
            "mask_decoder": self.sam_mask_decoder,
            "obj_ptr_proj": self.obj_ptr_proj,
            "obj_ptr_tpos_proj": self.obj_ptr_tpos_proj,
        }

    def _setup_trainable_modules(self, trainable_modules: List[str]) -> None:
        """
        Freeze all modules except those in trainable_modules list.

        Args:
            trainable_modules: List of module names that should remain trainable
        """
        module_mapping = self._get_module_mapping()
        utils.setup_trainable_modules(self, module_mapping, trainable_modules)

    def freeze_module(self, module_name: str) -> None:
        """
        Dynamically freeze a specific module.

        Args:
            module_name: Name of the module to freeze
        """
        module_mapping = self._get_module_mapping()
        utils.freeze_module_by_name(module_mapping, module_name)

    def unfreeze_module(self, module_name: str) -> None:
        """
        Dynamically unfreeze a specific module.

        Args:
            module_name: Name of the module to unfreeze
        """
        module_mapping = self._get_module_mapping()
        utils.unfreeze_module_by_name(module_mapping, module_name)

    def get_trainable_modules(self) -> List[str]:
        """
        Return list of currently trainable module names.

        Returns:
            List of module names that have trainable parameters
        """
        module_mapping = self._get_module_mapping()
        return utils.get_trainable_module_names(module_mapping)

    def save_config(self, path: str) -> None:
        """Save model configuration to file."""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": self.device,
            "image_size": self.image_size,
            "max_objects": self.model_config.max_objects,
        }
        utils.save_model_config(config, path)
