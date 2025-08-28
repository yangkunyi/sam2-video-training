"""
Unified SAM2 model module for loading and video tracking.
This module combines model loading and tracking functionality for simplicity.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import yaml
from loguru import logger
from beartype import beartype

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base


@beartype
class SAM2Model(nn.Module):
    """
    Unified SAM2 model class that handles both loading and tracking functionality.
    This class combines the functionality of SimpleSAM2ModelLoader and SimpleSAM2Tracker.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        config_path: str = "",
        trainable_modules: List[str] = None,
        device: str = "cuda",
        image_size: int = 512,
        num_maskmem: int = 7,
        **kwargs,
    ):
        """
        Initialize SAM2 model with configuration.

        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config_path: Path to SAM2 config file
            trainable_modules: List of module names to train
            device: Device to load model on
            image_size: Image size for processing
            num_maskmem: Number of mask memories
            **kwargs: Additional model parameters
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.trainable_modules = trainable_modules or [
            "memory_attention",
            "memory_encoder",
        ]
        self.device = device

        # Model parameters
        self.image_size = image_size
        self.num_maskmem = num_maskmem
        self.image_encoder = None
        self.memory_attention = None
        self.memory_encoder = None
        self.max_objects = kwargs.get("max_objects", 10)  # Maximum number of objects to track

        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate paths
        if checkpoint_path:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = None

    def load(self, device: str = None) -> "SAM2Model":
        """
        Load SAM2 model and set up for tracking.

        Args:
            device: Device to load model on (overwrites initialization device if provided)

        Returns:
            Self for method chaining
        """
        if SAM2Base is None:
            raise ImportError("SAM2 is not available. Please install SAM2.")

        device = device or self.device
        logger.info("Loading SAM2 model...")

        # Load the base SAM2 model
        base_model = build_sam2(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            mode="eval",
        )

        # Handle wrapped models
        if hasattr(base_model, "sam_model_instance"):
            base_model = base_model.sam_model_instance

        # Extract components
        self.image_encoder = base_model.image_encoder
        self.memory_attention = base_model.memory_attention
        self.memory_encoder = base_model.memory_encoder

        # Set up SAM2 base functionality
        self._setup_sam2_base(base_model)

        # Move to device
        self.to(torch.device(device))
        logger.info(f"Model loaded and moved to {device}")

        # Configure for selective training
        if self.trainable_modules:
            self.configure_for_training(self.trainable_modules)

        return self

    def _setup_sam2_base(self, base_model: SAM2Base) -> None:
        """Copy SAM2 base functionality to this model."""
        # Copy all essential methods and attributes from base_model
        for attr_name in dir(base_model):
            if not attr_name.startswith("_") and hasattr(self, "__dict__"):
                attr_value = getattr(base_model, attr_name)
                if not callable(attr_value) or attr_name == "forward_image":
                    setattr(self, attr_name, attr_value)

    def configure_for_training(self, trainable_modules: List[str]) -> None:
        """
        Configure model for selective module training.

        Args:
            trainable_modules: List of module names to train
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.train()
        self.freeze_all_parameters()

        # Unfreeze specified modules
        modules_found = False
        for module_name in trainable_modules:
            if self._unfreeze_module(module_name):
                modules_found = True

        if not modules_found:
            logger.warning(f"No modules found: {trainable_modules}")

        # Report training configuration
        trainable_count = self.count_trainable_parameters()
        total_count = self.count_total_parameters()

        logger.info(f"Training configuration:")
        logger.info(f"  Trainable modules: {trainable_modules}")
        logger.info(f"  Trainable params: {trainable_count:,}")
        logger.info(f"  Total params: {total_count:,}")
        logger.info(f"  Percentage: {trainable_count / total_count * 100:.2f}%")

    def freeze_all_parameters(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module_name: str) -> bool:
        """Unfreeze a specific module by name."""
        found = False

        if hasattr(self, module_name):
            module = getattr(self, module_name)
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen: {module_name}")
                found = True

        # Search recursively in submodules
        for name, child in self.named_children():
            if name == module_name:
                for param in child.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen: {name}")
                found = True
            elif isinstance(child, nn.Module):
                if self._find_and_unfreeze_recursive(child, module_name):
                    found = True

        return found

    def _find_and_unfreeze_recursive(self, module: nn.Module, target_name: str) -> bool:
        """Recursively find and unfreeze module."""
        found = False
        for name, child in module.named_children():
            if name == target_name:
                for param in child.parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen: {name}")
                found = True
            elif isinstance(child, nn.Module):
                if self._find_and_unfreeze_recursive(child, target_name):
                    found = True
        return found

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor | Dict]:
        """
        Forward pass for video tracking with multiple objects.

        Args:
            images: Video frames [B, T, C, H, W]
            masks: Optional ground truth masks [B, T, N, H, W] where N is number of objects
            prompts: Optional prompts for first frame for multiple objects

        Returns:
            Dictionary with predicted masks
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        B, T, C, H, W = images.shape
        num_objects = masks.shape[2] if masks is not None else 1

        # Create tracking output structure
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        # Process frames sequentially
        all_pred_masks = []
        all_pred_masks_high_res = []

        for frame_idx in range(T):
            # Get current frame image
            frame_image = images[:, frame_idx]  # [B, C, H, W]

            # Determine if this is a conditioning frame (first frame only)
            is_init_cond = frame_idx == 0

            # Prepare prompts for this frame (handle multiple objects)
            point_inputs_list, mask_inputs_list = self._prepare_prompts_multi_object(
                prompts,
                masks[:, frame_idx] if masks is not None else None,
                is_init_cond,
                num_objects,
            )

            # Process single frame for multiple objects
            if hasattr(self, "track_step"):
                frame_output = self._track_with_memory_multi_object(
                    frame_image,
                    frame_idx,
                    output_dict,
                    point_inputs_list,
                    mask_inputs_list,
                    is_init_cond,
                    T,
                    num_objects,
                )
            else:
                logger.warning(
                    "track_step not found. Falling back to simple forward pass."
                )
                frame_output = self._simple_forward_multi_object(frame_image, masks, frame_idx, num_objects)

            # Store frame output
            if is_init_cond:
                output_dict["cond_frame_outputs"][frame_idx] = frame_output
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = frame_output

            # Collect predictions
            pred_masks = frame_output.get(
                "pred_masks", torch.zeros(B, num_objects, H // 4, W // 4)
            )
            pred_masks_high_res = frame_output.get(
                "pred_masks_high_res", torch.zeros(B, num_objects, H, W)
            )

            all_pred_masks.append(pred_masks)
            all_pred_masks_high_res.append(pred_masks_high_res)

        # Stack predictions across time
        pred_masks = torch.stack(all_pred_masks, dim=1)  # [B, T, N, H//4, W//4]
        pred_masks_high_res = torch.stack(
            all_pred_masks_high_res, dim=1
        )  # [B, T, N, H, W]

        return {
            "pred_masks": pred_masks,
            "pred_masks_high_res": pred_masks_high_res,
            "output_dict": output_dict,
        }

    def _prepare_prompts_multi_object(
        self,
        prompts: Optional[List[Dict[str, Any]]],
        gt_masks: Optional[torch.Tensor],
        is_init_cond: bool,
        num_objects: int,
    ) -> Tuple[List[Optional[Dict]], List[Optional[torch.Tensor]]]:
        """Prepare prompts for frame processing for multiple objects."""
        if not is_init_cond:
            return [None] * num_objects, [None] * num_objects

        point_inputs_list = []
        mask_inputs_list = []

        # For first frame, use provided prompts or GT masks for each object
        if prompts is not None and len(prompts) > 0:
            # Handle point prompts for each object
            for obj_idx in range(min(num_objects, len(prompts))):
                obj_prompt = prompts[obj_idx]
                if isinstance(obj_prompt, dict) and "point_coords" in obj_prompt:
                    point_inputs = {
                        "point_coords": obj_prompt["point_coords"],
                        "point_labels": obj_prompt.get(
                            "point_labels", torch.ones(obj_prompt["point_coords"].shape[:-1])
                        ),
                    }
                    point_inputs_list.append(point_inputs)
                    mask_inputs_list.append(None)
                else:
                    point_inputs_list.append(None)
                    mask_inputs_list.append(None)
            
            # Pad if needed
            while len(point_inputs_list) < num_objects:
                point_inputs_list.append(None)
                mask_inputs_list.append(None)
        elif gt_masks is not None:
            # Use GT masks for each object
            for obj_idx in range(num_objects):
                mask_inputs_list.append(gt_masks[:, obj_idx] if gt_masks.dim() > 3 else gt_masks)
                point_inputs_list.append(None)
        else:
            # No prompts or masks
            point_inputs_list = [None] * num_objects
            mask_inputs_list = [None] * num_objects

        return point_inputs_list, mask_inputs_list

    def _track_with_memory_multi_object(
        self,
        frame_image: torch.Tensor,
        frame_idx: int,
        output_dict: Dict,
        point_inputs_list: List[Optional[Dict]],
        mask_inputs_list: List[Optional[torch.Tensor]],
        is_init_cond: bool,
        num_frames: int,
        num_objects: int,
    ) -> Dict:
        """Track frame with memory using SAM2 track_step method for multiple objects."""
        if not hasattr(self, "forward_image"):
            return self._simple_forward_multi_object(frame_image, mask_inputs_list[0] if mask_inputs_list else None, frame_idx, num_objects)

        # Extract image features
        backbone_out = self.forward_image(frame_image)
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Process each object separately (this is a simplified approach)
        # In a full implementation, objects would be tracked jointly
        all_outputs = []
        for obj_idx in range(num_objects):
            point_inputs = point_inputs_list[obj_idx] if obj_idx < len(point_inputs_list) else None
            mask_inputs = mask_inputs_list[obj_idx] if obj_idx < len(mask_inputs_list) else None
            
            # Process with memory for each object
            obj_output = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=num_frames,
                run_mem_encoder=True,
            )
            all_outputs.append(obj_output)
        
        # Combine outputs (simplified - in practice would need more sophisticated merging)
        if all_outputs:
            combined_output = all_outputs[0].copy()
            if "pred_masks" in combined_output and len(all_outputs) > 1:
                pred_masks = [out.get("pred_masks", torch.zeros_like(combined_output["pred_masks"])) for out in all_outputs]
                combined_output["pred_masks"] = torch.stack(pred_masks, dim=1)  # [B, N, H//4, W//4]
            if "pred_masks_high_res" in combined_output and len(all_outputs) > 1:
                pred_masks_high_res = [out.get("pred_masks_high_res", torch.zeros_like(combined_output["pred_masks_high_res"])) for out in all_outputs]
                combined_output["pred_masks_high_res"] = torch.stack(pred_masks_high_res, dim=1)  # [B, N, H, W]
            return combined_output
        else:
            return self._simple_forward_multi_object(frame_image, mask_inputs_list[0] if mask_inputs_list else None, frame_idx, num_objects)

    def _simple_forward_multi_object(
        self, frame_image: torch.Tensor, masks: Optional[torch.Tensor], frame_idx: int, num_objects: int
    ) -> Dict:
        """Simple forward pass for testing without full SAM2 for multiple objects."""
        batch_size = frame_image.shape[0]
        h, w = frame_image.shape[2:]

        # Generate dummy predictions for multiple objects
        pred_masks = torch.zeros(batch_size, num_objects, h // 4, w // 4)
        pred_masks_high_res = torch.zeros(batch_size, num_objects, h, w)

        return {
            "pred_masks": pred_masks,
            "pred_masks_high_res": pred_masks_high_res,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()

        return {
            "model_type": "SAM2Model",
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": str(next(self.parameters()).device)
            if list(self.parameters())
            else "not_loaded",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_params / total_params * 100
            if total_params > 0
            else 0,
            "trainable_modules": self.trainable_modules,
            "image_size": self.image_size,
            "num_maskmem": self.num_maskmem,
        }

    def save_config(self, path: str) -> None:
        """Save model configuration to file."""
        config_data = self.get_info()
        with open(path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        logger.info(f"Model config saved to {path}")


def create_sam2_model(
    checkpoint_path: str,
    config_path: str,
    trainable_modules: List[str] = None,
    device: str = "cuda",
    **kwargs,
) -> SAM2Model:
    """
    Factory function to create SAM2 model.

    Args:
        checkpoint_path: Path to SAM2 checkpoint
        config_path: Path to SAM2 config file
        trainable_modules: List of modules to train
        device: Device to load model on
        **kwargs: Additional model parameters

    Returns:
        Loaded SAM2 model
    """
    model = SAM2Model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        trainable_modules=trainable_modules,
        device=device,
        **kwargs,
    )
    return model.load()
