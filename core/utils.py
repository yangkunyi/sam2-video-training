# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for SAM2 training, including prompt generation and mask processing.

This module contains utility functions extracted from sam2train.py to improve
code organization and reusability.
"""

import numpy as np
import torch
import cv2
import yaml
from scipy import ndimage
from typing import Dict, Tuple, List, Any
from torch import nn
from loguru import logger


def generate_point_prompt(
    mask: torch.Tensor,
    num_pos_points: int = 1,
    num_neg_points: int = 0,
    include_center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从二值 mask 中采样 prompt 点。

    参数
    ----
    mask : Tensor
        形状 [B, 1, H, W]，前景为 1，背景为 0。
    num_pos_points : int
        需要采样的正样本（前景）点数量。
    num_neg_points : int
        需要采样的负样本（背景）点数量。
    include_center : bool
        是否在正样本点中加入 mask 的几何中心点。如果 `True`，
        则总正样本点数仍为 `num_pos_points`（中心点占一个位置）。

    返回
    ----
    points : Tensor
        形状 [B, num_pos_points + num_neg_points, 2]，坐标格式 (x, y)。
    labels : Tensor
        形状 [B, num_pos_points + num_neg_points]，1 表示正样本，0 表示负样本。
    """
    B, _, H, W = mask.shape
    device = mask.device
    dtype = torch.float32

    # 去掉 channel 维，便于后续处理
    mask = mask.squeeze(1)  # [B, H, W]

    # 预分配结果
    total_points = num_pos_points + num_neg_points
    points_all = torch.empty(B, total_points, 2, dtype=dtype, device=device)
    labels_all = torch.empty(B, total_points, dtype=torch.int32, device=device)

    for b in range(B):
        m = mask[b]  # [H, W]

        # ----------- 正样本 -----------
        pos_coords = torch.stack(torch.where(m == 1), dim=1)  # [N_pos, 2] (y, x)
        num_pos_available = pos_coords.shape[0]

        # 计算几何中心
        cy, cx = ndimage.center_of_mass(m.cpu().numpy())
        center = torch.tensor([cx, cy], dtype=dtype, device=device)

        # 采样正样本点
        if include_center and num_pos_points > 0:
            # 中心点作为第一个正样本
            pos_pts = [center]
            need_extra = max(0, num_pos_points - 1)
        else:
            pos_pts = []
            need_extra = num_pos_points

        # 随机采样其余正样本
        if need_extra > 0:
            idx = torch.randperm(num_pos_available, device=device)[:need_extra]
            sampled = pos_coords[idx].flip(-1).to(dtype)  # (x, y)
            pos_pts.append(sampled)

        # 拼成正样本张量
        if num_pos_points > 0:
            pos_pts = torch.cat(pos_pts, 0)[:num_pos_points]  # 裁剪到指定数量
        else:
            pos_pts = torch.empty(0, 2, dtype=dtype, device=device)

        # ----------- 负样本 -----------
        neg_coords = torch.stack(torch.where(m == 0), dim=1)  # [N_neg, 2] (y, x)
        num_neg_available = neg_coords.shape[0]
        if num_neg_points > 0 and num_neg_available > 0:
            idx = torch.randperm(num_neg_available, device=device)[:num_neg_points]
            neg_pts = neg_coords[idx].flip(-1).to(dtype)  # (x, y)
        else:
            neg_pts = torch.empty(0, 2, dtype=dtype, device=device)

        # ----------- 合并 -----------
        pts = torch.cat([pos_pts, neg_pts], 0)
        lbls = torch.cat(
            [
                torch.ones(pos_pts.shape[0], dtype=torch.int32, device=device),
                torch.zeros(neg_pts.shape[0], dtype=torch.int32, device=device),
            ]
        )

        points_all[b] = pts
        labels_all[b] = lbls

    return points_all, labels_all


def generate_box_prompt(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据二值 mask 自动生成一个方框提示（box prompt）。

    参数
    ----
    mask : torch.Tensor
        形状 [B, 1, H, W]，前景为 1，背景为 0。

    返回
    ----
    points : torch.Tensor
        形状 [B, 2, 2]，表示每个样本的左上角与右下角坐标，格式为 (x, y)。
    labels : torch.Tensor
        形状 [B, 2]，两个点分别标记为 2（左上角）和 3（右下角）。
    """
    B, _, H, W = mask.shape

    points = torch.empty((B, 2, 2), dtype=torch.float32, device=mask.device)
    labels = torch.empty((B, 2), dtype=torch.int32, device=mask.device)

    for i in range(B):
        m = mask[i, 0].cpu().numpy()  # [H, W]
        ys, xs = np.where(m > 0)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # 左上角
        points[i, 0, 0] = float(x_min)
        points[i, 0, 1] = float(y_min)
        labels[i, 0] = 2  # 左上角标签

        # 右下角
        points[i, 1, 0] = float(x_max)
        points[i, 1, 1] = float(y_max)
        labels[i, 1] = 3  # 右下角标签

    return points, labels


def find_connected_components(mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Find connected components in a binary mask and return list of individual component masks.
    Uses OpenCV's erosion and dilation operations to connect small regions.

    Args:
        mask: Binary mask tensor [H, W]

    Returns:
        List of individual component masks as tensors
    """
    # Convert to numpy for OpenCV operations
    mask_np = mask.cpu().numpy().astype(np.uint8)
    # Define kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply dilation to connect small regions
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    # Use OpenCV's connectedComponents to find connected components
    num_components, labeled_mask = cv2.connectedComponents(dilated_mask)
    connected_areas = []
    for component_id in range(1, num_components):  # Skip background (0)
        # Create binary mask for this component
        component_mask = (labeled_mask == component_id).astype(np.uint8)
        # Apply erosion to restore original size (optional)
        component_mask = cv2.erode(component_mask, kernel, iterations=1)
        # Convert to PyTorch tensor and move to original device
        component_tensor = torch.from_numpy(component_mask.astype(np.float32)).to(
            mask.device
        )
        connected_areas.append(component_tensor)

    return connected_areas


def cat_to_obj_mask(cat_frame_masks) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Splits a single mask into multiple masks for each object.

    Args:
        cat_frame_masks: torch.tensor [num_categories, 1, H, W]

    Returns:
        Tuple of (object_masks, obj_to_cat_mapping)
        - object_masks: torch.Tensor with individual object masks
        - obj_to_cat_mapping: Dict mapping object IDs to category IDs
    """
    N = cat_frame_masks.shape[0]
    obj_id = 0
    obj_to_cat = {}
    obj_masks_list = []
    for catergory_idx in range(N):
        category_mask = cat_frame_masks[catergory_idx][0]
        if category_mask.sum() == 0:
            continue
        connected_areas = find_connected_components(category_mask)
        for area_mask in connected_areas:
            obj_masks_list.append(area_mask)
            obj_to_cat[obj_id] = catergory_idx
            obj_id += 1
    return torch.stack(obj_masks_list).unsqueeze(1), obj_to_cat


# Model Parameter Utilities
def count_trainable_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_info(
    model: nn.Module,
    checkpoint_path: str,
    config_path: str,
    device: str,
    image_size: int,
) -> Dict[str, Any]:
    """
    Get comprehensive model information including parameter counts.

    Args:
        model: PyTorch model
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration
        device: Device the model is on
        image_size: Input image size

    Returns:
        Dictionary with model information
    """
    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "device": device,
        "image_size": image_size,
    }


# Module Management Utilities
def setup_trainable_modules(
    model: nn.Module, module_mapping: Dict[str, nn.Module], trainable_modules: List[str]
) -> None:
    """
    Freeze all modules except those in trainable_modules list.

    Args:
        model: PyTorch model
        module_mapping: Dictionary mapping module names to modules
        trainable_modules: List of module names that should remain trainable
    """
    all_modules = list(module_mapping.keys())

    for module_name in all_modules:
        module = module_mapping.get(module_name)
        if module is not None:
            is_trainable = module_name in trainable_modules
            for param in module.parameters():
                param.requires_grad = is_trainable

            logger.info(
                f"Module '{module_name}': {'trainable' if is_trainable else 'frozen'}"
            )


def freeze_module_by_name(
    module_mapping: Dict[str, nn.Module], module_name: str
) -> None:
    """
    Dynamically freeze a specific module.

    Args:
        module_mapping: Dictionary mapping module names to modules
        module_name: Name of the module to freeze
    """
    module = module_mapping.get(module_name)
    if module is not None:
        for param in module.parameters():
            param.requires_grad = False
        logger.info(f"Module '{module_name}' frozen")
    else:
        logger.warning(f"Module '{module_name}' not found")


def unfreeze_module_by_name(
    module_mapping: Dict[str, nn.Module], module_name: str
) -> None:
    """
    Dynamically unfreeze a specific module.

    Args:
        module_mapping: Dictionary mapping module names to modules
        module_name: Name of the module to unfreeze
    """
    module = module_mapping.get(module_name)
    if module is not None:
        for param in module.parameters():
            param.requires_grad = True
        logger.info(f"Module '{module_name}' unfrozen")
    else:
        logger.warning(f"Module '{module_name}' not found")


def get_trainable_module_names(module_mapping: Dict[str, nn.Module]) -> List[str]:
    """
    Return list of currently trainable module names.

    Args:
        module_mapping: Dictionary mapping module names to modules

    Returns:
        List of module names that have trainable parameters
    """
    trainable_modules = []
    for module_name, module in module_mapping.items():
        if module is not None:
            # Check if any parameter in the module is trainable
            if any(param.requires_grad for param in module.parameters()):
                trainable_modules.append(module_name)

    return trainable_modules


# Configuration Utilities
def save_model_config(config_dict: Dict[str, Any], path: str) -> None:
    """
    Save model configuration to YAML file.

    Args:
        config_dict: Configuration dictionary to save
        path: Output file path
    """
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


# @logger.catch
# def create_composite_visualization(
#     image: torch.Tensor,  # [C, H, W]
#     gt_mask: torch.Tensor,  # [C, H, W]
#     pred_mask: torch.Tensor,  # [C, H, W]
#     prompts: List[List[PromptData]],
#     title: str = "Training Sample",
#     num_categories: int = 13,
# ) -> np.ndarray:
#     """Create composite visualization: Image | GT Mask | Prompts | Prediction."""
#     import matplotlib.pyplot as plt
#     import cv2
#     import random
#     import colorsys

#     # Generate random colors for different categories
#     def generate_random_colors(n):
#         colors = []
#         for i in range(n):
#             # Generate vibrant colors using HSV color space
#             hue = i / n
#             saturation = 0.8 + random.random() * 0.2  # 0.8-1.0
#             value = 0.8 + random.random() * 0.2  # 0.8-1.0
#             r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
#             colors.append((int(r * 255), int(g * 255), int(b * 255)))
#         return colors

#     category_colors = generate_random_colors(num_categories)

#     # Flatten nested prompt structure to get all prompts
#     all_prompts = prompts[0]
#     # Convert tensors to numpy with proper denormalization
#     if isinstance(image, torch.Tensor):
#         image_np = image.permute(1, 2, 0).cpu().detach().numpy()
#         # Denormalize if using ImageNet normalization
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         image_np = image_np * std + mean
#         image_np = np.clip(image_np, 0, 1)  # Ensure valid range
#         # Convert to uint8 for display
#         if image_np.max() <= 1.0:
#             image_np = (image_np * 255).astype(np.uint8)
#     else:
#         image_np = image
#     # Handle multi-channel masks - process each category separately
#     if isinstance(gt_mask, torch.Tensor):
#         if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
#             gt_mask_np = gt_mask.cpu().detach().numpy()  # [C, H, W]
#         else:
#             gt_mask_np = gt_mask.squeeze().cpu().detach().numpy()
#     else:
#         gt_mask_np = gt_mask
#     if isinstance(pred_mask, torch.Tensor):
#         if pred_mask.dim() == 3 and pred_mask.shape[0] > 1:
#             pred_mask_np = pred_mask.cpu().detach().numpy()  # [C, H, W]
#         else:
#             pred_mask_np = pred_mask.squeeze().cpu().detach().numpy()
#     else:
#         pred_mask_np = pred_mask
#     # Create 2x2 subplot
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#     fig.suptitle(title, fontsize=16)
#     # 1. Original Image
#     axes[0, 0].imshow(image_np)
#     axes[0, 0].set_title("Original Image")
#     axes[0, 0].axis("off")

#     # Helper function to create alpha composite with contours
#     def create_alpha_composite_with_contours(base_image, mask_array, colors, alpha=0.4):
#         # Create a transparent overlay
#         overlay = np.zeros_like(base_image, dtype=np.uint8)

#         # Apply semi-transparent masks
#         if mask_array.ndim == 3:  # Multi-category mask
#             for category_idx in range(mask_array.shape[0]):
#                 mask = mask_array[category_idx] > 0.5
#                 if mask.any():
#                     color = colors[category_idx % len(colors)]
#                     for c in range(3):
#                         overlay[:, :, c][mask] = color[c]
#         else:  # Single category mask
#             mask = mask_array > 0.5
#             if mask.any():
#                 color = colors[0]
#                 for c in range(3):
#                     overlay[:, :, c][mask] = color[c]

#         # Create alpha composite
#         composite = base_image.copy()
#         # Alpha blending formula: composite = base * (1 - alpha) + overlay * alpha
#         composite = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)

#         # Add contours
#         if mask_array.ndim == 3:  # Multi-category mask
#             for category_idx in range(mask_array.shape[0]):
#                 mask = (mask_array[category_idx] > 0.5).astype(np.uint8)
#                 if mask.any():
#                     contours, _ = cv2.findContours(
#                         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                     )
#                     color = colors[category_idx % len(colors)]
#                     cv2.drawContours(composite, contours, -1, color, 2)
#         else:  # Single category mask
#             mask = (mask_array > 0.5).astype(np.uint8)
#             if mask.any():
#                 contours, _ = cv2.findContours(
#                     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                 )
#                 color = colors[0]
#                 cv2.drawContours(composite, contours, -1, color, 2)

#         return composite

#     # 2. Ground Truth Mask with alpha composite and contours
#     gt_display = create_alpha_composite_with_contours(
#         image_np, gt_mask_np, category_colors, alpha=0.2
#     )
#     axes[0, 1].imshow(gt_display)
#     axes[0, 1].set_title("GT Mask (Semi-transparent)")
#     axes[0, 1].axis("off")

#     # 3. Prompts Overlay - visualize all prompts
#     prompt_img = image_np.copy()
#     prompt_mask_np = np.zeros_like(gt_mask_np)

#     for prompt_data in all_prompts:
#         if not hasattr(prompt_data, "obj_id"):
#             continue

#         # Get category based on obj_id
#         category_id = prompt_data.obj_id % num_categories
#         color = category_colors[category_id % len(category_colors)]
#         prompt_type = prompt_data.prompt_type
#         if prompt_type == "point" and prompt_data.points is not None:
#             points = prompt_data.points.cpu().detach().numpy()
#             labels = (
#                 prompt_data.labels.cpu().detach().numpy()
#                 if prompt_data.labels is not None
#                 else None
#             )
#             for idx, point in enumerate(points):
#                 if labels is not None:
#                     # Use darker/lighter variants for positive/negative
#                     point_color = tuple(
#                         int(c * 0.8) if labels[idx] == 0 else c for c in color
#                     )
#                 else:
#                     point_color = color
#                 cv2.circle(
#                     prompt_img, (int(point[0]), int(point[1])), 8, point_color, -1
#                 )
#                 # Add white border for better visibility
#                 cv2.circle(
#                     prompt_img, (int(point[0]), int(point[1])), 8, (255, 255, 255), 2
#                 )

#         elif prompt_type == "bbox" and prompt_data.bbox is not None:
#             bbox = prompt_data.bbox.cpu().detach().numpy()
#             x1, y1, x2, y2 = map(int, bbox)
#             cv2.rectangle(prompt_img, (x1, y1), (x2, y2), color, 3)

#         elif prompt_type == "mask" and prompt_data.mask is not None:
#             mask_prompt = prompt_data.mask.cpu().detach().numpy()
#             # Create alpha composite for this mask
#             prompt_mask_np[category_id] = np.maximum(
#                 prompt_mask_np[category_id], mask_prompt
#             )

#     if prompt_type == "mask":
#         prompt_img = create_alpha_composite_with_contours(
#             image_np, prompt_mask_np, category_colors, alpha=0.2
#         )
#     axes[1, 0].imshow(prompt_img)
#     axes[1, 0].set_title("Prompts (All Objects)")
#     axes[1, 0].axis("off")

#     # 4. Prediction Overlay with alpha composite and contours
#     pred_display = create_alpha_composite_with_contours(
#         image_np, pred_mask_np, category_colors, alpha=0.2
#     )
#     axes[1, 1].imshow(pred_display)
#     axes[1, 1].set_title("Prediction (Semi-transparent)")
#     axes[1, 1].axis("off")

#     # Convert matplotlib figure to numpy array
#     fig.canvas.draw()
#     # Use buffer_rgba() for newer matplotlib versions
#     buf = fig.canvas.buffer_rgba()
#     composite = np.asarray(buf)
#     # Convert from RGBA to RGB by dropping alpha channel
#     composite = composite[:, :, :3]
#     plt.close(fig)
#     return composite


# @logger.catch
# def log_training_visualizations(  # type: ignore
#     logger,
#     batch_data: Dict,
#     predictions: torch.Tensor,
#     batch_idx: int,
#     stage: str = "train",
#     max_samples: int = 4,
#     num_categories: int = 13,
# ) -> None:
#     """Log training visualizations to SwanLab."""
#     if logger is None:
#         return
#     frames = batch_data["frames"]
#     gt_masks = batch_data["gt_masks"]
#     prompts = batch_data["prompts"]
#     # Handle batch dimension

#     if frames.dim() == 5:
#         frames = frames[0]  # [T, C, H, W]
#         gt_masks = gt_masks[0]  # [T, num_categories, H, W] or [T, H, W]
#         if isinstance(prompts, list):
#             prompts = prompts[0]

#     # Log first few frames
#     num_samples = min(max_samples, frames.shape[0], predictions.shape[0])
#     for i in range(num_samples):
#         frame = frames[i]  # [C, H, W]
#         gt_mask = (
#             gt_masks[i] if gt_masks.dim() > 2 else gt_masks
#         )  # Handle different mask shapes
#         pred_mask = predictions[i] if predictions.dim() > 2 else predictions
#         # Create composite visualization
#         composite = create_composite_visualization(
#             frame,
#             gt_mask,
#             pred_mask,
#             prompts,
#             title=f"{stage.capitalize()} - Batch {batch_idx}, Frame {i}",
#             num_categories=num_categories,
#         )
#         # Log to SwanLab
#         logger.log_image(
#             key=f"{stage}_visualization/batch_{batch_idx}_frame_{i}",
#             images=[composite],
#             step=logger.global_step if hasattr(logger, "global_step") else None,
#         )
