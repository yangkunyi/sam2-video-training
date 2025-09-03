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
from dataclasses import dataclass
from torch import Tensor
from typing import List, Dict, Optional
import imageio
import sys


@logger.catch(onerror=lambda _: sys.exit(1))
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
    mask = (mask.squeeze(1) > 0).to(torch.uint8)  # [B, H, W] as 0/1

    # 预分配结果
    total_points = num_pos_points + num_neg_points
    points_all = torch.empty(B, total_points, 2, dtype=dtype, device=device)
    labels_all = torch.empty(B, total_points, dtype=torch.int32, device=device)

    for b in range(B):
        m = mask[b]  # [H, W]

        # ----------- 正样本 -----------
        pos_coords = torch.stack(torch.where(m == 1), dim=1)  # [N_pos, 2] (y, x)
        num_pos_available = pos_coords.shape[0]

        if num_pos_points > 0 and num_pos_available == 0:
            raise ValueError("generate_point_prompt: no positive pixels available for sampling")

        # 计算几何中心（仅在存在正样本时）
        if num_pos_available > 0:
            cy, cx = ndimage.center_of_mass(m.cpu().numpy())
            center = torch.tensor([cx, cy], dtype=dtype, device=device)
        else:
            center = torch.empty(2, dtype=dtype, device=device)

        # 采样正样本点
        if include_center and num_pos_points > 0:
            # 中心点作为第一个正样本
            pos_pts = [center.unsqueeze(0)]
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
            pos_pts = torch.cat(pos_pts, dim=0)  # 裁剪到指定数量
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


@logger.catch(onerror=lambda _: sys.exit(1))
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
        m = (mask[i, 0] > 0).cpu().numpy()  # [H, W]
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            raise ValueError("generate_box_prompt: no positive pixels to form a bounding box")
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


@logger.catch(onerror=lambda _: sys.exit(1))
def find_connected_components(mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Find connected components in a binary mask and return list of individual component masks.
    Uses morphological opening (erosion followed by dilation) to eliminate small regions.
    Args:
        mask: Binary mask tensor [H, W]
    Returns:
        List of individual component masks as tensors
    """
    # Convert to numpy for OpenCV operations
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Define kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply opening operation: erosion followed by dilation
    # This eliminates small regions and noise
    eroded_mask = cv2.erode(mask_np, kernel, iterations=1)
    opened_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

    # Use OpenCV's connectedComponents to find connected components
    num_components, labeled_mask = cv2.connectedComponents(opened_mask)

    connected_areas = []
    for component_id in range(1, num_components):  # Skip background (0)
        # Create binary mask for this component
        component_mask = (labeled_mask == component_id).astype(np.uint8)

        # Convert to PyTorch tensor and move to original device
        component_tensor = torch.from_numpy(component_mask.astype(np.float32)).to(
            mask.device
        )
        connected_areas.append(component_tensor)

    return connected_areas


@logger.catch(onerror=lambda _: sys.exit(1))
def cat_to_obj_mask(cat_frame_masks: torch.Tensor) -> Tuple[torch.Tensor, List[int], int]:
    """
    Splits a single mask into multiple masks for each object.

    Args:
        cat_frame_masks: torch.tensor [num_categories, 1, H, W]

    Returns:
        Tuple of (object_masks, obj_to_cat_mapping)
        - object_masks: torch.Tensor with individual object masks
        - obj_to_cat_mapping: Dict mapping object IDs to category IDs
    """
    # cat_frame_masks: [num_categories, B, H, W] with B=1 here
    N = int(cat_frame_masks.shape[0])
    obj_to_cat = []
    obj_masks_list = []
    for catergory_idx in range(N):
        category_mask = (cat_frame_masks[catergory_idx][0] > 0).to(torch.float32)
        if category_mask.sum() == 0:
            continue
        connected_areas = find_connected_components(category_mask)
        for area_mask in connected_areas:
            obj_masks_list.append(area_mask)
            obj_to_cat.append(catergory_idx)
    if not obj_masks_list:
        raise ValueError("cat_to_obj_mask: no objects found in category masks (fail-fast)")
    return torch.stack(obj_masks_list).unsqueeze(1), obj_to_cat, N


@logger.catch(onerror=lambda _: sys.exit(1))
def merge_object_results_to_category(
    previous_stages_out: List[Dict[str, Any]],
    obj_to_cat: List[int],
    num_categories: int,
) -> List[Dict[str, Any]]:
    """
    Merge per-object outputs back to per-category outputs for each frame.

    Rules
    - Mask-like tensors (logits) are merged by pixelwise max across objects in the same category
      to approximate logical OR at the probability level.
    - Score/IoU-like tensors are merged by weighted average using per-object mask area
      (sum of sigmoid(logits)) as weights. If weights sum to 0 for a category, fall back to mean.
    - Null (None) values are ignored/preserved as None.
    - Mask memory related items are ignored (not included in merged output).

    Args:
        previous_stages_out: List of per-frame dicts as produced by forward_tracking
        obj_to_cat: List mapping object index -> category index

    Returns:
        List of per-frame dicts aggregated to category level.
    """

    if not previous_stages_out:
        return []

    # Determine category groups from obj_to_cat (ensure contiguous categories 0..max)
    if len(obj_to_cat) == 0:
        return [
            {k: None for k in ("pred_masks", "pred_masks_high_res")}
            for _ in previous_stages_out
        ]
    # Categories are indexed by their id in obj_to_cat; use max id + 1
    cat_to_indices: List[List[int]] = [[] for _ in range(num_categories)]
    for obj_idx, cat_idx in enumerate(obj_to_cat):
        cat_to_indices[int(cat_idx)].append(int(obj_idx))

    def _area_weights_from_masks(mask_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-object weights from mask logits as probability mass area.

        mask_logits: [N, 1, H, W]
        returns: [N]
        """
        probs = torch.sigmoid(mask_logits)  # [N, 1, H, W]
        areas = probs.sum(dim=(1, 2, 3))  # [N]
        return areas

    def _grouped_max(
        tensor: torch.Tensor,  # [N, ...]
        groups: List[List[int]],
    ) -> torch.Tensor:
        """Pixelwise max across objects within the same category."""
        if tensor.numel() == 0:
            # Create an empty tensor with category dim if no objects
            return tensor.new_zeros((len(groups),) + tuple(tensor.shape[1:]))
        out_per_cat: List[torch.Tensor] = []
        for idxs in groups:
            if len(idxs) == 0:
                out_per_cat.append(tensor.new_zeros(tensor.shape[1:]))
            else:
                out_per_cat.append(tensor[idxs].max(dim=0).values)
        return torch.stack(out_per_cat, dim=0)

    def _grouped_weighted_avg(
        tensor: torch.Tensor,  # [N, ...]
        groups: List[List[int]],
        weights: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Weighted average across objects within the same category.

        Broadcasting: weights will be broadcast across remaining dims of `tensor`.
        """
        if tensor.numel() == 0:
            return tensor.new_zeros((len(groups),) + tuple(tensor.shape[1:]))
        # Prepare weights broadcast shape: [N, 1, 1, ...]
        w = weights.view(weights.shape[0], *([1] * (tensor.dim() - 1)))
        out_per_cat: List[torch.Tensor] = []
        for idxs in groups:
            if len(idxs) == 0:
                out_per_cat.append(tensor.new_zeros(tensor.shape[1:]))
                continue
            sub = tensor[idxs]
            sub_w = w[idxs]
            denom = sub_w.sum(dim=0)
            if torch.all(denom == 0):
                out_per_cat.append(sub.mean(dim=0))
            else:
                out_per_cat.append((sub * sub_w).sum(dim=0) / denom)
        return torch.stack(out_per_cat, dim=0)

    merged_per_frame: List[Dict[str, Any]] = []

    for frame_out in previous_stages_out:
        merged: Dict[str, Any] = {}

        # Determine weights from available masks
        if isinstance(frame_out.get("pred_masks_high_res"), torch.Tensor):
            weights_source = frame_out["pred_masks_high_res"]  # [N, 1, H, W]
        elif isinstance(frame_out.get("pred_masks"), torch.Tensor):
            weights_source = frame_out["pred_masks"]  # [N, 1, H, W]
        else:
            raise ValueError(
                "Weights source not found in frame outputs; expected 'pred_masks' or 'pred_masks_high_res'"
            )

        weights: torch.Tensor = _area_weights_from_masks(weights_source)

        # 1) Merge mask-like keys by pixelwise max (union)
        for key in (
            "pred_masks",
            "pred_masks_high_res",
            "multistep_pred_masks",
            "multistep_pred_masks_high_res",
        ):
            val = frame_out.get(key)
            if isinstance(val, torch.Tensor):  # [N, ...]
                merged[key] = _grouped_max(val, cat_to_indices)

        # 2) Merge multi-mask proposals per step by pixelwise max
        for key in ("multistep_pred_multimasks", "multistep_pred_multimasks_high_res"):
            val = frame_out.get(key)
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], torch.Tensor)
            ):
                step_out: List[torch.Tensor] = []
                for step_tensor in val:  # [N, K, H, W]
                    step_out.append(_grouped_max(step_tensor, cat_to_indices))
                merged[key] = step_out

        # 3) Merge IoUs and scores per step by weighted average
        for key in ("multistep_pred_ious", "multistep_object_score_logits"):
            val = frame_out.get(key)
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], torch.Tensor)
            ):
                step_out: List[torch.Tensor] = []
                for step_tensor in val:  # [N, K]
                    step_out.append(
                        _grouped_weighted_avg(step_tensor, cat_to_indices, weights)
                    )
                merged[key] = step_out

        merged["point_inputs"] = frame_out["point_inputs"]
        merged["mask_inputs"] = frame_out["mask_inputs"]
        if isinstance(frame_out.get("multistep_point_inputs"), list):
            merged["multistep_point_inputs"] = [
                None for _ in frame_out["multistep_point_inputs"]
            ]

        # Explicitly ignore mask memory related items
        # (maskmem_features, maskmem_pos_enc) are not included

        merged_per_frame.append(merged)

    return merged_per_frame


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


@logger.catch(onerror=lambda _: sys.exit(1))
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
        raise KeyError(f"Module '{module_name}' not found")


@logger.catch(onerror=lambda _: sys.exit(1))
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
        raise KeyError(f"Module '{module_name}' not found")


@logger.catch(onerror=lambda _: sys.exit(1))
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


def extract_shape_info(obj: Any) -> Any:
    """递归地将张量转换为可序列化的Python类型（列表/标量）"""
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)  # 将形状元组转为列表（JSON更友好）
    elif isinstance(obj, dict):
        return {k: extract_shape_info(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [extract_shape_info(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(extract_shape_info(item) for item in obj)
    else:
        return obj  # 基本类型（int, float, str等）直接返回


@logger.catch(onerror=lambda _: sys.exit(1))
def create_composite_visualization(
    frame_0: torch.Tensor,  # [C, H, W]
    image: torch.Tensor,  # [C, H, W]
    gt_mask: torch.Tensor,  # [C, H, W]
    pred_mask: torch.Tensor,  # [C, H, W]
    prompts: Dict[str, Any],
    obj_to_cat: List[int],
    num_categories: int = 13,
    title: str = "Visualization",
) -> np.ndarray:
    """Create composite visualization: Image | GT Mask | Prompts | Prediction."""
    import matplotlib.pyplot as plt
    import cv2
    import random
    import colorsys

    # Generate random colors for different categories
    def generate_random_colors(n):
        colors = []
        for i in range(n):
            # Generate vibrant colors using HSV color space
            hue = i / n
            saturation = 0.8 + random.random() * 0.2  # 0.8-1.0
            value = 0.8 + random.random() * 0.2  # 0.8-1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    category_colors = generate_random_colors(num_categories)

    # Convert tensors to numpy with proper denormalization
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().detach().numpy()
        # Denormalize if using ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)  # Ensure valid range
        # Convert to uint8 for display
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image

    if isinstance(frame_0, torch.Tensor):
        frame_0_np = frame_0.permute(1, 2, 0).cpu().detach().numpy()
        # Denormalize if using ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_0_np = frame_0_np * std + mean
        frame_0_np = np.clip(frame_0_np, 0, 1)  # Ensure valid range
        # Convert to uint8 for display
        if frame_0_np.max() <= 1.0:
            frame_0_np = (frame_0_np * 255).astype(np.uint8)
    else:
        frame_0_np = frame_0

    # Handle multi-channel masks - process each category separately
    if isinstance(gt_mask, torch.Tensor):
        if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
            gt_mask_np = gt_mask.cpu().detach().numpy()  # [C, H, W]
        else:
            gt_mask_np = gt_mask.squeeze().cpu().detach().numpy()
    else:
        gt_mask_np = gt_mask

    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.dim() == 3 and pred_mask.shape[0] > 1:
            pred_mask_np = pred_mask.cpu().detach().numpy()  # [C, H, W]
        else:
            pred_mask_np = pred_mask.squeeze().cpu().detach().numpy()
    else:
        pred_mask_np = pred_mask

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Original Image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Helper function to create alpha composite with contours
    def create_alpha_composite_with_contours(base_image, mask_array, colors, alpha=0.4):
        # Create a transparent overlay
        overlay = np.zeros_like(base_image, dtype=np.uint8)

        # Apply semi-transparent masks
        if mask_array.ndim == 3:  # Multi-category mask
            for category_idx in range(mask_array.shape[0]):
                mask = mask_array[category_idx] > 0.5
                if mask.any():
                    color = colors[category_idx % len(colors)]
                    for c in range(3):
                        overlay[:, :, c][mask] = color[c]
        else:  # Single category mask
            mask = mask_array > 0.5
            if mask.any():
                color = colors[0]
                for c in range(3):
                    overlay[:, :, c][mask] = color[c]

        # Create alpha composite
        composite = base_image.copy()
        # Alpha blending formula: composite = base * (1 - alpha) + overlay * alpha
        composite = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)

        # Add contours
        if mask_array.ndim == 3:  # Multi-category mask
            for category_idx in range(mask_array.shape[0]):
                mask = (mask_array[category_idx] > 0.5).astype(np.uint8)
                if mask.any():
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    color = colors[category_idx % len(colors)]
                    cv2.drawContours(composite, contours, -1, color, 2)
        else:  # Single category mask
            mask = (mask_array > 0.5).astype(np.uint8)
            if mask.any():
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                color = colors[0]
                cv2.drawContours(composite, contours, -1, color, 2)

        return composite

    # 2. Ground Truth Mask with alpha composite and contours
    gt_display = create_alpha_composite_with_contours(
        image_np, gt_mask_np, category_colors, alpha=0.2
    )
    axes[0, 1].imshow(gt_display)
    axes[0, 1].set_title("GT Mask (Semi-transparent)")
    axes[0, 1].axis("off")

    # 3. Prompts Overlay - visualize all prompts
    prompt_img = frame_0_np.copy()
    prompt_mask_np = np.zeros_like(gt_mask_np)

    if prompts["prompt_type"] == "point":
        for i in range(prompts["point_inputs"]["point_coords"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            points = prompts["point_inputs"]["point_coords"][i].cpu().detach().numpy()
            labels = prompts["point_inputs"]["point_labels"][i].cpu().detach().numpy()
            for idx, point in enumerate(points):
                label = labels[idx]
                if label == 1:  # 画点 (圆形标记)
                    cv2.circle(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        8,
                        color,
                        -1,
                    )
                    # 添加白色边框
                    cv2.circle(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        8,
                        (255, 255, 255),
                        2,
                    )
                elif label == 0:  # 画叉 (十字标记)
                    cv2.drawMarker(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        color,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=16,
                        thickness=3,
                        line_type=cv2.LINE_AA,
                    )
    elif prompts["prompt_type"] == "bbox":
        for i in range(prompts["point_inputs"]["point_labels"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            x1, y1, x2, y2 = 0, 0, 0, 0

            color = category_colors[obj_to_cat[i]]
            points = prompts["point_inputs"]["point_coords"][i].cpu().detach().numpy()
            labels = prompts["point_inputs"]["point_labels"][i].cpu().detach().numpy()
            for idx, point in enumerate(points):
                label = labels[idx]
                if label == 2:  # 画矩形 (矩形标记)
                    x1, y1 = point[0], point[1]
                if label == 3:
                    x2, y2 = point[0], point[1]

            # 画矩形
            cv2.rectangle(
                prompt_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=2,
            )

    elif prompts["prompt_type"] == "mask":
        for i in range(prompts["mask_inputs"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            mask_prompt = prompts["mask_inputs"][i].cpu().detach().numpy()
            # Create alpha composite for this mask
            prompt_mask_np[obj_to_cat[i]] = np.maximum(
                prompt_mask_np[obj_to_cat[i]], mask_prompt
            )

    if prompts["prompt_type"] == "mask":
        prompt_img = create_alpha_composite_with_contours(
            frame_0_np, prompt_mask_np, category_colors, alpha=0.2
        )
    axes[1, 0].imshow(prompt_img)
    axes[1, 0].set_title("Prompts (All Objects)")
    axes[1, 0].axis("off")

    # 4. Prediction Overlay with alpha composite and contours
    pred_display = create_alpha_composite_with_contours(
        image_np, pred_mask_np, category_colors, alpha=0.2
    )
    axes[1, 1].imshow(pred_display)
    axes[1, 1].set_title("Prediction (Semi-transparent)")
    axes[1, 1].axis("off")

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() for newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    composite = np.asarray(buf)
    # Convert from RGBA to RGB by dropping alpha channel
    composite = composite[:, :, :3]
    plt.close(fig)
    return composite


@logger.catch(onerror=lambda _: sys.exit(1))
def create_visualization_gif(
    frames: torch.Tensor,
    gt_masks: torch.Tensor,
    outs_per_frame: List[Dict[str, torch.Tensor]],
    obj_to_cat: List[int],
    max_length: int = 4,
    stride: int = 1,
) -> str:
    """Create visualization GIF and return path to GIF file in /tmp."""

    # Create list to store frames for GIF
    gif_frames = []

    # [T, C, H, W]
    predictions = torch.stack(
        [out["multistep_pred_multimasks_high_res"][0] for out in outs_per_frame]
    ).squeeze(2)
    num_categories = predictions.shape[1]
    point_prompts = outs_per_frame[0]["point_inputs"]
    mask_prompts = outs_per_frame[0]["mask_inputs"]

    if point_prompts is not None:
        unique_values = point_prompts["point_labels"].unique()
        if 2 in unique_values or 3 in unique_values:
            prompt_type = "bbox"
        else:
            prompt_type = "point"
    else:
        prompt_type = "mask"

    # Process frames with stride
    length = min(max_length, frames.shape[0], predictions.shape[0])
    indices = range(0, length, stride)

    prompts = {
        "point_inputs": point_prompts if point_prompts is not None else None,
        "mask_inputs": mask_prompts if mask_prompts is not None else None,
        "prompt_type": prompt_type,
    }

    for i in indices:
        frame = frames[i]  # [C, H, W]
        gt_mask = gt_masks[i]
        pred_mask = predictions[i]

        # Create composite visualization
        composite = create_composite_visualization(
            frame_0=frames[0],
            image=frame,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            prompts=prompts,
            title=f"Frame {i}",
            obj_to_cat=obj_to_cat,
            num_categories=num_categories,
        )

        # Add frame to GIF list
        gif_frames.append(composite)

    # Create GIF from frames
    if gif_frames:
        # Create temporary file in /tmp directory
        import tempfile

        gif_filename = tempfile.mktemp(
            suffix=".gif", prefix="visualization_", dir="/tmp"
        )

        # Save as GIF
        with imageio.get_writer(gif_filename, mode="I", duration=0.5, loop=0) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

        return gif_filename

    return ""
