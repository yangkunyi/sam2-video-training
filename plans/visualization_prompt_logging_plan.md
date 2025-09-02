# Plan: Visualization & Prompt Merging Integration

## Scope Understanding
- Goal: align visualization with current batch/forward outputs, merge prompts to category level, and log during train/val using `outs_per_frame` directly.
- Constraints: minimal, surgical changes; respect existing structures; no speculative features (KISS, YAGNI, DRY).

## Plan

### 1) Inspect Data Flow
- Identify shapes/types of `batch_data["frames"]`, `batch_data["gt_masks"]`, and `batch_data["prompts"]`.
- Identify the structure and keys of `outs_per_frame` from `core/trainer.py::forward`.
- Verify expected inputs for visualization: `image [C,H,W]`, `mask [num_categories,H,W]` or `[H,W]`, and prompts as list of `PromptData`.
- Confirm logger instance availability and interface for `logger.log_image`.

### 2) Normalize Predictions
- Convert `outs_per_frame` predictions into category mask layout `[num_categories,H,W]` or binary `[H,W]` consistent with GT mask.
- Handle probability/logit vs hard masks minimally: threshold sigmoid/softmax or `argmax` only when necessary for visualization output.

### 3) Align `create_composite_visualization` to Current Flow
- Remove any hard-coded assumptions (e.g., `prompts[0]`), supporting both batched and per-frame prompt inputs.
- Keep inputs minimal and explicit: single frame, GT mask, Pred mask, Prompts, `num_categories`.
- Robust denormalization for images only if range ≤ 1 (ImageNet mean/std path); ensure safe dtype/range before display.
- Uniformly support single-channel and multi-category masks.
- Fix prompt overlay guard: the `if prompt_type == "mask"` currently outside the loop should switch to an `any_mask_prompt` flag.
- Prefer deterministic category colors across a call (HSV with fixed steps; no per-call randomness).

### 4) Implement Category-level Prompt Merging
- Add helper (in `core/utils.py`): `merge_prompts_to_category(prompts, num_categories, hw)` returning:
  - `point_inputs_by_cat`: dictionary or list-of-lists grouping points by `category_id = obj_id % num_categories`.
  - `mask_inputs_by_cat`: tensor `[num_categories,H,W]` as logical OR/maximum across masks per category.
  - Optionally group bboxes per category for rendering.
- Replace placeholders:
  - `merged["point_inputs"] = None` → fill with grouped points structure.
  - `merged["mask_inputs"] = None` → fill with aggregated per-category mask tensor.
- Responsibility: merging only combines raw prompt inputs; no drawing/visualization logic inside the merger.

### 5) Use Merged Prompts in Visualization
- Update visualization to render from the merged structures:
  - Draw category-colored points (positive/negative variants if labels provided).
  - Alpha-blend `mask_inputs_by_cat` with contours per category.
  - Draw bboxes if present (optional), colored by category.

### 6) Hook Logging in Trainer
- In `core/trainer.py`, after `outs_per_frame = self.forward(batch)` within `training_step` and `validation_step`:
  - Extract prediction masks compatible with visualization (e.g., `outs_per_frame["pred_mask"]` or `outs_per_frame["masks"]`).
  - Call `log_training_visualizations(logger=self.logger, batch_data=batch, predictions=<pred_masks>, batch_idx=batch_idx, stage="train"/"val", max_samples=<small>)`.
- Add a simple frequency guard to avoid log spam:
  - Log every `n` steps or only at start/end of epoch; expose `n` via Hydra (`trainer.log_every_n_steps`) or set a conservative default.

### 7) Sanity-check
- Use a tiny synthetic batch: `frames[1,3,H,W]`, `gt_masks` (binary/multi-cat), simple `prompts` list, and random `predictions`.
- Validate overlays, contours, and prompt rendering run headless (matplotlib non-interactive backend).

### 8) Documentation
- Add concise docstrings to new helpers describing inputs/outputs and assumptions.
- Brief inline comments noting thresholds or shape expectations.
- Optional short note in README or code comments on enabling/disabling logging and frequency control.

## Quality Gates (KISS/YAGNI/DRY)
- Complexity_Check: Prefer minimal conversions; avoid extra abstractions.
- Necessity_Check: Only implement merging and logging paths required now.
- Responsibility_Check: Separate merging, visualization, and logging concerns.
- Interface_Check: Keep visualization/merging function signatures minimal and explicit.

