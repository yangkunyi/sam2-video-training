# Loss Integration Plan (core/loss_fns.py -> core/trainer.py)

## Objective
Integrate the new loss implementations in `core/loss_fns.py` into the training loop, replacing the legacy reference and ensuring Hydra configs stay minimal and aligned with KISS/YAGNI/DRY.

## Current State
- Trainer imports `core.loss.SAM2TrainingLoss` (module no longer exists).
- `core/loss_fns.py` defines low-level losses and `MultiStepMultiMasksAndIous`, but was not wired into the trainer.
- `SAM2Model.forward` returns per-frame outputs and merges object-level results to category-level.
- Hydra configs already have a `loss` group with basic weights.

## Plan
1. Replace trainer loss import
   - Change `from core.loss import SAM2TrainingLoss` to `from core.loss_fns import SAM2TrainingLoss`.

2. Provide minimal high-level loss wrapper
   - Add `SAM2TrainingLoss` class in `core/loss_fns.py` that wraps `MultiStepMultiMasksAndIous` and exposes a simple interface:
     - Inputs: `outs_per_frame: List[Dict]`, `target_masks: Tensor[T, N, H, W]`.
     - Outputs: `(total_loss, {loss_mask, loss_dice, loss_iou})`.
   - Avoid circular imports by defining `CORE_LOSS_KEY` and small DDP utils locally.

3. Adjust trainer training/validation steps
   - Use model `forward(batch)` to get `outs_per_frame` directly.
   - Pass `outs_per_frame` and `batch.masks` to criterion.
   - Standardize logged metric keys to `train/*` and `val/*` (e.g., `val/total_loss`).

4. Ensure model forward returns expected structure
   - Confirm `core/sam2model.py` returns a List[Dict] of category-merged per-frame outputs; update return type accordingly.

5. Fix aggregation helper
   - In `merge_object_results_to_category`, derive number of categories with `max(obj_to_cat)+1` so categories are indexed properly.

6. Config alignment (Hydra + dataclasses)
   - Keep existing `LossConfig` intact (bce/dice/iou/temporal/smooth). No new knobs unless required.
   - No YAML changes needed beyond existing defaults.

## Acceptance Criteria
- Training runs without referencing `core.loss`.
- Loss computes from `outs_per_frame` and `batch.masks` for both train/val.
- Metrics logged as `train/total_loss`, `train/loss_*`, `val/total_loss`, `val/loss_*`.
- Best model checkpointing monitors `val/total_loss` (train.py already uses this).

## Notes on KISS/YAGNI/DRY
- KISS: Use a single wrapper `SAM2TrainingLoss` around already present building blocks.
- YAGNI: No extra abstraction layers, plugins, or registry mechanisms added.
- DRY: Reuse `MultiStepMultiMasksAndIous`; avoid duplicating logic in trainer.

## Follow-ups (Optional)
- Add tiny CPU unit tests for the loss wrapper with synthetic masks.
- Consider enabling/validating IoU supervision once predictions expose stable IoU logits.

