## Loss Refactor Plan: Remove DDP, Drop Wrapper, Use MultiStepMultiMasksAndIous Directly

### Objective
Simplify the loss pipeline by removing distributed (DDP) code from `core/loss_fns.py`, deleting the `SAM2TrainingLoss` wrapper, and using `MultiStepMultiMasksAndIous` directly in the Lightning trainer. Keep configs minimal and avoid introducing new abstractions.

### Rationale (KISS/YAGNI/DRY)
- KISS: Eliminate unnecessary distributed helpers and wrapper layer.
- YAGNI: We only train on single process now; no DDP logic needed.
- DRY: Reuse the existing `MultiStepMultiMasksAndIous` loss module without duplicating orchestration.

### Scope
- Files: `core/loss_fns.py`, `core/trainer.py` (imports and criterion usage only).
- Configs/YAMLs: No changes required.

### Changes

1) core/loss_fns.py — remove DDP-related code
- Remove helpers: `is_dist_avail_and_initialized`, `get_world_size`.
- In `MultiStepMultiMasksAndIous.forward`:
  - Replace:
    - `num_objects = torch.tensor(..., device=..., dtype=torch.float)`
    - DDP section that all-reduces and divides by world size
  - With:
    - `num_objects = float(targets_batch.shape[1])` (or keep as scalar tensor on same device).
- Keep `CORE_LOSS_KEY = "core_loss"` and the existing loss computations intact.

2) core/loss_fns.py — remove wrapper class
- Delete the `SAM2TrainingLoss` class entirely.
- Ensure the module exports `MultiStepMultiMasksAndIous`, low-level loss functions, and `CORE_LOSS_KEY` only.

3) core/trainer.py — use MultiStepMultiMasksAndIous directly
- Imports:
  - Replace `from core.loss_fns import SAM2TrainingLoss` with:
    - `from core.loss_fns import MultiStepMultiMasksAndIous, CORE_LOSS_KEY`
- LightningModule.__init__:
  - Build `weight_dict` from config:
    - `{"loss_mask": cfg.loss.bce_weight, "loss_dice": cfg.loss.dice_weight, "loss_iou": cfg.loss.iou_weight, "loss_class": 0.0}`
  - Initialize: `self.criterion = MultiStepMultiMasksAndIous(weight_dict=weight_dict)`
  - If needed, pass optional args (`focal_alpha`, `focal_gamma`, etc.) with sensible defaults.
- training_step / validation_step:
  - Call model: `outs_per_frame = self.model(batch)`
  - Compute losses: `losses = self.criterion(outs_per_frame, batch.masks)`
  - Total loss: `total_loss = losses[CORE_LOSS_KEY]`
  - Components for logging: `{loss_mask, loss_dice, loss_iou} = losses`
  - Log keys remain `train/total_loss`, `train/loss_mask`, `train/loss_dice`, `train/loss_iou` and similarly for `val/*`.
- Validation aggregation already expects `val/total_loss`. Keep consistent.

### No Config Changes
- Keep `configs/loss/default.yaml` as-is (bce/dice/iou/temporal/smooth). `temporal_weight` is currently unused; leave it for now (no-op).

### Sanity Checks
- Shapes: `outs_per_frame` is `List[Dict]` per frame; each dict must contain keys used by the loss:
  - `"multistep_pred_multimasks_high_res"` (List[Tensor[N, K, H, W]])
  - `"multistep_pred_ious"` (List[Tensor[N, K]])
  - `"multistep_object_score_logits"` (List[Tensor[N, 1]])
- Targets: `batch.masks` is `[T, N, H, W]` with fixed N per batch.
- Single GPU/CPU only: no distributed calls remain.

### Acceptance Criteria
- Code compiles and runs training without referencing `SAM2TrainingLoss`.
- No imports or calls to `torch.distributed` in `core/loss_fns.py`.
- Trainer logs `train/total_loss` and `val/total_loss` using the reduced loss from `MultiStepMultiMasksAndIous`.

### Step-by-Step Implementation
1. Edit `core/loss_fns.py`:
   - Remove DDP helpers and their usages.
   - Simplify `num_objects` calculation.
   - Delete `SAM2TrainingLoss` class definition.
2. Edit `core/trainer.py`:
   - Update imports to `MultiStepMultiMasksAndIous, CORE_LOSS_KEY`.
   - Initialize `self.criterion` with a `weight_dict` from config.
   - Update `training_step` and `validation_step` to:
     - call criterion -> dict, pick `CORE_LOSS_KEY` as total, log components.
3. Run a quick local smoke test (tiny batch) to ensure no key errors and logs align.

### Rollback Plan
- Revert trainer to previous wrapper import if issues arise.
- Restore `SAM2TrainingLoss` from VCS history if needed.

