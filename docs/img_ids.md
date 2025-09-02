# `img_ids` in `core/sam2model.py`

This note explains what `img_ids` is, how it is constructed, and how it is used during forward tracking.

## Context

Code site:

```python
# core/sam2model.py
img_ids = input.flat_obj_to_img_idx[frame_idx]
```

## Meaning

- Purpose: A per-frame mapping from "objects at this time step" to the corresponding flattened image indices.
- Type/shape: A 1D tensor of length `O` (number of objects per frame), indexing into the flattened image batch of size `B*T`.
- Interpretation: If multiple objects belong to the same (video, frame), their entries in `img_ids` are identical — that is expected.

## Construction

Defined in `core/data_utils.py` as a property of `BatchedVideoDatapoint`:

```python
@property
def flat_obj_to_img_idx(self) -> torch.IntTensor:
    frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
    flat_idx = video_idx * self.num_frames + frame_idx
    return flat_idx
```

Where the inputs come from:

- `img_batch`: shape `[T, B, C, H, W]`.
- `obj_to_frame_idx`: shape `[T, O, 2]`, where each entry is `[frame_idx=t, video_idx]` for an object present at time `t`.
- `flat_obj_to_img_idx`: shape `[T, O]`, computed by flattening `(t, b)` to `b*T + t` so it can index into `flat_img_batch`.
- `flat_img_batch`: shape `[(B*T), C, H, W]` via `self.img_batch.transpose(0, 1).flatten(0, 1)`.

Thus, for a given `frame_idx = t`, `img_ids = flat_obj_to_img_idx[t]` has length `O` and contains indices in `[0, B*T-1]` pointing to the correct frames across the batch for that time step.

## Example

Let `B=2`, `T=3`. The flattened image indices map as:

- `(t=0, b=0) → 0`, `(1, 0) → 1`, `(2, 0) → 2`
- `(0, 1) → 3`, `(1, 1) → 4`, `(2, 1) → 5`

At `t=1`, if there are objects from both videos, `img_ids = [1, 4, ...]`. If two objects come from the same `(t=1, b=0)` frame, `img_ids` may include `[1, 1, ...]`.

## Usage in `forward_tracking`

Two paths share the same gather semantics:

1) Precomputed features for all `B*T` images are available.

```python
current_vision_feats       = [x[:, img_ids] for x in vision_feats]
current_vision_pos_embeds  = [x[:, img_ids] for x in vision_pos_embeds]
```

This selects columns (batch dimension) corresponding to the current frame's objects, producing features with batch size `O`.

2) Compute on-the-fly for the current `img_ids` (to avoid redundant backbone work):

```python
unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
image = input.flat_img_batch[unique_img_ids]
backbone_out = self.forward_image(image)
# ... prepare features ...
vision_feats       = [x[:, inv_ids] for x in vision_feats]
vision_pos_embeds  = [x[:, inv_ids] for x in vision_pos_embeds]
```

- `unique_img_ids` ensures each image’s backbone pass runs once.
- `inv_ids` expands the computed features back to per-object shape `O` (duplicating features where multiple objects share an image).

## Notes / Pitfalls

- Dtype: Ensure `img_ids` is an integer (`long`) tensor when used for indexing.
- Duplication is expected: multiple objects in the same frame share the same `img_id`.
- Shapes at a glance:
  - `img_batch`: `[T, B, C, H, W]`
  - `flat_img_batch`: `[(B*T), C, H, W]`
  - `obj_to_frame_idx`: `[T, O, 2]`
  - `flat_obj_to_img_idx`: `[T, O]`
  - `img_ids` at time `t`: `[O]`

