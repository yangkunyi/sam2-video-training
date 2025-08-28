from core.data.dataset import COCODataset, create_dataloader
from icecream import ic
import random

myloader = create_dataloader(
    dataset_type="coco",
    dataset_path="/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/cholecseg8k/coco_style/merged_gt_coco_annotations_test.json",
    num_workers=1,
    shuffle=False,
    number_of_pos_points=1,
    num_of_neg_points=0,
    include_center=True,
    prompt_types=["point"],
)

data = next(iter(myloader))
ic(data["img_batch"].shape)
ic(data["masks"].shape)
ic(data["obj_to_frame_idx"].shape)
ic(data["prompts"])
ic(data["video_paths"])
ic(data["num_objects"])
ic(data["flat_obj_to_img_idx"].shape)
ic(data["batch_size"])
ic(data["max_objects"])
ic(data["max_len"])
