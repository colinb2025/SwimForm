"""
The 15-point swim keypoint schema.

Replaces COCO-17's face points (nose, eyes, ears) with three spine points that a
horizontal, half-submerged swimmer actually shows: top of head, base of neck, and
the spine at the waist. Everything downstream refers to keypoints BY NAME; this
order only fixes array layout and the YOLO label column order.

If the Label Studio config uses different names for any point, change them here
and nowhere else.
"""

KEYPOINTS = (
    "top_of_head",
    "spine_at_neck",
    "spine_at_waist",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
)
IDX = {k: i for i, k in enumerate(KEYPOINTS)}
NUM_KEYPOINTS = len(KEYPOINTS)

# Index of each point's mirror image, for horizontal-flip augmentation.
FLIP_IDX = [IDX[k.replace("l_", "#").replace("r_", "l_").replace("#", "r_")]
            if k[:2] in ("l_", "r_") else IDX[k] for k in KEYPOINTS]

EDGES = (
    ("top_of_head", "spine_at_neck"), ("spine_at_neck", "spine_at_waist"),
    ("spine_at_neck", "l_shoulder"), ("spine_at_neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("spine_at_waist", "l_hip"), ("spine_at_waist", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
)

# Camera views. Names match the `view` column in rules/swim_rules.xlsx.
SIDE, HEAD_ON = "side", "head_on"
VIEWS = (SIDE, HEAD_ON)
# Workbook view for metrics measured on the 3D stroke fused from both clips.
BOTH = "both"

CONF = 0.5  # keypoint confidence at or above which a point is trusted
