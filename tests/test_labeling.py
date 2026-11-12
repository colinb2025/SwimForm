"""
The labeling loop on the 15-point schema, end to end without a GPU or Label Studio:
model guesses (COCO-17 or 15-point) -> prep picks frames and pre-fills them -> a Label
Studio export -> convert writes a YOLO-pose dataset -> the scorecard reads it.
"""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from swimform import pose as harness
from swimform import scorecard
from swimform.labeling import config as C
from swimform.labeling import convert, prep
from swimform.schema import FLIP_IDX, KEYPOINTS, NUM_KEYPOINTS

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_config_is_generated_from_the_schema():
    assert (ROOT / "labeling" / "labeling_config.xml").read_text() == C.build_config()


def test_config_has_exactly_the_15_points_with_hotkeys():
    cfg = C.build_config()
    for k in KEYPOINTS:
        assert f'<Label value="{k}"' in cfg
    assert "nose" not in cfg and cfg.count("hotkey=") == NUM_KEYPOINTS
    assert len(set(C.HOTKEYS[:NUM_KEYPOINTS])) == NUM_KEYPOINTS


@pytest.mark.parametrize("name,expected", [
    ("l_shoulder", "l_shoulder"), ("left_shoulder", "l_shoulder"), ("Right Ankle", "r_ankle"),
    ("neck", "spine_at_neck"), ("mid_hip", "spine_at_waist"), ("head-top", "top_of_head"), ("nose", None)])
def test_label_names_are_matched_by_meaning(name, expected):
    assert C.canonical(name) == expected


def test_coco_guesses_map_onto_the_schema():
    coco = {k: [10.0 * i, 5.0 * i, 0.9] for i, k in enumerate(harness.COCO)}
    pts = C.model_points(coco)
    assert "top_of_head" not in pts                     # the face isn't the crown: left to the labeler
    assert set(pts) == set(KEYPOINTS) - {"top_of_head"}
    neck = pts["spine_at_neck"]
    assert neck[3] is True and neck[0] == pytest.approx((coco["l_shoulder"][0] + coco["r_shoulder"][0]) / 2)
    assert pts["l_wrist"][:3] == tuple(coco["l_wrist"]) and pts["l_wrist"][3] is False


def test_harness_names_points_from_the_model():
    assert harness.layout(15)[0] == list(KEYPOINTS)
    assert harness.layout(17)[0] == harness.COCO
    with pytest.raises(ValueError):
        harness.layout(21)
    boxes = np.array([[0, 0, 10, 10], [100, 100, 300, 200.0]])
    assert harness.choose(boxes, None) == 1                       # biggest first
    assert harness.choose(boxes, (5, 5)) == 0                     # then follow the swimmer


# ------------------------------------------------------------------ the loop
def _clip(path, n=60, size=(320, 180)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), (180, 120, 40), np.uint8)
        cv2.circle(img, (40 + 3 * i, 90), 12, (255, 255, 255), -1)
        w.write(img)
    w.release()


def _guesses(names, n=60, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        if i % 10 == 9:
            frames.append({"frame": i, "t": i / 30, "box": None, "kp": None})   # swimmer lost
            continue
        conf = rng.uniform(0.3, 0.95)
        frames.append({"frame": i, "t": i / 30, "box": [0, 0, 1, 1],
                       "kp": {k: [float(40 + 3 * i + 5 * j), float(60 + j), float(conf)] for j, k in enumerate(names)}})
    return {"summary": {"fps": 30, "size": [320, 180]}, "frames": frames}


@pytest.fixture()
def round_dir(tmp_path):
    vids, pose = tmp_path / "clips", tmp_path / "pose"
    vids.mkdir(), pose.mkdir()
    for c, names in (("swimmerA", harness.COCO), ("swimmerB", harness.COCO), ("swimmerC", list(KEYPOINTS))):
        _clip(vids / f"{c}.mp4")
        (pose / f"{c}_pose.json").write_text(json.dumps(_guesses(names, seed=len(c))))
    out = tmp_path / "round1"
    prep.main(["--videos", str(vids), "--pose", str(pose), "--out", str(out), "--per-clip", "8", "--min-gap", "3",
               "--prefill-min-conf", "0.2"])
    return out


def test_prep_writes_15_point_tasks_and_config(round_dir):
    tasks = json.loads((round_dir / "tasks.json").read_text())
    assert len(tasks) == 24 and len(list((round_dir / "images").glob("*.jpg"))) == 24
    assert (round_dir / "labeling_config.xml").read_text() == C.build_config()
    placed = {r["value"]["keypointlabels"][0] for t in tasks for p in t["predictions"] for r in p["result"]}
    assert placed <= set(KEYPOINTS) and {"spine_at_neck", "spine_at_waist", "l_elbow"} <= placed
    assert {t["data"]["picked_as"] for t in tasks} >= {"uncertain", "missed"}


def _export(tasks):
    """What Label Studio exports after a labeler accepts the pre-placed points, marks the
    elbows occluded, and fixes up a few frames."""
    out = []
    for t in tasks:
        res = [dict(r) for p in t["predictions"] for r in p["result"]]
        if not res:
            continue
        i = len(out)
        for r in list(res):
            name = r["value"]["keypointlabels"][0]
            if i == 1:   # a project that spells names its own way
                r["value"] = dict(r["value"], keypointlabels=[name.replace("l_", "left_").replace("r_", "right_")])
            vis = "occluded" if name.endswith("elbow") else "visible"
            res.append({"id": r["id"], "type": "choices", "from_name": "vis", "to_name": "img",
                        "value": {"choices": [vis]}})
        if i == 2:
            res.append({"id": "x", "type": "keypointlabels", "original_width": 320, "original_height": 180,
                        "value": {"x": 1, "y": 1, "keypointlabels": ["nose"]}})
        frame = "unusable" if i == 3 else "usable"
        res.append({"id": f"f{i}", "type": "choices", "from_name": "frame", "to_name": "img",
                    "value": {"choices": [frame]}})
        out.append({"data": t["data"], "annotations": [{"result": res}]})
    return out


def test_export_converts_to_a_15_point_dataset_the_scorecard_reads(round_dir, tmp_path, capsys):
    tasks = json.loads((round_dir / "tasks.json").read_text())
    exp = tmp_path / "export.json"
    exp.write_text(json.dumps(_export(tasks)))
    ds = tmp_path / "dataset_r1"
    convert.main(["--export", str(exp), "--images", str(round_dir / "images"), "--out", str(ds), "--val-frac", "0.34"])
    printed = capsys.readouterr().out
    assert "nose x1" in printed                                    # unknown label reported, not silently kept

    y = (ds / "data.yaml").read_text()
    assert f"kpt_shape: [{NUM_KEYPOINTS}, 3]" in y and f"flip_idx: {list(FLIP_IDX)}" in y

    labels = list((ds / "labels").rglob("*.txt"))
    assert labels
    for lab in labels:
        nums = lab.read_text().split()
        assert len(nums) == 1 + 4 + 3 * NUM_KEYPOINTS
    # the renamed frame kept its points; elbows came through as occluded (v=1)
    vis = np.array([[float(v) for v in lab.read_text().split()[5:]] for lab in labels]).reshape(len(labels), -1, 3)[..., 2]
    i_elbow = KEYPOINTS.index("l_elbow")
    assert set(vis[:, i_elbow]) <= {0.0, 1.0} and (vis[:, i_elbow] == 1).any()
    assert (vis[:, KEYPOINTS.index("l_shoulder")] == 2).sum() >= len(labels) - 1

    # held out by clip: no clip appears in both splits
    split = {s: {p.name.split("__f")[0] for p in (ds / "images" / s).glob("*.jpg")} for s in ("train", "val")}
    assert split["train"] and split["val"] and not split["train"] & split["val"]

    frames = scorecard.load_split(ds / "data.yaml", "val")
    assert frames and frames[0].gt.shape == (NUM_KEYPOINTS, 3)
