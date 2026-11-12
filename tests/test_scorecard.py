"""Pose-model scorecard: grades predictions against hand labels the way the analysis uses them."""
import numpy as np
import pytest

from swimform import scorecard as S
from swimform.schema import IDX, NUM_KEYPOINTS

W, H = 1280, 720


def swimmer(rng):
    """A plausible labeled swimmer: points spread along a horizontal body."""
    k = np.zeros((NUM_KEYPOINTS, 3))
    k[:, 0] = rng.uniform(300, 900, NUM_KEYPOINTS)
    k[:, 1] = rng.uniform(300, 420, NUM_KEYPOINTS)
    k[:, 2] = 2
    box = np.array([k[:, 0].min() - 20, k[:, 1].min() - 20, k[:, 0].max() + 20, k[:, 1].max() + 20])
    return box, k


def frames(n=40, seed=0, noise=0.01, conf=0.9, mutate=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        box, gt = swimmer(rng)
        size = np.sqrt((box[2] - box[0]) * (box[3] - box[1]))
        pred = gt.copy()
        pred[:, :2] += rng.normal(0, noise * size, (NUM_KEYPOINTS, 2))
        pred[:, 2] = conf
        f = S.Frame(f"img{i}.jpg", (W, H), box, gt, pred)
        if mutate:
            mutate(i, f, size)
        out.append(f)
    return out


def test_an_accurate_confident_model_passes():
    card = S.score(frames())
    o = card["overall"]
    assert o["passes"], o["fails"]
    assert o["confident_wrong"] == 0 and o["accuracy_far"] == 1 and o["coverage_visible"] == 1


def test_confidently_wrong_points_fail_the_keypoint_that_has_them():
    def bad_elbow(i, f, size):
        if i % 4 == 0:   # 25% of left-elbow predictions land 30% of body size away, confidently
            f.pred[IDX["l_elbow"], 0] += 0.3 * size

    card = S.score(frames(mutate=bad_elbow))
    row = next(r for r in card["keypoints"] if r["keypoint"] == "l_elbow")
    assert row["confident_wrong"] == pytest.approx(0.25)
    assert not card["overall"]["passes"]
    assert any(m.startswith("l_elbow: confidently wrong") for m in card["overall"]["fails"])
    assert all(not r["fails"] for r in card["keypoints"] if r["keypoint"] != "l_elbow")


def test_wrong_but_unsure_points_cost_coverage_not_correctness():
    def unsure(i, f, size):
        f.pred[IDX["r_wrist"], 0] += 0.5 * size
        f.pred[IDX["r_wrist"], 2] = 0.2          # below CONF: the analysis would ignore it

    card = S.score(frames(mutate=unsure))
    row = next(r for r in card["keypoints"] if r["keypoint"] == "r_wrist")
    assert row["coverage_visible"] == 0 and np.isnan(row["confident_wrong"])
    assert any("confident on visible points" in m for m in row["fails"])


def test_left_right_swaps_are_counted():
    def swap_hips(i, f, size):
        if i % 2 == 0:
            a, b = IDX["l_hip"], IDX["r_hip"]
            f.gt[b, :2] = f.gt[a, :2] + [0.2 * size, 0]        # make sure the pair is well separated
            f.pred[a, :2], f.pred[b, :2] = f.gt[b, :2].copy(), f.gt[a, :2].copy()

    o = S.score(frames(mutate=swap_hips))["overall"]
    assert o["swap"] > S.BAR["swap_max"] and not o["passes"]
    assert any("left/right swapped" in m for m in o["fails"])


def test_phantom_points_where_the_labeler_placed_none():
    def unlabeled_ankle(i, f, size):
        f.gt[IDX["l_ankle"]] = [np.nan, np.nan, 0]   # labeler couldn't locate it; model claims it anyway

    card = S.score(frames(mutate=unlabeled_ankle))
    row = next(r for r in card["keypoints"] if r["keypoint"] == "l_ankle")
    assert row["phantom"] == 1 and row["labeled"] == 0
    assert any("label more frames" in m for m in row["fails"])


def test_missed_swimmer_counts_every_labeled_point_against_the_frame():
    def lose(i, f, size):
        if i < 4:
            f.pred = None

    card = S.score(frames(mutate=lose))
    assert card["overall"]["missed_swimmer"] == pytest.approx(0.1)
    assert card["worst_frames"][0]["missed"]
    assert any("missed the swimmer" in m for m in card["overall"]["fails"])


def test_match_picks_the_detection_over_the_labeled_swimmer():
    box, gt = swimmer(np.random.default_rng(1))
    f = S.Frame("x.jpg", (W, H), box, gt)
    bystander = box + [500, -300, 500, -300]
    k_swim, k_by = np.ones((NUM_KEYPOINTS, 3)), np.zeros((NUM_KEYPOINTS, 3))
    assert (S.match(f, [bystander, box], [k_by, k_swim]) == 1).all()
    assert S.match(f, [bystander], [k_by]) is None


def test_reads_a_yolo_dataset_from_data_yaml(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "data.yaml").write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\n"
                                        f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n")
    box, gt = swimmer(rng)
    gt[IDX["r_knee"], 2] = 1
    gt[IDX["top_of_head"]] = [0, 0, 0]
    cx, cy, bw, bh = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H, (box[2] - box[0]) / W, (box[3] - box[1]) / H
    kp = " ".join(f"{x / W:.6f} {y / H:.6f} {int(v)}" for x, y, v in gt)
    (tmp_path / "labels" / "val" / "clipA__f0001.txt").write_text(f"0 {cx} {cy} {bw} {bh} {kp}\n")
    cv2.imwrite(str(tmp_path / "images" / "val" / "clipA__f0001.jpg"), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "images" / "val" / "unusable.jpg"), np.zeros((H, W, 3), np.uint8))   # no label
    fr = S.load_split(tmp_path / "data.yaml")
    assert len(fr) == 1 and fr[0].size == (W, H)
    assert np.allclose(fr[0].gt[IDX["l_elbow"], :2], gt[IDX["l_elbow"], :2], atol=0.01)
    assert fr[0].gt[IDX["r_knee"], 2] == 1 and np.isnan(fr[0].gt[IDX["top_of_head"], 0])
    assert np.allclose(fr[0].box, box, atol=0.01)


def test_label_with_the_wrong_number_of_keypoints_is_explained(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.2 0.2 " + " ".join(["0.5 0.5 2"] * 17) + "\n")   # a COCO-17 label
    with pytest.raises(ValueError, match="15 keypoints"):
        S.read_label(p, W, H)


def test_cli_writes_a_report_and_compares_rounds(tmp_path, monkeypatch):
    base = frames(n=30)
    worse = frames(n=30, mutate=lambda i, f, size: f.pred.__setitem__((IDX["l_wrist"], 0), f.pred[IDX["l_wrist"], 0] + 0.4 * size))
    monkeypatch.setattr(S, "load_split", lambda data, split="val": base)
    monkeypatch.setattr(S, "predict", lambda fr, w, imgsz=960: worse if "r2" in w else base)
    code = S.main(["--data", "d.yaml", "--weights", "r2.pt", "r3.pt", "--out", str(tmp_path)])
    md = (tmp_path / "scorecard.md").read_text()
    assert code == 1                                       # r2 fails, so not every model passes
    assert "Side by side" in md and "r2.pt: NOT READY" in md and "r3.pt: PASS" in md
    assert "l_wrist ⚠" in md and "Frames to look at first" in md
    assert (tmp_path / "scorecard.json").exists()
