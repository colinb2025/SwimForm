"""
Pose-model scorecard: is the fine-tuned model good enough to feed the analysis?

Grades a model against held-out hand-labeled frames (the YOLO-format val split that
labeling.convert writes) the way the analysis consumes its output. The analysis trusts a
keypoint when its confidence is at least schema.CONF and ignores it otherwise, so what
matters is:

  coverage         of the points you labeled, the share the model is confident about.
                   Low coverage costs checks ("couldn't check"), not correctness.
  accuracy         of the confident points, the share within 5% / 10% of the swimmer's
                   size (sqrt of the labeled box area) of your label.
  confident-wrong  confident points more than 10% away. These become wrong measurements
                   with nothing to warn anyone: the number that must stay smallest.
  phantom          confident points where you placed none (you couldn't locate the joint,
                   so the model shouldn't claim to).
  left/right swaps limb pairs the model put on the wrong side. Swaps flip rotation and
                   breathing-side metrics, and are the classic swimming failure.
  missed swimmer   labeled frames where the model found nobody overlapping your box.

Every number is reported per keypoint, and split into points you marked visible vs
occluded. The pass bar (BAR) is a draft; it errs strict because a bad model is the worst
outcome for the product.

  python -m swimform.scorecard --data dataset_r3/data.yaml --weights runs/pose/train/weights/best.pt
  python -m swimform.scorecard --data dataset_r3/data.yaml --weights r2.pt r3.pt     # compare rounds
  python -m swimform.scorecard --data dataset_r3/data.yaml --weights best.pt --draw 12 --out card/
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import CONF, FLIP_IDX, KEYPOINTS, NUM_KEYPOINTS

NEAR, FAR = 0.05, 0.10          # distance bands, as a share of swimmer size
PAIRS = [(i, FLIP_IDX[i]) for i in range(NUM_KEYPOINTS) if FLIP_IDX[i] > i]

# Draft pass bar. Applied per keypoint (min_points labeled points needed to judge one).
BAR = {
    "confident_wrong_max": 0.05,     # at most 5% of confident points >10% off
    "accuracy_far_min": 0.90,        # at least 90% of confident points within 10%
    "coverage_visible_min": 0.80,    # confident on at least 80% of points you marked visible
    "phantom_max": 0.10,             # at most 10% of unlabeled points claimed confidently
    "swap_max": 0.02,                # at most 2% of limb pairs swapped
    "missed_swimmer_max": 0.02,      # find the swimmer in at least 98% of frames
    "min_points": 20,
}


# ------------------------------------------------------------------ inputs
@dataclass
class Frame:
    """One labeled frame: the hand label and (after prediction) the model's best match."""
    image: str
    size: tuple                                   # (w, h) pixels
    box: np.ndarray                               # labeled box x1, y1, x2, y2 (px)
    gt: np.ndarray                                # (K, 3): x, y px, v (0 none, 1 occluded, 2 visible)
    pred: np.ndarray | None = None                # (K, 3): x, y px, confidence; None = no swimmer found
    notes: list = field(default_factory=list)


def read_label(path, w, h):
    """YOLO pose label -> list of (box px, kpts (K,3)). One line per person."""
    out = []
    for line in Path(path).read_text().split("\n"):
        v = line.split()
        if not v:
            continue
        nums = np.array(v[1:], dtype=float)
        if len(nums) != 4 + 3 * NUM_KEYPOINTS:
            raise ValueError(f"{path}: expected {4 + 3 * NUM_KEYPOINTS} numbers per person after the "
                             f"class (box + {NUM_KEYPOINTS} keypoints x/y/v), got {len(nums)}")
        cx, cy, bw, bh = nums[:4] * [w, h, w, h]
        k = nums[4:].reshape(NUM_KEYPOINTS, 3).copy()
        k[:, 0] *= w
        k[:, 1] *= h
        k[k[:, 2] == 0, :2] = np.nan
        out.append((np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), k))
    return out


def _image_size(path):
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f"can't read image {path}")
    return im.shape[1], im.shape[0]


def load_split(data_yaml, split="val"):
    """Frames of one split from a YOLO data.yaml. Takes the first person per label file:
    the labeling rules say one swimmer per frame."""
    root, images = _split_dir(Path(data_yaml), split)
    frames = []
    for img in sorted(p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        lab = _label_path(img)
        if not lab.exists():
            continue                                    # unusable frame: nothing labeled
        w, h = _image_size(img)
        people = read_label(lab, w, h)
        if people:
            frames.append(Frame(str(img), (w, h), people[0][0], people[0][1]))
    if not frames:
        raise SystemExit(f"no labeled images found under {images}")
    return frames


def _label_path(img):
    """YOLO convention: the last 'images' folder in the path becomes 'labels'."""
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _split_dir(data_yaml, split):
    """Minimal data.yaml reader (path:, val:) so this needs no YAML library."""
    conf = {}
    for line in data_yaml.read_text().splitlines():
        if ":" in line and not line.startswith((" ", "\t", "#")):
            k, v = line.split(":", 1)
            conf[k.strip()] = v.split("#")[0].strip().strip("'\"")
    base = Path(conf.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    rel = conf.get(split)
    if not rel:
        raise SystemExit(f"{data_yaml} has no '{split}:' entry")
    return base, base / rel


# ------------------------------------------------------------------ prediction
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(frame, boxes, kpts, min_iou=0.3):
    """Pick the detected person overlapping the labeled swimmer most; None if nobody does."""
    best, best_iou = None, min_iou
    for b, k in zip(boxes, kpts):
        i = _iou(frame.box, b)
        if i >= best_iou:
            best, best_iou = k, i
    return None if best is None else np.asarray(best, dtype=float)


def predict(frames, weights, imgsz=960):
    """Fill frame.pred with ultralytics weights. Returns frames (copies)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out = []
    for f in frames:
        r = model.predict(f.image, imgsz=imgsz, verbose=False)[0]
        g = Frame(f.image, f.size, f.box, f.gt)
        if r.keypoints is not None and len(r.boxes):
            xy = r.keypoints.xy.cpu().numpy()
            cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
            if xy.shape[1] != NUM_KEYPOINTS:
                raise SystemExit(f"{weights} predicts {xy.shape[1]} keypoints; the schema has {NUM_KEYPOINTS}")
            g.pred = match(f, r.boxes.xyxy.cpu().numpy(), np.concatenate([xy, cf[..., None]], axis=2))
        out.append(g)
    return out


# ------------------------------------------------------------------ scoring
def _size(box):
    return float(np.sqrt(max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0)))


def _swapped(gt, pred, i, j):
    """True when pred's (i, j) sit much closer to gt's (j, i): the sides are switched."""
    if np.isnan(gt[[i, j], :2]).any() or np.isnan(pred[[i, j], :2]).any():
        return False
    straight = np.linalg.norm(pred[i, :2] - gt[i, :2]) + np.linalg.norm(pred[j, :2] - gt[j, :2])
    crossed = np.linalg.norm(pred[i, :2] - gt[j, :2]) + np.linalg.norm(pred[j, :2] - gt[i, :2])
    return straight > 0 and crossed < 0.5 * straight


def score(frames, conf=CONF):
    """Per-keypoint and overall numbers for frames with .pred filled in."""
    K = NUM_KEYPOINTS
    c = {name: np.zeros(K) for name in ("labeled", "labeled_vis", "labeled_occ", "confident", "confident_vis",
                                        "confident_occ", "near", "far", "wrong", "unlabeled", "phantom")}
    pairs = swaps = missed = 0
    per_frame = []
    for f in frames:
        lab = f.gt[:, 2] > 0
        vis, occ = f.gt[:, 2] == 2, f.gt[:, 2] == 1
        c["labeled"] += lab
        c["labeled_vis"] += vis
        c["labeled_occ"] += occ
        c["unlabeled"] += ~lab
        if f.pred is None:
            missed += 1
            per_frame.append({"image": f.image, "missed": True, "wrong": int(lab.sum()), "swaps": 0})
            continue
        sure = f.pred[:, 2] >= conf
        d = np.linalg.norm(f.pred[:, :2] - f.gt[:, :2], axis=1) / _size(f.box)
        d = np.where(lab, d, np.inf)
        c["confident"] += sure & lab
        c["confident_vis"] += sure & vis
        c["confident_occ"] += sure & occ
        c["near"] += sure & lab & (d <= NEAR)
        c["far"] += sure & lab & (d <= FAR)
        wrong = sure & lab & (d > FAR)
        c["wrong"] += wrong
        c["phantom"] += sure & ~lab
        fs = 0
        for i, j in PAIRS:
            if lab[i] and lab[j] and sure[i] and sure[j]:
                pairs += 1
                if _swapped(f.gt, f.pred, i, j):
                    swaps += 1
                    fs += 1
        per_frame.append({"image": f.image, "missed": False, "wrong": int(wrong.sum()), "swaps": fs,
                          "worst": KEYPOINTS[int(np.nanargmax(np.where(sure & lab, d, -1)))] if (sure & lab).any() else None})

    def rate(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    kp = []
    for k in range(K):
        row = {"keypoint": KEYPOINTS[k], "labeled": int(c["labeled"][k]),
               "coverage_visible": rate(c["confident_vis"], c["labeled_vis"])[k],
               "coverage_occluded": rate(c["confident_occ"], c["labeled_occ"])[k],
               "accuracy_near": rate(c["near"], c["confident"])[k],
               "accuracy_far": rate(c["far"], c["confident"])[k],
               "confident_wrong": rate(c["wrong"], c["confident"])[k],
               "phantom": rate(c["phantom"], c["unlabeled"])[k]}
        row["fails"] = _fails(row)
        kp.append(row)
    tot = {k: v.sum() for k, v in c.items()}
    overall = {
        "frames": len(frames), "missed_swimmer": missed / len(frames) if frames else float("nan"),
        "coverage_visible": _r(tot["confident_vis"], tot["labeled_vis"]),
        "coverage_occluded": _r(tot["confident_occ"], tot["labeled_occ"]),
        "accuracy_near": _r(tot["near"], tot["confident"]), "accuracy_far": _r(tot["far"], tot["confident"]),
        "confident_wrong": _r(tot["wrong"], tot["confident"]), "phantom": _r(tot["phantom"], tot["unlabeled"]),
        "swap": _r(swaps, pairs), "limb_pairs": pairs,
    }
    overall["fails"] = []
    if overall["missed_swimmer"] > BAR["missed_swimmer_max"]:
        overall["fails"].append(f"missed the swimmer in {overall['missed_swimmer']:.0%} of frames")
    if pairs and overall["swap"] > BAR["swap_max"]:
        overall["fails"].append(f"left/right swapped on {overall['swap']:.1%} of limb pairs")
    for row in kp:
        overall["fails"] += [f"{row['keypoint']}: {m}" for m in row["fails"]]
    overall["passes"] = not overall["fails"]
    per_frame.sort(key=lambda r: (r["missed"], r["wrong"] + 2 * r["swaps"]), reverse=True)
    return {"overall": overall, "keypoints": kp, "worst_frames": per_frame, "bar": BAR, "conf": conf}


def _r(a, b):
    return float(a / b) if b else float("nan")


def _fails(row):
    if row["labeled"] < BAR["min_points"]:
        return [f"only {row['labeled']} labeled points; label more frames to judge it"]
    out = []
    checks = [("confident_wrong", ">", BAR["confident_wrong_max"], "confidently wrong"),
              ("accuracy_far", "<", BAR["accuracy_far_min"], "within 10%"),
              ("coverage_visible", "<", BAR["coverage_visible_min"], "confident on visible points"),
              ("phantom", ">", BAR["phantom_max"], "phantom points")]
    for key, op, lim, words in checks:
        v = row[key]
        if v is None or np.isnan(v):
            continue
        if (op == ">" and v > lim) or (op == "<" and v < lim):
            out.append(f"{words} {v:.0%} (bar {'≤' if op == '>' else '≥'} {lim:.0%})")
    return out


# ------------------------------------------------------------------ output
def _pct(v):
    return "  –  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:5.0%}"


def table(card, name="model"):
    o = card["overall"]
    lines = [f"## {name}: {'PASS' if o['passes'] else 'NOT READY'}", "",
             f"{o['frames']} frames · swimmer missed {_pct(o['missed_swimmer']).strip()} · "
             f"left/right swaps {_pct(o['swap']).strip()} of {o['limb_pairs']} limb pairs · "
             f"confident-wrong {_pct(o['confident_wrong']).strip()} overall", "",
             "| keypoint | labeled | coverage vis | coverage occ | within 5% | within 10% | confident-wrong | phantom |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in card["keypoints"]:
        flag = " ⚠" if r["fails"] else ""
        lines.append(f"| {r['keypoint']}{flag} | {r['labeled']} | {_pct(r['coverage_visible'])} | "
                     f"{_pct(r['coverage_occluded'])} | {_pct(r['accuracy_near'])} | {_pct(r['accuracy_far'])} | "
                     f"{_pct(r['confident_wrong'])} | {_pct(r['phantom'])} |")
    if o["fails"]:
        lines += ["", "What keeps it from passing:"] + [f"- {m}" for m in o["fails"]]
    worst = [w for w in card["worst_frames"] if w["missed"] or w["wrong"] or w["swaps"]][:10]
    if worst:
        lines += ["", "Frames to look at first:"]
        for w in worst:
            why = "swimmer not found" if w["missed"] else ", ".join(
                x for x in [f"{w['wrong']} confident-wrong" if w["wrong"] else "",
                            f"{w['swaps']} swapped pair(s)" if w["swaps"] else "",
                            f"worst: {w['worst']}" if w.get("worst") else ""] if x)
            lines.append(f"- {Path(w['image']).name}: {why}")
    return "\n".join(lines)


def compare(cards):
    names = list(cards)
    keys = [("confident_wrong", "confident-wrong (lower is better)"), ("accuracy_far", "within 10%"),
            ("coverage_visible", "coverage (visible)"), ("swap", "left/right swaps"),
            ("missed_swimmer", "swimmer missed"), ("phantom", "phantom points")]
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k, label in keys:
        lines.append(f"| {label} | " + " | ".join(_pct(cards[n]["overall"][k]).strip() for n in names) + " |")
    lines.append("| verdict | " + " | ".join("PASS" if cards[n]["overall"]["passes"] else "not ready" for n in names) + " |")
    return "\n".join(lines)


def draw(frames, card, outdir, n=12):
    """Overlay label (green) and model (red, confident only) on the worst frames."""
    import cv2
    from .schema import EDGES
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_image = {f.image: f for f in frames}
    for w in card["worst_frames"][:n]:
        f = by_image[w["image"]]
        im = cv2.imread(f.image)
        for pts, colour, ok in ((f.gt, (60, 200, 60), f.gt[:, 2] > 0),
                                (f.pred, (40, 40, 230), None if f.pred is None else f.pred[:, 2] >= card["conf"])):
            if pts is None:
                continue
            for a, b in EDGES:
                if ok[a] and ok[b]:
                    cv2.line(im, tuple(int(v) for v in pts[a, :2]), tuple(int(v) for v in pts[b, :2]), colour, 2)
            for k in np.flatnonzero(ok):
                cv2.circle(im, tuple(int(v) for v in pts[k, :2]), 4, colour, -1)
        cv2.imwrite(str(outdir / Path(f.image).name), im)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml from swimform.labeling.convert")
    ap.add_argument("--weights", nargs="+", required=True, help="one or more .pt files (several = compare)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", help="folder for scorecard.md, scorecard.json and drawn frames")
    ap.add_argument("--draw", type=int, default=0, help="draw this many worst frames per model into --out")
    a = ap.parse_args(argv)
    frames = load_split(a.data, a.split)
    cards, text = {}, []
    for w in a.weights:
        name = Path(w).parent.parent.name + "/" + Path(w).name if Path(w).parent.name == "weights" else Path(w).name
        done = predict(frames, w, a.imgsz)
        cards[name] = score(done)
        text.append(table(cards[name], name))
        if a.draw and a.out:
            draw(done, cards[name], Path(a.out) / "frames" / Path(name).stem, a.draw)
    if len(cards) > 1:
        text.insert(0, "## Side by side\n\n" + compare(cards))
    report = f"# Pose model scorecard\n\n{len(frames)} held-out frames from {a.data} ({a.split}).\n\n" + "\n\n".join(text)
    print(report)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "scorecard.md").write_text(report)
        (Path(a.out) / "scorecard.json").write_text(json.dumps(_jsonable(cards), indent=1))
    return 0 if all(c["overall"]["passes"] for c in cards.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
