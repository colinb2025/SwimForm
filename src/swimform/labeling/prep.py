"""
Round N of the labeling loop: pick frames worth labeling, export them as images,
and pre-fill the current model's guesses as Label Studio predictions.

  python -m swimform.labeling.prep --videos <dir> --pose <dir> --out <dir> --per-clip 40

Frames are chosen by a mix of:
  uncertain  - detected, but low mean keypoint confidence (where the model is wobbling)
  missed     - no detection at all (submerged / splash; the model must learn these)
  confident  - a few easy frames, to keep the set honest and catch silent drift
and spaced apart in time so you aren't labeling the same instant twice.

Labels use the 15-point schema (swimform.schema). The guesses can come from the stock
COCO-17 model (round 0: its twelve limb points carry over, the spine points start at
the shoulder and hip midpoints, the top of the head is left to you) or from our own
15-point weights (every later round).
"""
import argparse, json, os, glob, cv2, numpy as np

from .config import build_config, model_points

VIDEO_EXT = ("mov", "mp4", "m4v", "avi", "mkv", "MOV", "MP4", "M4V")


def pick(frames, n, min_gap):
    """Split the budget across uncertain / missed / confident, spaced by min_gap."""
    scored = []
    for f in frames:
        if f["kp"] is None:
            scored.append((f["frame"], "missed", -1.0))
        else:
            c = float(np.mean([v[2] for v in f["kp"].values()]))
            scored.append((f["frame"], "uncertain" if c < 0.6 else "confident", c))
    buckets = {"uncertain": [], "missed": [], "confident": []}
    for idx, kind, c in scored: buckets[kind].append((idx, c))
    buckets["uncertain"].sort(key=lambda t: t[1])            # least confident first
    buckets["confident"].sort(key=lambda t: -t[1])
    quota = {"uncertain": int(n*0.6), "missed": int(n*0.2), "confident": n - int(n*0.6) - int(n*0.2)}
    chosen, taken = [], []
    for kind in ("uncertain", "missed", "confident"):
        got = 0
        for idx, c in buckets[kind]:
            if got >= quota[kind]: break
            if any(abs(idx-t) < min_gap for t in taken): continue
            taken.append(idx); chosen.append((idx, kind)); got += 1
    return sorted(chosen)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of source clips")
    ap.add_argument("--pose", required=True, help="directory of *_pose.json from swimform.pose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-clip", type=int, default=40)
    ap.add_argument("--min-gap", type=int, default=6, help="min frames between picks")
    ap.add_argument("--prefill-min-conf", type=float, default=0.5,
                    help="only pre-place a point above this confidence; 1.0 = label from scratch")
    ap.add_argument("--model-version", default="round0-yolo11m")
    a = ap.parse_args(argv)

    img_dir = os.path.join(a.out, "images"); os.makedirs(img_dir, exist_ok=True)
    tasks, counts = [], {}
    vids = {os.path.splitext(os.path.basename(v))[0]: v
            for ext in VIDEO_EXT for v in glob.glob(os.path.join(a.videos, f"*.{ext}"))}

    for pj in sorted(glob.glob(os.path.join(a.pose, "*_pose.json"))):
        stem = os.path.basename(pj)[:-len("_pose.json")]
        if stem not in vids:
            print(f"  ! no video for {stem}, skipping"); continue
        d = json.load(open(pj)); frames = d["frames"]
        want = dict(pick(frames, a.per_clip, a.min_gap))
        cap = cv2.VideoCapture(vids[stem]); idx = 0; saved = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            if idx in want:
                H, W = img.shape[:2]
                fn = f"{stem}__f{idx:05d}.jpg"
                cv2.imwrite(os.path.join(img_dir, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                res, f = [], frames[idx]
                if f["kp"]:
                    for k, (x, y, c, _derived) in model_points(f["kp"]).items():
                        if c < a.prefill_min_conf: continue
                        res.append({"original_width": W, "original_height": H, "image_rotation": 0,
                                    "from_name": "kp", "to_name": "img", "type": "keypointlabels",
                                    "id": f"{idx}_{k}",
                                    "value": {"x": 100*x/W, "y": 100*y/H, "width": 0.4,
                                              "keypointlabels": [k]}})
                tasks.append({"data": {"image": f"/data/local-files/?d=images/{fn}",
                                       "clip": stem, "frame": idx, "picked_as": want[idx]},
                              "predictions": [{"model_version": a.model_version, "result": res}] if res else []})
                saved += 1
            idx += 1
        cap.release(); counts[stem] = saved
        print(f"  {stem}: {saved} frames")

    json.dump(tasks, open(os.path.join(a.out, "tasks.json"), "w"), indent=1)
    open(os.path.join(a.out, "labeling_config.xml"), "w").write(build_config())
    prefilled = sum(1 for t in tasks if t["predictions"])
    print(f"\n{len(tasks)} tasks -> {a.out}/tasks.json  ({prefilled} with pre-placed points)")

if __name__ == "__main__":
    main()
