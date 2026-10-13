"""
Turn a Label Studio export (JSON-MIN or full JSON) into a YOLO-pose dataset
and fine-tune from it.

  python -m swimform.labeling.convert --export export.json --images images/ --out dataset/
  yolo pose train data=dataset/data.yaml model=yolo11m-pose.pt epochs=100 imgsz=960

The val split is held out BY CLIP, not by frame: frames from one clip are near
duplicates, so a frame-level split reports a score you will not reproduce on a
new swimmer.
"""
import argparse, json, os, shutil, random, re
from collections import defaultdict

COCO = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow",
        "l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]
IDX = {k: i for i, k in enumerate(COCO)}

def parse(export):
    """-> {image_filename: {"w":, "h":, "kp": {name: (x_px, y_px, vis)}, "usable": bool}}"""
    out = {}
    for t in export:
        anns = t.get("annotations") or []
        if not anns: continue
        img = t.get("data", {}).get("image", "")
        fn = re.sub(r"^.*[?&]d=", "", img).split("/")[-1]
        rec = {"w": None, "h": None, "kp": {}, "usable": True}
        vis_by_id = {}
        for r in anns[0].get("result", []):
            if r.get("type") == "choices":
                ch = r["value"]["choices"][0]
                if ch in ("visible", "occluded"): vis_by_id[r.get("id")] = ch
                elif ch == "unusable": rec["usable"] = False
        for r in anns[0].get("result", []):
            if r.get("type") != "keypointlabels": continue
            v = r["value"]; rec["w"] = r["original_width"]; rec["h"] = r["original_height"]
            name = v["keypointlabels"][0]
            vis = 1 if vis_by_id.get(r.get("id")) == "occluded" else 2
            rec["kp"][name] = (v["x"]/100*rec["w"], v["y"]/100*rec["h"], vis)
        if rec["kp"] and rec["usable"]: out[fn] = rec
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25, help="fraction of CLIPS held out")
    ap.add_argument("--pad", type=float, default=0.08, help="box padding as a fraction of size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    recs = parse(json.load(open(a.export)))
    by_clip = defaultdict(list)
    for fn in recs: by_clip[fn.split("__f")[0]].append(fn)
    clips = sorted(by_clip)
    random.Random(a.seed).shuffle(clips)
    n_val = max(1, round(len(clips)*a.val_frac))
    val_clips = set(clips[:n_val])
    print(f"{len(recs)} labeled frames over {len(clips)} clips; val clips: {sorted(val_clips)}")

    for split in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)

    kept = {"train": 0, "val": 0}
    for fn, r in recs.items():
        split = "val" if fn.split("__f")[0] in val_clips else "train"
        W, H = r["w"], r["h"]
        pts = list(r["kp"].values())
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        bw = (max(xs)-min(xs)); bh = (max(ys)-min(ys))
        cx = (max(xs)+min(xs))/2/W; cy = (max(ys)+min(ys))/2/H
        bw = min(1.0, (bw*(1+2*a.pad))/W); bh = min(1.0, (bh*(1+2*a.pad))/H)
        if bw <= 0 or bh <= 0: continue
        row = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
        for k in COCO:
            if k in r["kp"]:
                x, y, v = r["kp"][k]; row.append(f"{x/W:.6f} {y/H:.6f} {v}")
            else:
                row.append("0.000000 0.000000 0")
        shutil.copy(os.path.join(a.images, fn), os.path.join(a.out, "images", split, fn))
        open(os.path.join(a.out, "labels", split, os.path.splitext(fn)[0]+".txt"), "w").write(" ".join(row)+"\n")
        kept[split] += 1

    yaml = (f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [17, 3]\n"
            f"flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
            f"names:\n  0: swimmer\n")
    open(os.path.join(a.out, "data.yaml"), "w").write(yaml)
    print(f"train {kept['train']}  val {kept['val']}  -> {a.out}/data.yaml")
    print("\nnext:\n  yolo pose train data=%s/data.yaml model=yolo11m-pose.pt epochs=100 imgsz=960 "
          "batch=8 patience=25" % a.out)

if __name__ == "__main__":
    main()
