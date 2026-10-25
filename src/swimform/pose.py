"""
Pose harness: run a pose model over swim clips, track the main swimmer, write
per-frame keypoints to JSON, render an overlay video, and report per-joint
detection rates.

Works with the stock COCO-17 model (round 0 of labeling) and with our fine-tuned
15-point weights: keypoint names are taken from how many points the model predicts.
The JSON is the PoseTrack contract (swimform.posedata), which loads either.

  python -m swimform.pose out/ weights.pt clip1.mp4 clip2.mov ...
"""
import json, sys, os

import numpy as np

from .schema import EDGES as SCHEMA_EDGES, KEYPOINTS

COCO = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow",
        "l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]
COCO_EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
CONF = 0.5  # keypoint confidence threshold for "detected"


def layout(n_points):
    """(keypoint names, skeleton edges as index pairs) for a model predicting n points."""
    if n_points == len(KEYPOINTS):
        idx = {k: i for i, k in enumerate(KEYPOINTS)}
        return list(KEYPOINTS), [(idx[a], idx[b]) for a, b in SCHEMA_EDGES]
    if n_points == len(COCO):
        return COCO, COCO_EDGES
    raise ValueError(f"model predicts {n_points} keypoints; expected {len(KEYPOINTS)} (ours) or 17 (COCO)")


def choose(boxes, prev):
    """Index of the swimmer to follow: nearest last frame's swimmer, else the biggest box."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev is None:
        return int(np.argmax(areas))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    d = np.hypot(cx - prev[0], cy - prev[1])
    return int(np.argmin(d - 0.0005 * areas))


def analyze(video, model, outdir, stride=1):
    import cv2
    name = os.path.splitext(os.path.basename(video))[0]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(outdir, f"{name}_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames, prev, names = [], None, None
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        res = model.predict(img, verbose=False, conf=0.25)[0]
        if res.keypoints is not None and len(res.boxes):
            xy = res.keypoints.xy.cpu().numpy()        # (n, K, 2)
            cf = res.keypoints.conf.cpu().numpy()      # (n, K)
            if names is None:
                names, edges = layout(xy.shape[1])
            boxes = res.boxes.xyxy.cpu().numpy()
            j = choose(boxes, prev)
            kp, cfj, box = xy[j], cf[j], boxes[j]
            prev = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            frames.append({"frame": idx, "t": idx/fps,
                           "box": [round(float(v),1) for v in box],
                           "kp": {names[k]: [round(float(kp[k][0]),1), round(float(kp[k][1]),1),
                                             round(float(cfj[k]),3)] for k in range(len(names))}})
            for a,b in edges:
                if cfj[a] > CONF and cfj[b] > CONF:
                    cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0,255,180), 2)
            for k in range(len(names)):
                if cfj[k] > CONF:
                    cv2.circle(img, tuple(kp[k].astype(int)), 4, (0,120,255), -1)
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,255,0), 1)
        else:
            frames.append({"frame": idx, "t": idx/fps, "box": None, "kp": None})
        cv2.putText(img, f"{idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        writer.write(img); idx += 1
    cap.release(); writer.release()
    summary = summarize(frames, fps, (W, H), os.path.basename(video), names)
    with open(os.path.join(outdir, f"{name}_pose.json"), "w") as f:
        json.dump({"summary": summary, "frames": frames}, f)
    return summary


def summarize(frames, fps, size, video, names):
    names = names or list(KEYPOINTS)
    n = len(frames)
    det = sum(1 for f in frames if f["kp"])
    rates = {k: round(sum(1 for f in frames if f["kp"] and f["kp"][k][2] > CONF) / n, 3) if n else 0
             for k in names}
    return {"video": video, "frames": n, "fps": round(fps, 1), "size": list(size),
            "keypoints": list(names), "swimmer_detected_rate": round(det/n, 3) if n else 0,
            "joint_rates": rates}


def main(argv=None):
    from ultralytics import YOLO
    argv = sys.argv[1:] if argv is None else argv
    outdir = argv[0]; os.makedirs(outdir, exist_ok=True)
    model = YOLO(argv[1])
    out = []
    for v in argv[2:]:
        s = analyze(v, model, outdir)
        print(json.dumps(s)); sys.stdout.flush()
        out.append(s)
    with open(os.path.join(outdir, "baseline_summary.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
