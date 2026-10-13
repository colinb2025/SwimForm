"""
Baseline pose harness: run a pose model over swim clips, track the main swimmer,
write per-frame keypoints to JSON, render an overlay video, and report
per-joint detection rates.
"""
import json, sys, os, cv2, numpy as np
from ultralytics import YOLO

COCO = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow",
        "l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]
EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
CONF = 0.5  # keypoint confidence threshold for "detected"

def analyze(video, model, outdir, stride=1):
    name = os.path.splitext(os.path.basename(video))[0]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(outdir, f"{name}_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames, prev = [], None
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        res = model.predict(img, verbose=False, conf=0.25)[0]
        kp, box = None, None
        if res.keypoints is not None and len(res.boxes):
            xy = res.keypoints.xy.cpu().numpy()        # (n,17,2)
            cf = res.keypoints.conf.cpu().numpy()      # (n,17)
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            if prev is not None:   # prefer the detection closest to last frame's swimmer
                cx = (boxes[:,0]+boxes[:,2])/2; cy = (boxes[:,1]+boxes[:,3])/2
                d = np.hypot(cx-prev[0], cy-prev[1])
                j = int(np.argmin(d - 0.0005*areas))
            else:
                j = int(np.argmax(areas))
            kp, cfj, box = xy[j], cf[j], boxes[j]
            prev = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            frames.append({"frame": idx, "t": idx/fps,
                           "box": [round(float(v),1) for v in box],
                           "kp": {COCO[k]: [round(float(kp[k][0]),1), round(float(kp[k][1]),1),
                                            round(float(cfj[k]),3)] for k in range(17)}})
            # draw
            for a,b in EDGES:
                if cfj[a] > CONF and cfj[b] > CONF:
                    cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0,255,180), 2)
            for k in range(17):
                if cfj[k] > CONF:
                    cv2.circle(img, tuple(kp[k].astype(int)), 4, (0,120,255), -1)
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,255,0), 1)
        else:
            frames.append({"frame": idx, "t": idx/fps, "box": None, "kp": None})
        cv2.putText(img, f"{idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        writer.write(img); idx += 1
    cap.release(); writer.release()

    n = len(frames)
    det = sum(1 for f in frames if f["kp"])
    rates = {}
    for k in COCO:
        c = sum(1 for f in frames if f["kp"] and f["kp"][k][2] > CONF)
        rates[k] = round(c/n, 3) if n else 0
    summary = {"video": os.path.basename(video), "frames": n, "fps": round(fps,1),
               "size": [W,H], "swimmer_detected_rate": round(det/n,3) if n else 0,
               "joint_rates": rates}
    json.dump({"summary": summary, "frames": frames},
              open(os.path.join(outdir, f"{name}_pose.json"), "w"))
    return summary

if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    model = YOLO(sys.argv[2])
    out = []
    for v in sys.argv[3:]:
        s = analyze(v, model, outdir)
        print(json.dumps(s)); sys.stdout.flush()
        out.append(s)
    json.dump(out, open(os.path.join(outdir, "baseline_summary.json"), "w"), indent=2)
