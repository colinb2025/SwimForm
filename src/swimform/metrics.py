"""
Turn per-frame pose JSON into stroke cycles and the side-view metrics
defined in swim_rules.xlsx. Head-on metrics need a separate path.
"""
import json, sys, os, math
import numpy as np

CONF = 0.5

def load(path):
    d = json.load(open(path))
    return d["summary"], d["frames"]

def pt(f, name):
    v = f["kp"][name] if f["kp"] else None
    if v is None or v[2] < CONF: return None
    return np.array(v[:2])

def angle(a, b, c):
    """Interior angle at b, in degrees."""
    if a is None or b is None or c is None: return None
    v1, v2 = a-b, c-b
    n = np.linalg.norm(v1)*np.linalg.norm(v2)
    if n == 0: return None
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/n, -1, 1)))

def vec_angle_from_horizontal(a, b):
    if a is None or b is None: return None
    d = b-a
    return math.degrees(math.atan2(abs(d[1]), abs(d[0])))

def torso_len(f):
    """Scale reference: shoulder-to-hip distance, either side."""
    for s, h in (("l_shoulder","l_hip"), ("r_shoulder","r_hip")):
        a, b = pt(f,s), pt(f,h)
        if a is not None and b is not None:
            d = np.linalg.norm(a-b)
            if d > 1: return d
    return None

def wrist_series(frames, side):
    """Vertical position of one wrist per frame, NaN where missing."""
    out = []
    for f in frames:
        p = pt(f, f"{side}_wrist")
        out.append(np.nan if p is None else p[1])
    return np.array(out, dtype=float)

def find_cycles(y, fps, min_sec=0.6, max_sec=2.5):
    """Stroke cycles from wrist-height minima (highest point of recovery)."""
    v = y.copy()
    ok = ~np.isnan(v)
    if ok.sum() < len(v)*0.4: return []
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    k = max(3, int(fps*0.08)) | 1
    sm = np.convolve(v, np.ones(k)/k, mode="same")
    peaks = []
    gap = int(fps*min_sec)
    for i in range(1, len(sm)-1):
        if sm[i] <= sm[i-1] and sm[i] <= sm[i+1]:
            if not peaks or i-peaks[-1] >= gap: peaks.append(i)
    cycles = []
    for a, b in zip(peaks, peaks[1:]):
        dur = (b-a)/fps
        if min_sec <= dur <= max_sec: cycles.append((a, b))
    return cycles

def cycle_quality(frames, a, b, joints):
    """Share of frames in the cycle where every needed joint is confident."""
    n = b-a
    if n <= 0: return 0.0
    good = sum(1 for f in frames[a:b] if f["kp"] and all(f["kp"][j][2] > CONF for j in joints))
    return good/n

def side_metrics(frames, fps, side):
    """Compute the above-water side-view metrics per cycle."""
    y = wrist_series(frames, side)
    cycles = find_cycles(y, fps)
    sh, el, wr, hip = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_hip"
    rows = []
    for a, b in cycles:
        q = cycle_quality(frames, a, b, [sh, el, wr])
        seg = frames[a:b]
        # mid-recovery = frame where wrist is highest (smallest y) in first half
        half = seg[:max(1,len(seg)//2)]
        cand = [(f["kp"][wr][1], f) for f in half if f["kp"] and f["kp"][wr][2] > CONF]
        rec_ang = rec_h = None
        if cand:
            f = min(cand, key=lambda t: t[0])[1]
            rec_ang = angle(pt(f,sh), pt(f,el), pt(f,wr))
            tl = torso_len(f)
            e, w = pt(f,el), pt(f,wr)
            if tl and e is not None and w is not None:
                rec_h = round(float((w[1]-e[1])/tl), 3)
        # head lift: ear->nose line vs shoulder->hip line
        hl = []
        for f in seg:
            if not f["kp"]: continue
            nose, ear, s, h = pt(f,"nose"), pt(f,f"{side}_ear"), pt(f,sh), pt(f,hip)
            if all(v is not None for v in (nose, ear, s, h)):
                a1 = math.degrees(math.atan2(*(nose-ear)[::-1]))
                a2 = math.degrees(math.atan2(*(h-s)[::-1]))
                hl.append(abs(((a1-a2+90+180) % 360) - 180))
        # body line: shoulder->ankle below horizontal
        bl = []
        for f in seg:
            if not f["kp"]: continue
            s, ank = pt(f,sh), pt(f,f"{side}_ankle")
            if s is not None and ank is not None:
                bl.append(vec_angle_from_horizontal(s, ank))
        rows.append({"cycle": [a,b], "dur_s": round((b-a)/fps,2), "quality": round(q,2),
                     "recovery_elbow_angle": None if rec_ang is None else round(rec_ang,1),
                     "recovery_elbow_height": rec_h,
                     "head_lift_angle": round(float(np.median(hl)),1) if hl else None,
                     "body_line_angle": round(float(np.median(bl)),1) if bl else None})
    rate = round(60/np.median([r["dur_s"] for r in rows]),1) if rows else None
    return {"side": side, "cycles": rows, "stroke_rate_cpm": rate}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        s, frames = load(p)
        fps = s["fps"]
        print("="*70); print(s["video"], f'{s["frames"]} frames @ {fps}fps  det={s["swimmer_detected_rate"]}')
        print(" joint rates:", {k:v for k,v in s["joint_rates"].items() if k in
              ("nose","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip","l_ankle","r_ankle")})
        for side in ("l","r"):
            m = side_metrics(frames, fps, side)
            print(f"  {side}: rate={m['stroke_rate_cpm']} cpm, cycles={len(m['cycles'])}")
            for c in m["cycles"]:
                print("   ", c)
