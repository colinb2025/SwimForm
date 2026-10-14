"""
Posture snapshots: the athlete's own average stroke, drawn at the moment a finding is
about (the catch for a dropped elbow, peak roll for low rotation, ...), with the joints
that matter highlighted.

The drawing illustrates; the number in the caption is the report's measurement and
the ideal range comes from the workbook, so the picture never contradicts the finding.
All coordinates are numbers produced by the analysis, so the SVG is built here as a
string with no user-supplied text inside it.
"""
import math

import numpy as np

from swimform.schema import EDGES, HEAD_ON, IDX, KEYPOINTS, SIDE

W, H, PAD = 340, 210, 22


def _arr(frames):
    a = np.full((len(frames), len(KEYPOINTS), 2), np.nan)
    for t, frame in enumerate(frames):
        for i, pt in enumerate(frame):
            if pt is not None:
                a[t, i] = pt
    return a


def _near(a):
    """The arm the camera saw best."""
    r = (~np.isnan(a[:, IDX["r_wrist"], 0])).sum()
    l = (~np.isnan(a[:, IDX["l_wrist"], 0])).sum()
    return "r" if r >= l else "l"


def _p(a, t, k):
    return a[t, IDX[k]]


def _ok(*pts):
    return all(not np.isnan(p).any() for p in pts)


def _best(values):
    v = np.asarray(values, float)
    if np.isnan(v).all():
        return None
    return int(np.nanargmin(v))


# ---------------------------------------------------------------- phase pickers
# Each returns (bin, highlight-joint names, annotation) or None.
def _underwater(a, s):
    """Bins where the hand is genuinely down in the water, not just below a shoulder
    that is rolled up during the recovery. Coordinates are relative to the shoulder
    midpoint, so 'down' means a good fraction of an upper arm below it."""
    sh, el, wr = (a[:, IDX[f"{s}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    lengths = np.linalg.norm(el - sh, axis=-1)
    if np.isnan(lengths).all():
        return np.zeros(len(wr), bool)
    return wr[:, 1] > 0.35 * np.nanmedian(lengths)


def _catch(a):
    s = _near(a)
    sh, el, wr = (a[:, IDX[f"{s}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    d = wr - el
    fv = np.degrees(np.arctan2(np.abs(d[:, 0]), d[:, 1]))
    fv[~((wr[:, 0] > sh[:, 0]) & _underwater(a, s))] = np.nan   # hand ahead of the shoulder, in the water
    t = _best(fv)
    return None if t is None else (t, [f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"], ("below_horizontal", f"{s}_shoulder", f"{s}_elbow"))


def _mid_pull(a):
    s = _near(a)
    sh, wr = a[:, IDX[f"{s}_shoulder"]], a[:, IDX[f"{s}_wrist"]]
    gap = np.abs(wr[:, 0] - sh[:, 0])
    gap[~_underwater(a, s)] = np.nan
    t = _best(gap)
    return None if t is None else (t, [f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"], ("angle", f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"))


def _exit(a):
    s = _near(a)
    rel = a[:, IDX[f"{s}_wrist"], 0] - a[:, IDX[f"{s}_hip"], 0]
    t = _best(rel)
    return None if t is None else (t, [f"{s}_wrist", f"{s}_hip"], ("xgap", f"{s}_wrist", f"{s}_hip"))


def _recovery(a):
    s = _near(a)
    sh, wr = a[:, IDX[f"{s}_shoulder"]], a[:, IDX[f"{s}_wrist"]]
    gap = np.abs(wr[:, 0] - sh[:, 0])
    gap[~(wr[:, 1] < np.nanmedian(wr[:, 1]))] = np.nan
    t = _best(gap)
    return None if t is None else (t, [f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"], ("angle", f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"))


def _extension(a):
    """The moment both hands are furthest out in front of the head together."""
    head = a[:, IDX["top_of_head"], 0]
    both = np.fmin(a[:, IDX["l_wrist"], 0] - head, a[:, IDX["r_wrist"], 0] - head)
    t = _best(-both)
    return None if t is None else (t, ["l_wrist", "r_wrist", "top_of_head"], ("ahead", "top_of_head", "l_wrist", "r_wrist"))


def _head(a):
    lift = a[:, IDX["top_of_head"], 1] - a[:, IDX["spine_at_neck"], 1]
    if np.isnan(lift).all():
        return None
    t = int(np.nanargmin(np.abs(lift - np.nanmedian(lift))))
    return (t, ["top_of_head", "spine_at_neck", "spine_at_waist"], ("spine", "spine_at_waist", "spine_at_neck", "top_of_head"))


def _body_line(a):
    s = _near(a)
    d = a[:, IDX[f"{s}_ankle"]] - a[:, IDX[f"{s}_shoulder"]]
    ang = np.degrees(np.arctan2(d[:, 1], np.abs(d[:, 0])))
    if np.isnan(ang).all():
        ang = (a[:, IDX["spine_at_waist"], 1] - a[:, IDX["spine_at_neck"], 1])
        if np.isnan(ang).all():
            return None
        t = int(np.nanargmin(np.abs(ang - np.nanmedian(ang))))
        return (t, ["spine_at_neck", "spine_at_waist"], ("horizontal", "spine_at_neck", "spine_at_waist"))
    t = int(np.nanargmin(np.abs(ang - np.nanmedian(ang))))
    return (t, [f"{s}_shoulder", f"{s}_hip", f"{s}_ankle"], ("horizontal", f"{s}_shoulder", f"{s}_ankle"))


def _knee(a):
    s = _near(a)
    hip, kn, an = (a[:, IDX[f"{s}_{j}"]] for j in ("hip", "knee", "ankle"))
    v1, v2 = hip - kn, an - kn
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)), -1, 1)))
    t = _best(ang)
    return None if t is None else (t, [f"{s}_hip", f"{s}_knee", f"{s}_ankle"], ("angle", f"{s}_hip", f"{s}_knee", f"{s}_ankle"))


def _roll(a):
    d = a[:, IDX["l_shoulder"]] - a[:, IDX["r_shoulder"]]
    tilt = np.abs(np.degrees(np.arctan2(d[:, 1], d[:, 0])))
    t = _best(-tilt)
    return None if t is None else (t, ["l_shoulder", "r_shoulder"], ("horizontal", "r_shoulder", "l_shoulder"))


def _entry(a):
    best, pick = np.inf, None
    for s in ("r", "l"):
        wr, sh = a[:, IDX[f"{s}_wrist"]], a[:, IDX[f"{s}_shoulder"]]
        mid = a[:, IDX["spine_at_neck"], 0]
        inward = -(wr[:, 0] - sh[:, 0]) * np.sign(sh[:, 0] - mid)
        inward[~(wr[:, 1] > sh[:, 1])] = np.nan     # hand in the water, in front
        if np.isnan(inward).all():
            continue
        t = int(np.nanargmax(inward))
        if -inward[t] < best:
            best, pick = -inward[t], (t, [f"{s}_wrist", f"{s}_shoulder"], ("midline", f"{s}_wrist", f"{s}_shoulder"))
    return pick


def _breath(a):
    lat = np.abs(a[:, IDX["top_of_head"], 0] - a[:, IDX["spine_at_neck"], 0])
    t = _best(-lat)
    return None if t is None else (t, ["top_of_head", "spine_at_neck"], ("midline", "top_of_head"))


def _kick_width(a):
    sep = np.abs(a[:, IDX["l_ankle"], 0] - a[:, IDX["r_ankle"], 0])
    t = _best(-sep)
    return None if t is None else (t, ["l_ankle", "r_ankle", "l_hip", "r_hip"], ("xgap", "l_ankle", "r_ankle"))


# issue -> (view, picker, the moment in words)
SPECS = {
    "dropped_elbow": (SIDE, _catch, "at the catch"),
    "straight_arm_pull": (SIDE, _mid_pull, "mid-pull, hand under the shoulder"),
    "short_pull": (SIDE, _exit, "as the hand finishes the pull"),
    "high_head_position": (SIDE, _head, "on a typical stroke"),
    "sinking_hips": (SIDE, _body_line, "on a typical stroke"),
    "excessive_knee_bend": (SIDE, _knee, "at the deepest knee bend"),
    "over_gliding": (SIDE, _extension, "at full reach"),
    "wide_recovery_straight_arm": (SIDE, _recovery, "mid-recovery"),
    "crossover_entry": (HEAD_ON, _entry, "as the hand enters"),
    "low_body_rotation": (HEAD_ON, _roll, "at peak rotation"),
    "rotation_asymmetry": (HEAD_ON, _roll, "at peak rotation"),
    "late_breath": (HEAD_ON, _breath, "at the breath"),
    "head_lift_on_breath": (HEAD_ON, _breath, "at the breath"),
    "wide_kick": (HEAD_ON, _kick_width, "at the widest kick"),
}


# ---------------------------------------------------------------- drawing
def _fit(points):
    pts = np.array([p for p in points if not np.isnan(p).any()])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    k = min((W - 2 * PAD) / span[0], (H - 2 * PAD) / span[1])
    off = np.array([W, H]) / 2 - (lo + hi) / 2 * k
    return lambda p: p * k + off


def _f(x):
    return f"{x:.1f}"


def svg(average_strokes, issue_id):
    """-> (svg markup, moment description) or None if the joints weren't seen."""
    spec = SPECS.get(issue_id)
    if spec is None:
        return None
    view, picker, moment = spec
    frames = average_strokes.get(view)
    if not frames:
        return None
    a = _arr(frames)
    picked = picker(a)
    if picked is None:
        return None
    t, hot, note = picked
    frame = a[t]
    if np.isnan(frame).all() or any(np.isnan(frame[IDX[k]]).any() for k in hot):
        return None
    tf = _fit(frame)
    P = {k: tf(frame[IDX[k]]) for k in KEYPOINTS}
    hotset = set(hot)
    out = [f'<svg viewBox="0 0 {W} {H}" class="snap snap-{view}" role="img" aria-hidden="true">',
           f'<rect class="snap-bg" x="0" y="0" width="{W}" height="{H}" rx="6"/>']

    # annotation guides go under the body
    kind = note[0]
    if kind in ("vertical",):
        e = P[note[1]]
        out.append(f'<line class="snap-guide" x1="{_f(e[0])}" y1="{_f(e[1] - 8)}" x2="{_f(e[0])}" y2="{_f(min(H - 4, e[1] + 70))}"/>')
    if kind == "horizontal":
        p0 = P[note[1]]
        out.append(f'<line class="snap-guide" x1="8" y1="{_f(p0[1])}" x2="{W - 8}" y2="{_f(p0[1])}"/>')
    if kind == "spine":
        w, n = P[note[1]], P[note[2]]
        d = n - w
        if np.linalg.norm(d) > 0:
            far = n + d / np.linalg.norm(d) * 70
            out.append(f'<line class="snap-guide" x1="{_f(n[0])}" y1="{_f(n[1])}" x2="{_f(far[0])}" y2="{_f(far[1])}"/>')
    if kind == "midline" and not np.isnan(P["spine_at_neck"]).any():
        x = P["spine_at_neck"][0]
        out.append(f'<line class="snap-guide" x1="{_f(x)}" y1="6" x2="{_f(x)}" y2="{H - 6}"/>')
        if len(note) > 2:   # where the hand should go in: in line with the shoulder
            sx = P[note[2]][0]
            out.append(f'<line class="snap-target" x1="{_f(sx)}" y1="6" x2="{_f(sx)}" y2="{H - 6}"/>')
    if kind == "ahead" and not np.isnan(P[note[1]]).any():
        x = P[note[1]][0]
        out.append(f'<line class="snap-guide" x1="{_f(x)}" y1="6" x2="{_f(x)}" y2="{H - 6}"/>')
    if kind == "xgap":
        for k in note[1:]:
            x = P[k][0]
            out.append(f'<line class="snap-guide" x1="{_f(x)}" y1="6" x2="{_f(x)}" y2="{H - 6}"/>')

    for a_, b_ in EDGES:
        pa, pb = P[a_], P[b_]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        far = a_.startswith("l_") and b_.startswith("l_") and view == SIDE
        cls = "snap-bone hot" if (a_ in hotset and b_ in hotset) else ("snap-bone far" if far else "snap-bone")
        out.append(f'<line class="{cls}" x1="{_f(pa[0])}" y1="{_f(pa[1])}" x2="{_f(pb[0])}" y2="{_f(pb[1])}"/>')
    for k, p in P.items():
        if np.isnan(p).any():
            continue
        r = 6 if k == "top_of_head" else 3.2
        cls = "snap-joint hot" if k in hotset else "snap-joint"
        out.append(f'<circle class="{cls}" cx="{_f(p[0])}" cy="{_f(p[1])}" r="{r}"/>')

    # the measurement itself, drawn over the body
    def measure(p, q):
        if not (np.isnan(p).any() or np.isnan(q).any()):
            out.append(f'<line class="snap-measure" x1="{_f(p[0])}" y1="{_f(p[1])}" x2="{_f(q[0])}" y2="{_f(q[1])}"/>')
    if kind == "horizontal":
        measure(P[note[1]], P[note[2]])
    if kind == "xgap":
        p, q = P[note[1]], P[note[2]]
        y = np.nanmean([p[1], q[1]])
        measure(np.array([p[0], y]), np.array([q[0], y]))
    if kind == "ahead":
        for k in note[2:]:
            measure(np.array([P[note[1]][0], P[k][1]]), P[k])
    if kind == "midline" and not np.isnan(P["spine_at_neck"]).any():
        w = P[note[1]]
        measure(np.array([P["spine_at_neck"][0], w[1]]), w)

    if kind == "below_horizontal":
        # the upper arm's angle below a forward horizontal from the shoulder
        v, e = P[note[1]], P[note[2]]
        out.append(f'<line class="snap-guide" x1="{_f(v[0])}" y1="{_f(v[1])}" x2="{_f(min(W - 6, v[0] + 70))}" y2="{_f(v[1])}"/>')
        a2 = math.atan2(e[1] - v[1], e[0] - v[0])
        r = 26
        s_ = v + np.array([r, 0.0])
        e_ = v + r * np.array([math.cos(a2), math.sin(a2)])
        out.append(f'<path class="snap-arc" d="M{_f(s_[0])} {_f(s_[1])} A{r} {r} 0 0 1 {_f(e_[0])} {_f(e_[1])}"/>')

    if kind == "angle":
        p1, v, p2 = P[note[1]], P[note[2]], P[note[3]]
        a1 = math.atan2(p1[1] - v[1], p1[0] - v[0])
        a2 = math.atan2(p2[1] - v[1], p2[0] - v[0])
        sweep = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        r = 18
        s_ = v + r * np.array([math.cos(a1), math.sin(a1)])
        e_ = v + r * np.array([math.cos(a1 + sweep), math.sin(a1 + sweep)])
        out.append(f'<path class="snap-arc" d="M{_f(s_[0])} {_f(s_[1])} A{r} {r} 0 0 {1 if sweep > 0 else 0} {_f(e_[0])} {_f(e_[1])}"/>')
    out.append("</svg>")
    return "".join(out), moment
