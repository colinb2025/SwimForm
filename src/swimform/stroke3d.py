"""
A 3D average stroke, fused from a side-view clip and a head-on clip of different laps.

The two clips never show the same instant, so joints cannot be triangulated frame by
frame. Freestyle repeats, though, so each clip is cut into stroke cycles and averaged
into one typical cycle, indexed by phase (0 = cycle start, 1 = next start). The side
view supplies forward and depth (X, Y); the head-on view supplies lateral and depth
(Z, Y). Depth is seen by both, which is what lines them up:

  1. phase: the head-on cycle is shifted until its depth curves best match the side
     view's (the two clips' cycles start at different moments in the stroke);
  2. scale: a least-squares fit of head-on depth onto side depth converts head-on
     pixels into side pixels (the cameras sit at different distances);
  3. fuse: X from the side, Z from head-on, Y averaged from both, in torso lengths.

How well the depth curves agree after alignment (r^2) is reported with every 3D
measurement. Poor agreement means the two laps were swum differently, or a view is
misread, and the fused stroke is not trusted.

Accuracy limits: this is a TYPICAL stroke, so it suits faults that repeat every
cycle, not one-offs, and it assumes the cameras are far enough away that perspective
is roughly uniform (weak perspective).
"""
import math
import numpy as np

from . import metrics as M
from . import passes as P
from .posedata import PoseTrack
from .schema import HEAD_ON, IDX, KEYPOINTS, SIDE

N_BINS = 100
MIN_AGREEMENT = 0.8     # r^2 of the depth fit below which the fused stroke is not used
MIN_CYCLES = 3

FUSED_REGISTRY = {}     # metric_id -> fn(stroke) -> float | None


def fused_metric(metric_id):
    def deco(fn):
        FUSED_REGISTRY[metric_id] = fn
        return fn
    return deco


class Stroke3D:
    def __init__(self, points, agreement, shift, scale, cycles, available):
        self.points = points          # (N_BINS, 15, 3) torso lengths: X fwd, Y down, Z lateral (+left)
        self.agreement = agreement    # r^2 of the side-vs-head-on depth fit
        self.shift = shift            # head-on phase offset that best matched, in bins
        self.scale = scale            # side px per head-on px
        self.cycles = cycles          # {"side": n, "head_on": n} cycles averaged
        self.available = available    # (15,) share of bins with a full 3D position per joint

    def p(self, name):
        return self.points[:, IDX[name]]

    @property
    def trusted(self):
        return self.agreement >= MIN_AGREEMENT and min(self.cycles.values()) >= MIN_CYCLES


def _origin(tr):
    """Shoulder midpoint per frame (falls back to the neck)."""
    o = 0.5 * (tr.p("l_shoulder") + tr.p("r_shoulder"))
    neck = tr.p("spine_at_neck")
    bad = np.isnan(o[:, 0])
    o[bad] = neck[bad]
    return o


def _resample(series, cycles):
    """(T, K, 2) -> (cycles, N_BINS, K, 2) by linear interpolation in phase."""
    out = np.full((len(cycles), N_BINS) + series.shape[1:], np.nan)
    for c, (a, b) in enumerate(cycles):
        pos = a + (b - a) * np.arange(N_BINS) / N_BINS
        i0 = np.floor(pos).astype(int)
        f = (pos - i0)[:, None, None]
        i1 = np.minimum(i0 + 1, len(series) - 1)
        out[c] = series[i0] * (1 - f) + series[i1] * f   # NaN if either neighbour missing
    return out


def mean_cycle(tr: PoseTrack):
    """
    Typical stroke cycle for one view: (N_BINS, 15, 2) in pixels relative to the shoulder
    midpoint, horizontal axis = forward (side) or lateral, +swimmer's left (head-on).
    Also returns the cycle count and the side view's torso length in pixels.

    A side clip with a turn in it (down and back) is averaged pass by pass, then the
    passes are lined up and combined, each arm and leg taken from the passes where it
    faced the camera.
    """
    if tr.view == SIDE:
        ps = P.split(tr)
        if not ps:
            return None, 0, None
        if len(ps) > 1:
            return _combine_passes(ps)
    return _single_mean_cycle(tr)


def _shift_to(ref, other):
    """Circular shift (bins) that best lines `other`'s cycle up with `ref`'s, by mean
    squared distance over every joint both have. Each pass's cycles start at its own
    near arm's pull, so a pass seen from the other side is about half a cycle out."""
    best, best_err = 0, np.inf
    for d in range(N_BINS):
        diff = ref - np.roll(other, -d, axis=0)
        ok = ~np.isnan(diff).any(axis=-1)
        if ok.sum() < 0.3 * ok.size:
            continue
        err = float((diff[ok] ** 2).sum(-1).mean())
        if err < best_err:
            best, best_err = d, err
    return best


def _combine_passes(ps):
    means = []
    for p in ps:
        med, n, torso = _single_mean_cycle(p.track)
        if med is not None:
            means.append((p, med, n, torso))
    if not means:
        return None, 0, None
    if len(means) == 1:
        return means[0][1:]
    ref = max(means, key=lambda m: m[2])[1]
    aligned = [(p, np.roll(med, -_shift_to(ref, med), axis=0)) for p, med, _, _ in means]
    stack = np.stack([a for _, a in aligned])                       # (passes, N_BINS, 15, 2)
    out = np.full(ref.shape, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, name in enumerate(KEYPOINTS):
            use = [i for i, (p, _) in enumerate(aligned)
                   if name[:2] not in ("l_", "r_") or p.near == name[0]]
            if not use or np.isnan(stack[use, :, k, 0]).all():
                use = list(range(len(aligned)))                      # that side was never near
            out[:, k] = np.nanmedian(stack[use, :, k], axis=0)
        torsos = [t for *_, t in means if t and not math.isnan(t)]
    return out, sum(m[2] for m in means), (float(np.median(torsos)) if torsos else None)


def _single_mean_cycle(tr: PoseTrack):
    ctx = M.Context(tr)
    if len(ctx.cycles) == 0:
        return None, 0, None
    o = _origin(ctx.tr)
    rel = np.stack([ctx.tr.p(k) - o for k in KEYPOINTS], axis=1)   # (T, 15, 2)
    if tr.view == SIDE:
        rel[..., 0] *= ctx.sign
    cyc = _resample(rel, ctx.cycles)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(cyc, axis=0)
            enough = (~np.isnan(cyc[..., 0])).sum(axis=0) >= max(2, len(ctx.cycles) // 2)
    med[~enough] = np.nan
    torso = getattr(ctx, "torso", None)
    return med, len(ctx.cycles), torso


def _align(ys, yh):
    """Best circular shift of head-on depth curves onto side ones: (shift, a, c, r2)."""
    best = (0, 1.0, 0.0, -np.inf)
    for d in range(N_BINS):
        h = np.roll(yh, -d, axis=0)
        ok = ~np.isnan(ys) & ~np.isnan(h)
        if ok.sum() < 0.3 * ys.size:
            continue
        x, y = h[ok], ys[ok]
        A = np.column_stack([x, np.ones_like(x)])
        (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        if a <= 0:
            continue
        res = y - (a * x + c)
        r2 = 1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        if r2 > best[3]:
            best = (d, float(a), float(c), float(r2))
    return best


# Joints whose depth swings enough over the stroke to pin down the phase.
_ALIGN_JOINTS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "l_shoulder", "r_shoulder")


def fuse(side: PoseTrack, head_on: PoseTrack):
    """Fuse the two views into a Stroke3D, or return (None, reason)."""
    if side.view != SIDE or head_on.view != HEAD_ON:
        raise ValueError("fuse(side_track, head_on_track)")
    ms, ns, torso = mean_cycle(side)
    mh, nh, _ = mean_cycle(head_on)
    if ms is None or mh is None:
        return None, "no stroke cycles in one of the views"
    if not torso or math.isnan(torso):
        return None, "torso length not measurable in the side view"
    idx = [IDX[k] for k in _ALIGN_JOINTS]
    d, a, c, r2 = _align(ms[:, idx, 1], mh[:, idx, 1])
    if not np.isfinite(r2):
        return None, "too few joints seen in both views to line them up"
    mh = np.roll(mh, -d, axis=0)
    y_side = ms[..., 1]
    y_head = a * mh[..., 1] + c
    Y = np.where(np.isnan(y_side), y_head, np.where(np.isnan(y_head), y_side, 0.5 * (y_side + y_head)))
    X = ms[..., 0]
    Z = a * mh[..., 0]
    pts = np.stack([X, Y, Z], axis=-1) / torso
    available = (~np.isnan(pts).any(axis=-1)).mean(axis=0)
    return Stroke3D(pts, round(r2, 3), d, round(a, 4), {SIDE: ns, HEAD_ON: nh}, available), None


def angle3d(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / n, -1, 1)))


def _mid_recovery_bin(st, s):
    """Bin where the hand passes the shoulder above the water (the recovery half)."""
    w, sh = st.p(f"{s}_wrist"), st.p(f"{s}_shoulder")
    if np.isnan(w).any(axis=1).mean() > 0.5:
        return None
    high = w[:, 1] < np.nanmedian(w[:, 1])           # the above-water half of the stroke
    rel = np.abs(w[:, 0] - sh[:, 0])
    rel[~high | np.isnan(rel)] = np.inf
    return None if not np.isfinite(rel.min()) else int(np.argmin(rel))


@fused_metric("recovery_elbow_angle")
def recovery_elbow_angle(st):
    """True 3D interior elbow angle at mid-recovery, averaged over the arms measured."""
    vals = []
    for s in ("r", "l"):
        i = _mid_recovery_bin(st, s)
        if i is None:
            continue
        v = angle3d(st.p(f"{s}_shoulder")[i], st.p(f"{s}_elbow")[i], st.p(f"{s}_wrist")[i])
        if not math.isnan(v):
            vals.append(float(v))
    return round(float(np.mean(vals)), 3) if vals else None


def compute(side: PoseTrack, head_on: PoseTrack):
    """-> {"stroke": Stroke3D | None, "reason": str | None, "values": {metric_id: v | None}}"""
    st, reason = fuse(side, head_on)
    values = {}
    if st is not None and st.trusted:
        for mid, fn in FUSED_REGISTRY.items():
            try:
                values[mid] = fn(st)
            except Exception:
                values[mid] = None
    elif st is not None:
        reason = (f"side and head-on clips do not line up well enough "
                  f"(agreement r²={st.agreement:.2f}, need {MIN_AGREEMENT})"
                  if st.agreement < MIN_AGREEMENT else "too few stroke cycles to average")
    return {"stroke": st, "reason": reason, "values": values}
