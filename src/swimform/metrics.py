"""
Pose track -> stroke cycles -> the measurements defined in rules/swim_rules.xlsx.

Every metric_id in the Metrics sheet has a function here, registered for the view
the workbook says it is measured from. Each returns one value per stroke cycle, or
None for a cycle where the joints it needs were not trusted. Aggregation across
cycles, and the decision that too few cycles were readable to say anything, happen
in `swimform.analyze`: a metric with no trustworthy cycles is reported as not
assessed rather than guessed.

  python -m swimform.metrics out/clip_pose.json --view side

Geometry is in pixels, which keeps angles true. Side-view "forward" is the swimmer's
direction of travel, inferred from the spine, so a swimmer moving left or right
measures the same.
"""
import argparse, math
import numpy as np

from .posedata import PoseTrack
from .schema import HEAD_ON, SIDE

MIN_CYCLE_S, MAX_CYCLE_S = 0.6, 2.5
MIN_PULL_VISIBILITY = 0.7  # share of frames the near hand must be trusted to measure the pull

REGISTRY = {}  # metric_id -> (view, fn(ctx, cycle) -> float | None | SKIP)

# A metric returns SKIP for a cycle where it does not apply (no breath taken in that
# cycle), as opposed to None for a cycle it could not read. Skipped cycles do not
# count against readability: breathing every other cycle is technique, not bad footage.
SKIP = "skip"

BREATH_MIN_TURN = 0.15  # sideways head shift, as a share of shoulder width, that counts as a breath


def metric(metric_id, view):
    def deco(fn):
        REGISTRY[metric_id] = (view, fn)
        return fn
    return deco


# ---------------------------------------------------------------- helpers
def _fill(y, min_valid=0.4):
    """Interpolate NaN gaps; None if too little of the signal is present."""
    ok = ~np.isnan(y)
    if ok.sum() < max(4, min_valid * len(y)):
        return None
    i = np.arange(len(y))
    out = y.copy()
    out[~ok] = np.interp(i[~ok], i[ok], y[ok])
    return out


def _smooth(y, k):
    k = max(3, int(k)) | 1
    pad = np.pad(y, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _peaks(y, half_window):
    """Indices that are the maximum of their +/- half_window neighbourhood."""
    hw = max(1, int(half_window))
    out = []
    for i in range(len(y)):
        lo, hi = max(0, i - hw), min(len(y), i + hw + 1)
        if y[i] >= y[lo:hi].max() and (not out or i - out[-1] > hw):
            out.append(i)
    return out


def _merge_flat_peaks(y, peaks, min_dip=0.3):
    """Drop a peak unless the signal dips well below the previous kept peak first.

    A long glide holds the hand at full reach, which reads as several maxima on
    one plateau; without this, one stroke cycle is counted as two.
    """
    if not peaks:
        return peaks
    rng = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    kept = [peaks[0]]
    for j in peaks[1:]:
        if np.min(y[kept[-1]:j + 1]) < y[kept[-1]] - min_dip * rng:
            kept.append(j)
        elif y[j] > y[kept[-1]]:
            kept[-1] = j
    return kept


def _cycles_from_peaks(peaks, fps):
    return [(a, b) for a, b in zip(peaks, peaks[1:]) if MIN_CYCLE_S <= (b - a) / fps <= MAX_CYCLE_S]


def interior_angle(a, b, c):
    """Angle at b in degrees; NaN if any point is missing."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / n, -1, 1)))


def _val(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 3)


def _nanmed(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _clean_track(xy, fps, smooth_s=0.15, max_gap_s=0.2):
    """Bridge short dropouts and lightly smooth one keypoint's (T, 2) trajectory.

    Extremes (peak roll, deepest knee bend, highest head) are read off these
    trajectories; on raw per-frame keypoints the extreme is mostly jitter.
    Long gaps stay missing: nothing is invented across them.
    """
    out = xy.copy()
    n = len(xy)
    ok = ~np.isnan(xy[:, 0])
    if ok.sum() < 2:
        return out
    idx = np.arange(n)
    gap = int(round(max_gap_s * fps))
    fill = ~ok
    # only fill interior gaps no longer than `gap`
    i = 0
    while i < n:
        if fill[i]:
            j = i
            while j + 1 < n and fill[j + 1]:
                j += 1
            if i == 0 or j == n - 1 or (j - i + 1) > gap:
                fill[i:j + 1] = False
            i = j + 1
        else:
            i += 1
    for d in (0, 1):
        out[fill, d] = np.interp(idx[fill], idx[ok], xy[ok, d])
    k = max(3, int(round(smooth_s * fps))) | 1
    w = np.ones(k)
    have = ~np.isnan(out[:, 0])
    for d in (0, 1):
        v = np.where(have, out[:, d], 0.0)
        num = np.convolve(v, w, mode="same")
        den = np.convolve(have.astype(float), w, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, d] = np.where(have, num / den, np.nan)
    return out


class _CleanView:
    """Duck-types the parts of PoseTrack the metrics use, on cleaned trajectories."""

    def __init__(self, tr):
        self._tr, self.n, self.view = tr, tr.n, tr.view
        self._cache = {}

    def p(self, name):
        if name not in self._cache:
            self._cache[name] = _clean_track(self._tr.p(name), self._tr.fps)
        return self._cache[name]

    def ok(self, *names):
        m = np.ones(self.n, bool)
        for k in names:
            m &= ~np.isnan(self.p(k)[:, 0])
        return m

    def detection_rates(self):
        return self._tr.detection_rates()


class Context:
    """Per-clip derived quantities shared by the metric functions."""

    def __init__(self, tr: PoseTrack):
        self.raw, self.fps = tr, tr.fps
        self.tr = _CleanView(tr)
        self.cycles = []
        self.anchor = None
        if tr.view == SIDE:
            self._init_side()
        else:
            self._init_head_on()

    # -- side view: the camera sees one arm clearly (the near side)
    def _init_side(self):
        tr = self.tr
        spine = tr.p("spine_at_neck") - tr.p("spine_at_waist")
        sx = _nanmed(spine[:, 0])
        if math.isnan(sx):  # fall back to shoulders vs hips
            sx = _nanmed(tr.p("r_shoulder")[:, 0] - tr.p("r_hip")[:, 0])
        self.sign = 1.0 if (math.isnan(sx) or sx >= 0) else -1.0   # +1: travelling right
        rates = tr.detection_rates()
        self.s = "r" if rates["r_wrist"] + rates["r_elbow"] >= rates["l_wrist"] + rates["l_elbow"] else "l"
        self.torso = _nanmed(np.linalg.norm(spine, axis=-1))
        if math.isnan(self.torso):
            self.torso = _nanmed(np.linalg.norm(tr.p(f"{self.s}_shoulder") - tr.p(f"{self.s}_hip"), axis=-1))
        self.upper = _nanmed(np.linalg.norm(tr.p(f"{self.s}_elbow") - tr.p(f"{self.s}_shoulder"), axis=-1))
        reach = self.fwd(f"{self.s}_wrist", f"{self.s}_shoulder")
        # Only trust the full-stroke rhythm if the hand is genuinely seen through
        # most of the stroke; interpolating across a hidden pull invents it.
        seen = self.raw.ok(f"{self.s}_wrist", f"{self.s}_shoulder").mean()
        sig = _fill(reach) if seen >= MIN_PULL_VISIBILITY else None
        if sig is None:
            self._init_side_from_recoveries()
            return
        self.anchor = "pull"
        sm = _smooth(sig, self.fps * 0.1)
        self.reach_sm = sm
        self.reach_range = float(sm.max() - sm.min()) or 1.0
        peaks = _merge_flat_peaks(sm, _peaks(sm, self.fps * MIN_CYCLE_S / 2))
        # Anchor each cycle where the pull begins (the end of the full-reach plateau):
        # that instant is sharp, whereas the maximum wanders along a long glide.
        self.cycles = _cycles_from_peaks([self._plateau_end(p) for p in peaks], self.fps)

    def _init_side_from_recoveries(self):
        """Deck footage: the hand is only trusted above water. Each run of visible
        hand frames is one recovery, so cycles run exit-to-exit. Pull metrics then
        have nothing to measure and come back not assessed; above-water ones survive."""
        w = self.tr.ok(f"{self.s}_wrist", f"{self.s}_shoulder")
        starts, i, n = [], 0, len(w)
        min_run = max(3, int(0.15 * self.fps))
        while i < n:
            if w[i]:
                j = i
                while j + 1 < n and w[j + 1]:
                    j += 1
                if j - i + 1 >= min_run and 0 < i:   # skip a run cut off by the clip start
                    starts.append(i)
                i = j + 1
            else:
                i += 1
        self.anchor = "exit"
        self.cycles = _cycles_from_peaks(starts, self.fps)

    def _plateau_end(self, i):
        sm, peak = self.reach_sm, self.reach_sm[i]
        while i + 1 < len(sm) and sm[i + 1] >= peak - 0.05 * self.reach_range:
            i += 1
        return i

    def fwd(self, name, ref):
        """Forward (direction-of-travel) offset of `name` from `ref`, pixels."""
        return (self.tr.p(name)[:, 0] - self.tr.p(ref)[:, 0]) * self.sign

    def phases(self, cyc):
        """(pull start, hand exit) frame indices within a side-view cycle."""
        a, b = cyc
        if self.anchor == "exit":
            return None, a
        start = min(self._plateau_end(a), b - 1)
        exit_ = start + int(np.argmin(self.reach_sm[start:b]))
        return start, exit_

    # -- head-on view: shoulder roll gives the rhythm
    def _init_head_on(self):
        tr = self.tr
        self.tilt = self.line_tilt("r_shoulder", "l_shoulder")
        self.width = _nanmed(np.abs(tr.p("l_shoulder")[:, 0] - tr.p("r_shoulder")[:, 0]))
        sig = _fill(self.tilt)
        if sig is None:
            return
        sm = _smooth(sig, self.fps * 0.1)
        self.cycles = _cycles_from_peaks(_merge_flat_peaks(sm, _peaks(sm, self.fps * MIN_CYCLE_S / 2)), self.fps)

    def line_tilt(self, r, l):
        """Tilt of the r->l line from horizontal, degrees; positive = right side deeper."""
        a, b = self.tr.p(r), self.tr.p(l)
        return np.degrees(np.arctan2(a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]))


# ============================================================ side metrics
def _forearm_from_vertical(ctx, idx):
    s = ctx.s
    e, w = ctx.tr.p(f"{s}_elbow")[idx], ctx.tr.p(f"{s}_wrist")[idx]
    d = w - e
    return np.degrees(np.arctan2(np.abs(d[..., 0]), d[..., 1]))   # 0 = hand straight below elbow


def _pull_window(ctx, cyc, lo, hi):
    start, exit_ = ctx.phases(cyc)
    if start is None:
        return None
    n = exit_ - start
    if n < 3:
        return None
    return np.arange(start + int(lo * n), start + max(int(hi * n), int(lo * n) + 1))


@metric("catch_forearm_angle", SIDE)
def catch_forearm_angle(ctx, cyc):
    """Best (most vertical) forearm angle early in the pull, degrees from vertical."""
    w = _pull_window(ctx, cyc, 0.0, 0.4)
    if w is None:
        return None
    v = _forearm_from_vertical(ctx, w)
    return None if np.isnan(v).mean() > 0.5 else _val(np.nanmin(v))


@metric("catch_upper_arm_angle", SIDE)
def catch_upper_arm_angle(ctx, cyc):
    """Shoulder->elbow angle below horizontal when the forearm first reaches vertical."""
    w = _pull_window(ctx, cyc, 0.0, 0.75)
    if w is None:
        return None
    fv = _forearm_from_vertical(ctx, w)
    if np.isnan(fv).mean() > 0.5:
        return None
    hit = np.where(fv <= 10)[0]
    i = w[hit[0]] if len(hit) else w[int(np.nanargmin(fv))]
    s = ctx.s
    d = ctx.tr.p(f"{s}_elbow")[i] - ctx.tr.p(f"{s}_shoulder")[i]
    return _val(np.degrees(np.arctan2(d[1], d[0] * ctx.sign)))


@metric("mid_pull_elbow_angle", SIDE)
def mid_pull_elbow_angle(ctx, cyc):
    """Interior elbow angle when the hand passes under the shoulder."""
    start, exit_ = ctx.phases(cyc)
    if start is None:
        return None
    s = ctx.s
    rel = ctx.fwd(f"{s}_wrist", f"{s}_shoulder")[start:exit_ + 1]
    if np.isnan(rel).all():
        return None
    i = start + int(np.nanargmin(np.abs(rel)))
    ang = interior_angle(ctx.tr.p(f"{s}_shoulder")[i], ctx.tr.p(f"{s}_elbow")[i], ctx.tr.p(f"{s}_wrist")[i])
    return _val(ang)


@metric("pull_exit_position", SIDE)
def pull_exit_position(ctx, cyc):
    """(hand - hip) along the direction of travel at exit / torso; positive = past the hip."""
    if ctx.anchor != "pull":
        return None
    _, exit_ = ctx.phases(cyc)
    s = ctx.s
    lo, hi = max(cyc[0], exit_ - 1), min(cyc[1], exit_ + 2)
    x = _nanmed(ctx.fwd(f"{s}_wrist", f"{s}_hip")[lo:hi])
    return _val(-x / ctx.torso)


def _mid_recovery(ctx, cyc):
    _, exit_ = ctx.phases(cyc)
    s = ctx.s
    rel = ctx.fwd(f"{s}_wrist", f"{s}_shoulder")[exit_:cyc[1]]
    if len(rel) < 3 or np.isnan(rel).all():
        return None
    return exit_ + int(np.nanargmin(np.abs(rel)))


@metric("recovery_elbow_angle", SIDE)
def recovery_elbow_angle(ctx, cyc):
    i = _mid_recovery(ctx, cyc)
    if i is None:
        return None
    s = ctx.s
    return _val(interior_angle(ctx.tr.p(f"{s}_shoulder")[i], ctx.tr.p(f"{s}_elbow")[i], ctx.tr.p(f"{s}_wrist")[i]))


@metric("recovery_elbow_height", SIDE)
def recovery_elbow_height(ctx, cyc):
    """(hand depth - elbow depth) / upper-arm length at mid-recovery; positive = elbow higher."""
    i = _mid_recovery(ctx, cyc)
    if i is None:
        return None
    s = ctx.s
    return _val((ctx.tr.p(f"{s}_wrist")[i, 1] - ctx.tr.p(f"{s}_elbow")[i, 1]) / ctx.upper)


@metric("head_lift_angle", SIDE)
def head_lift_angle(ctx, cyc):
    """Neck->top-of-head line vs waist->neck spine line, degrees; positive = head raised.

    The workbook asks for non-breathing strokes, but breaths are invisible from the
    side without face points, so this takes the median over the whole cycle. With one
    breath per cycle the breath frames are a minority and the median ignores them.
    """
    a, b = cyc
    tr = ctx.tr
    head = tr.p("top_of_head")[a:b] - tr.p("spine_at_neck")[a:b]
    spine = tr.p("spine_at_neck")[a:b] - tr.p("spine_at_waist")[a:b]
    ang = lambda v: np.degrees(np.arctan2(-v[:, 1], v[:, 0] * ctx.sign))
    d = ang(head) - ang(spine)
    if np.isnan(d).mean() > 0.5:
        return None
    return _val(np.nanmedian(d))


@metric("body_line_angle", SIDE)
def body_line_angle(ctx, cyc):
    """Shoulder->ankle line below horizontal, degrees."""
    a, b = cyc
    s = ctx.s
    d = ctx.tr.p(f"{s}_ankle")[a:b] - ctx.tr.p(f"{s}_shoulder")[a:b]
    v = np.degrees(np.arctan2(d[:, 1], np.abs(d[:, 0])))
    return None if np.isnan(v).mean() > 0.5 else _val(np.nanmedian(v))


@metric("kick_knee_angle", SIDE)
def kick_knee_angle(ctx, cyc):
    """Smallest interior knee angle in the cycle (deepest knee bend), near leg."""
    a, b = cyc
    s = ctx.s
    v = interior_angle(ctx.tr.p(f"{s}_hip")[a:b], ctx.tr.p(f"{s}_knee")[a:b], ctx.tr.p(f"{s}_ankle")[a:b])
    if np.isnan(v).mean() > 0.5:
        return None
    return _val(np.nanpercentile(v, 5))  # 5th percentile: deepest bend without trusting one noisy frame


@metric("glide_fraction", SIDE)
def glide_fraction(ctx, cyc):
    """Share of the cycle with both hands ahead of the head."""
    a, b = cyc
    tr = ctx.tr
    m = tr.ok("l_wrist", "r_wrist", "top_of_head")[a:b]
    if m.mean() < 0.6:
        return None
    lw = ctx.fwd("l_wrist", "top_of_head")[a:b]
    rw = ctx.fwd("r_wrist", "top_of_head")[a:b]
    both = (lw > 0) & (rw > 0)
    return _val(both[m].mean())


@metric("stroke_rate", SIDE)
def stroke_rate(ctx, cyc):
    return _val(60.0 * ctx.fps / (cyc[1] - cyc[0]))


# ========================================================== head-on metrics
def _peak_roll(ctx, cyc, r, l):
    a, b = cyc
    t = ctx.line_tilt(r, l)[a:b]
    if np.isnan(t).mean() > 0.5:
        return None
    return float(np.nanmax(t)), float(-np.nanmin(t))   # toward right side, toward left side


@metric("shoulder_rotation", HEAD_ON)
def shoulder_rotation(ctx, cyc):
    p = _peak_roll(ctx, cyc, "r_shoulder", "l_shoulder")
    return None if p is None else _val(np.mean(p))


@metric("hip_rotation", HEAD_ON)
def hip_rotation(ctx, cyc):
    p = _peak_roll(ctx, cyc, "r_hip", "l_hip")
    return None if p is None else _val(np.mean(p))


@metric("rotation_asymmetry", HEAD_ON)
def rotation_asymmetry(ctx, cyc):
    p = _peak_roll(ctx, cyc, "r_shoulder", "l_shoulder")
    return None if p is None else _val(abs(p[0] - p[1]))


def _entry_offsets(ctx, cyc):
    """Per arm: lateral offset toward the midline at hand entry / shoulder width."""
    tr = ctx.tr
    a, b = cyc
    mid = tr.p("spine_at_neck")[:, 0]
    out = []
    for s in ("r", "l"):
        w, sh = tr.p(f"{s}_wrist"), tr.p(f"{s}_shoulder")
        y = w[a:b, 1]
        if np.isnan(y).mean() > 0.8:   # deck footage may only show the recovery; that is enough
            continue
        top = a + int(np.nanargmin(y))            # highest point of the recovery
        end = min(len(w) - 1, top + int(0.45 * (b - a)))
        seg = w[top:end + 1]
        if len(seg) < 4:
            continue
        seen = ~np.isnan(seg[:, 0])
        run = int(np.argmin(seen)) if not seen.all() else len(seg)   # visible frames before the hand vanishes
        if run < 3:
            continue
        sp = np.linalg.norm(np.diff(seg[:run], axis=0), axis=1)
        slow = np.where(sp < 0.25 * sp.max())[0] if sp.max() > 0 else np.array([0])
        # Entry: the hand stops travelling, or (deck camera) its last visible frame.
        i = top + (slow[0] + 1 if len(slow) else run - 1)
        away = np.sign(sh[i, 0] - mid[i])          # direction away from the midline
        if np.isnan(away) or np.isnan(w[i, 0]) or ctx.width <= 0:
            continue
        out.append(-(w[i, 0] - sh[i, 0]) * away / ctx.width)
    return out


@metric("entry_crossover", HEAD_ON)
def entry_crossover(ctx, cyc):
    v = _entry_offsets(ctx, cyc)
    return _val(max(v)) if v else None


def _breaths(ctx):
    """Breath events (peak frame, side 'r'/'l', first frame, last frame), from the
    head turning sideways off the spine.

    The resting offset is subtracted first, so a camera slightly off the swimmer's
    centreline does not read as permanent breathing, and a turn must clear an absolute
    floor, so a clip with no breaths does not promote jitter into breaths.
    """
    if hasattr(ctx, "_breath_cache"):
        return ctx._breath_cache
    tr = ctx.tr
    lat = tr.p("top_of_head")[:, 0] - tr.p("spine_at_neck")[:, 0]
    events = []
    if np.isnan(lat).mean() < 0.5 and ctx.width > 0:
        lat = lat - _nanmed(lat)
        thr = max(0.4 * np.nanmax(np.abs(lat)), BREATH_MIN_TURN * ctx.width)
        on = np.abs(np.nan_to_num(lat)) > thr
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                k = i + int(np.nanargmax(np.abs(lat[i:j + 1])))
                # Head-on, the swimmer's right appears on image-left. (A mirrored
                # front-camera recording would swap this; the timing logic still works,
                # since it follows whichever wrist is on the breathing side.)
                events.append((k, "r" if lat[k] < 0 else "l", i, j))
                i = j + 1
            else:
                i += 1
    ctx._breath_cache = events
    return events


def _breath_in(ctx, cyc):
    return [e for e in _breaths(ctx) if cyc[0] <= e[0] < cyc[1]]


def _head_tracked(ctx, cyc):
    a, b = cyc
    return ctx.tr.ok("top_of_head", "spine_at_neck")[a:b].mean() >= 0.5


@metric("breath_head_lift", HEAD_ON)
def breath_head_lift(ctx, cyc):
    """Rise of the head during the breath vs non-breathing frames / head height."""
    tr = ctx.tr
    if not _head_tracked(ctx, cyc):
        return None
    events = _breath_in(ctx, cyc)
    if not events:
        return SKIP
    rel = tr.p("top_of_head")[:, 1] - tr.p("spine_at_neck")[:, 1]
    breathing = np.zeros(tr.n, bool)
    for _, _, i, j in _breaths(ctx):
        breathing[i:j + 1] = True
    base = _nanmed(rel[~breathing])
    head_h = abs(base)
    if math.isnan(base) or head_h == 0:
        return None
    k, _, i, j = events[0]
    during = rel[i:j + 1]
    if np.isnan(during).all():
        return None
    return _val(max(0.0, base - np.nanmin(during)) / head_h)


@metric("breath_timing", HEAD_ON)
def breath_timing(ctx, cyc):
    """(peak head turn - same-side hand exit) / cycle length; positive = breath late."""
    tr = ctx.tr
    if not _head_tracked(ctx, cyc):
        return None
    events = _breath_in(ctx, cyc)
    if not events:
        return SKIP
    k, side, _, _ = events[0]
    above = tr.p(f"{side}_wrist")[:, 1] < tr.p(f"{side}_shoulder")[:, 1]
    known = tr.ok(f"{side}_wrist", f"{side}_shoulder")
    rises = [i for i in range(1, tr.n) if above[i] and not above[i - 1] and known[i] and known[i - 1]]
    L = cyc[1] - cyc[0]
    near = [i for i in rises if abs(i - k) <= 0.5 * L]
    if not near:
        return None
    exit_ = min(near, key=lambda i: abs(i - k))
    return _val((k - exit_) / L)


@metric("kick_width", HEAD_ON)
def kick_width(ctx, cyc):
    a, b = cyc
    tr = ctx.tr
    sep = np.abs(tr.p("l_ankle")[a:b, 0] - tr.p("r_ankle")[a:b, 0])
    hips = _nanmed(np.abs(tr.p("l_hip")[:, 0] - tr.p("r_hip")[:, 0]))
    if np.isnan(sep).mean() > 0.5 or not hips or math.isnan(hips):
        return None
    return _val(np.nanmax(sep) / hips)


# ================================================================ driver
def compute(tr: PoseTrack, metric_ids=None):
    """
    Per-cycle values for every registered metric belonging to this track's view.

    -> {"view", "cycles": [(a, b), ...],
        "values": {metric_id: [value | None (unreadable) | SKIP (not applicable) per cycle]},
        "errors": {metric_id: message} for any metric that raised}
    """
    ctx = Context(tr)
    values, errors = {}, {}
    for mid, (view, fn) in REGISTRY.items():
        if view != tr.view or (metric_ids is not None and mid not in metric_ids):
            continue
        out = []
        for c in ctx.cycles:
            try:
                out.append(fn(ctx, c))
            except Exception as e:  # one odd clip must not take down the other metrics
                errors.setdefault(mid, f"{type(e).__name__}: {e}")
                out.append(None)
        values[mid] = out
    return {"view": tr.view, "fps": tr.fps,
            "cycles": [[int(a), int(b)] for a, b in ctx.cycles], "values": values, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_json", nargs="+")
    ap.add_argument("--view", choices=[SIDE, HEAD_ON], help="override summary['view']")
    a = ap.parse_args()
    for p in a.pose_json:
        tr = PoseTrack.from_json(p, view=a.view)
        r = compute(tr)
        print("=" * 70)
        print(p, f"view={tr.view}  frames={tr.n}  cycles={len(r['cycles'])}")
        for mid, vals in r["values"].items():
            got = [v for v in vals if v is not None and v != SKIP]
            med = round(float(np.median(got)), 2) if got else None
            print(f"  {mid:24s} median={med}  usable={len(got)}/{len(vals)}")
        for mid, err in r["errors"].items():
            print(f"  ! {mid} failed: {err}")


if __name__ == "__main__":
    main()
