"""
Split a side-view clip into passes: stretches of steady swimming in one direction.

Why: a side camera on the deck sees the arm nearest to it clearly and the far arm
partly hidden behind the body. Film a 50 (down the pool, turn, back) and the swimmer
shows one side on the way down and the other side on the way back, at the same lane
distance. Splitting the clip at the turn gives each arm its own near-side view, so
each arm is measured from the pass where the camera could actually see it.

Direction is read from the body, not assumed: the spine (low back -> neck) points the
way the swimmer is travelling. The turn itself (wall, push-off, underwater breakout)
is dropped, with a margin either side, because it is not steady swimming.

Which side faces the camera is read from how well each arm is seen (the near arm is
tracked more reliably). The camera stays on one deck, so the near side has to flip
when the direction flips; passes where the visibility evidence is weak take their
near side from that rule, using the passes where it is strong.

A clip with no turn comes back as one pass covering the whole clip, labelled with its
near side when the visibility evidence is clear.
"""
from dataclasses import dataclass

import numpy as np

from .posedata import PoseTrack
from .schema import IDX, SIDE

VOTE_WINDOW_S = 1.0     # majority vote over this window settles the per-frame direction
BRIDGE_S = 0.6          # unreadable stretches shorter than this don't break a pass
TRIM_BEFORE_S = 0.5     # steady swimming ends this long before a turn...
TRIM_AFTER_S = 1.5      # ...and starts this long after it (push-off and breakout)
MIN_PASS_S = 3.0        # a pass next to a turn shorter than this is dropped
STRONG_MARGIN = 0.08    # near-side minus far-side mean confidence that settles the near side alone


@dataclass
class Pass:
    start: int          # first frame in the clip (inclusive)
    end: int            # last frame + 1
    direction: int      # +1 travelling image-right, -1 image-left, 0 unknown
    near: str           # "r" or "l": the swimmer's side facing the camera
    near_from: str      # "visibility" | "direction" (inferred from the other passes) | "default"
    track: PoseTrack    # the frames of this pass

    @property
    def seconds(self):
        return (self.end - self.start) / self.track.fps


def _frame_direction(tr):
    """(T,) +1 / -1 / NaN: which way the spine points in each frame."""
    d = tr.p("spine_at_neck")[:, 0] - tr.p("spine_at_waist")[:, 0]
    for sd in ("r", "l"):   # no spine: either side's shoulder-to-hip line works
        alt = tr.p(f"{sd}_shoulder")[:, 0] - tr.p(f"{sd}_hip")[:, 0]
        d = np.where(np.isnan(d), alt, d)
    return np.sign(d)


def _vote(sign, k):
    """Majority over a centred window, ignoring missing frames; NaN where none voted."""
    s = np.nan_to_num(sign, nan=0.0)
    n = (~np.isnan(sign)).astype(float)
    w = np.ones(k)
    tot, cnt = np.convolve(s, w, mode="same"), np.convolve(n, w, mode="same")
    out = np.where(cnt > 0, np.sign(tot), np.nan)
    out[(cnt > 0) & (tot == 0)] = np.nan
    return out


def _runs(d, bridge):
    """Runs of one direction as (start, end, dir). A short unreadable or contrary
    blip inside a run is absorbed; a long gap or a real reversal splits it."""
    runs = []
    i, n = 0, len(d)
    while i < n:
        if np.isnan(d[i]):
            i += 1
            continue
        sgn, j, last = d[i], i, i
        while j + 1 < n:
            if d[j + 1] == sgn:
                j += 1
                last = j
                continue
            # look ahead: does the run resume within `bridge` frames?
            k = j + 1
            while k < n and k - last <= bridge and d[k] != sgn:
                k += 1
            if k < n and k - last <= bridge and d[k] == sgn:
                j = k
                last = k
                continue
            break
        runs.append([i, last + 1, int(sgn)])
        i = last + 1
    return runs


_LIMB = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def _near_margin(tr):
    """Right-side minus left-side mean keypoint confidence (a missed point counts 0).

    Confidence, not just whether a point clears the trust bar: a far-side joint seen
    through the body can still clear it, but the model is less sure of it."""
    c = tr.data[..., 2]
    r = c[:, [IDX[f"r_{j}"] for j in _LIMB]].mean()
    l = c[:, [IDX[f"l_{j}"] for j in _LIMB]].mean()
    return float(r - l)


def _slice(tr, a, b, **labels):
    meta = dict(tr.meta, pass_start=a, **labels)
    return PoseTrack(tr.data[a:b], tr.fps, (tr.W, tr.H), tr.view, meta)


def split(tr: PoseTrack):
    """-> [Pass, ...] in clip order. Side view only."""
    if tr.view != SIDE:
        raise ValueError("passes only apply to the side view")
    fps = tr.fps
    raw = _frame_direction(tr)
    d = _vote(raw, max(3, int(round(VOTE_WINDOW_S * fps))) | 1)
    bridge = int(round(BRIDGE_S * fps))
    # The vote smears a pass's direction a little way into an unreadable stretch; where
    # that stretch is long (the turn), it stays unreadable, so trims count from real frames.
    i, n = 0, tr.n
    while i < n:
        if np.isnan(raw[i]):
            j = i
            while j + 1 < n and np.isnan(raw[j + 1]):
                j += 1
            if j - i + 1 > bridge:
                d[i:j + 1] = np.nan
            i = j + 1
        else:
            i += 1
    runs = _runs(d, bridge)

    if len(runs) <= 1:   # no turn: the whole clip
        m = _near_margin(tr)
        direction = runs[0][2] if runs else 0
        near = "r" if m >= 0 else "l"
        if abs(m) < STRONG_MARGIN:   # no clear near side: leave it to metrics.Context, as before
            return [Pass(0, tr.n, direction, near, "default", tr)]
        return [Pass(0, tr.n, direction, near, "visibility", _slice(tr, 0, tr.n, near_side=near))]

    before, after = int(round(TRIM_BEFORE_S * fps)), int(round(TRIM_AFTER_S * fps))
    kept = []
    for idx, (a, b, sgn) in enumerate(runs):
        if idx > 0:
            a += after                 # coming off a turn
        if idx < len(runs) - 1:
            b -= before                # heading into one
        if b - a >= MIN_PASS_S * fps:
            kept.append((a, b, sgn))
    if not kept:
        return []

    # Near side: strong visibility evidence decides; the camera-stays-on-one-deck rule
    # (near side flips with direction) fills in the rest. Expressed as "the side that
    # faces the camera when the swimmer travels image-right".
    margins = [_near_margin(_slice(tr, a, b)) for a, b, _ in kept]
    weight = sum(m * sgn * (b - a) for (a, b, sgn), m in zip(kept, margins) if abs(m) >= STRONG_MARGIN)
    right_mover_near = None if weight == 0 else ("r" if weight > 0 else "l")
    flip = {"r": "l", "l": "r"}

    out = []
    for (a, b, sgn), m in zip(kept, margins):
        if abs(m) >= STRONG_MARGIN:
            near, why = ("r" if m > 0 else "l"), "visibility"
        elif right_mover_near is not None:
            near, why = (right_mover_near if sgn > 0 else flip[right_mover_near]), "direction"
        else:
            near, why = ("r" if m >= 0 else "l"), "default"
        out.append(Pass(a, b, sgn, near, why,
                        _slice(tr, a, b, near_side=near, direction=sgn, pass_index=len(out))))
    return out


def summary(passes):
    """JSON-friendly description of how a clip was split, for the report."""
    return [{"start": p.start, "end": p.end, "seconds": round(p.seconds, 1),
             "direction": p.direction, "near": p.near, "near_from": p.near_from} for p in passes]
