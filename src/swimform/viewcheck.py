"""
Check an uploaded clip before analysing it: is it the camera angle we asked for, is
the swimmer visible enough, and are there enough stroke cycles to measure?

Each problem comes back as a plain-language message that says how to fix it, for the
upload page. Catching a swapped or diagonal clip here is cheaper and more honest than
analysing it and reporting faults that are really camera-angle artefacts.

The angle is read from body geometry, not trusted from the upload slot:
  - side-on, the spine runs across the frame and the two shoulders nearly overlap;
  - head-on, the shoulders spread across the frame and the spine points at the camera.
"""
import math
import numpy as np

from . import metrics as M
from .posedata import PoseTrack
from .schema import HEAD_ON, SIDE

MIN_SWIMMER_VISIBLE = 0.4   # share of frames with both shoulders found
MIN_CYCLES = 3
SIDE_RATIO, HEAD_ON_RATIO = 3.0, 0.5   # spine span / shoulder span; each ~22 deg off square

VIEW_NAMES = {SIDE: "side view", HEAD_ON: "head-on view"}


def _med(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def guess_view(tr: PoseTrack):
    """-> ("side" | "head_on" | "unclear" | "unknown", spine-to-shoulder span ratio)."""
    spine = np.abs(tr.p("spine_at_neck")[:, 0] - tr.p("spine_at_waist")[:, 0])
    if np.isnan(spine).mean() > 0.7:   # fall back to shoulder-to-hip on the near side
        spine = np.abs(tr.p("r_shoulder")[:, 0] - tr.p("r_hip")[:, 0])
    shoulders = np.abs(tr.p("l_shoulder")[:, 0] - tr.p("r_shoulder")[:, 0])
    s, w = _med(spine), _med(shoulders)
    if math.isnan(s) or math.isnan(w):
        return "unknown", float("nan")
    ratio = s / max(w, 1.0)
    if ratio >= SIDE_RATIO:
        return SIDE, ratio
    if ratio <= HEAD_ON_RATIO:
        return HEAD_ON, ratio
    return "unclear", ratio


def check_clip(tr: PoseTrack, expected_view):
    """
    -> {"ok": bool, "problems": [message, ...], "view_guess", "visible", "cycles"}
    `tr.view` is whatever slot the clip was uploaded into; `expected_view` is that slot.
    """
    problems = []
    visible = float(tr.ok("l_shoulder", "r_shoulder").mean()) if tr.view == HEAD_ON \
        else float((tr.ok("l_shoulder") | tr.ok("r_shoulder")).mean())
    if visible < MIN_SWIMMER_VISIBLE:
        problems.append(
            f"We could only find the swimmer in {round(100 * visible)}% of this clip. "
            "Film one swimmer, close enough to fill about a third of the frame, with nobody "
            "else in the lane or on the deck in shot.")
        return {"ok": False, "problems": problems, "view_guess": None, "visible": visible, "cycles": 0}

    guess, ratio = guess_view(tr)
    other = HEAD_ON if expected_view == SIDE else SIDE
    if guess == other:
        problems.append(
            f"This looks like a {VIEW_NAMES[other]}, but it was added as the {VIEW_NAMES[expected_view]}. "
            "Did the two clips get swapped?")
    elif guess == "unclear":
        problems.append(
            "The camera seems to be at a diagonal. Move it so it faces the swimmer "
            + ("square-on from the side of the pool." if expected_view == SIDE
               else "straight down the lane, from the end wall."))

    cycles = len(M.Context(tr).cycles)
    if not problems and cycles < MIN_CYCLES:
        problems.append(
            f"We could only read {cycles} full stroke cycle{'s' if cycles != 1 else ''}. "
            "Film at least 15 seconds of steady swimming, without turns or stops.")
    return {"ok": not problems, "problems": problems, "view_guess": guess, "visible": visible, "cycles": cycles}
