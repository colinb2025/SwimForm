"""
Tracked frames: one still from each of the swimmer's own clips with the points the pose
model found drawn over it. It's the report's proof of work: the swimmer (and a coach
reading along) can see at a glance whether the model was actually following the body.

Only made when the keypoints came from the uploaded video (not in demo mode, where the
swimmer is simulated and the points wouldn't line up with anything).
"""
import logging

import numpy as np

from swimform.schema import EDGES, IDX, KEYPOINTS

log = logging.getLogger("swimform.tracked")

CONF = 0.5          # a point is drawn when the model was at least this sure
MAX_WIDTH = 960     # stills are scaled down to this; plenty for a report
# BGR. Left limbs and right limbs in different colours so a left/right swap is obvious;
# yellow and coral because both stand out against pool-blue water.
LEFT, RIGHT, MID, EDGE = (10, 214, 255), (77, 77, 255), (255, 255, 255), (30, 30, 30)


def name(slot):
    return f"tracked_{slot}.jpg"


def pick_frame(track):
    """Index of the frame that shows the tracking best: most points seen, then surest,
    looking at the middle of the clip (push-offs and turns at the ends aren't swimming)."""
    conf = track.data[:, :, 2]
    n = len(conf)
    if n == 0:
        return None
    lo, hi = (int(n * 0.1), max(int(n * 0.9), int(n * 0.1) + 1)) if n >= 10 else (0, n)
    seen = (conf[lo:hi] > CONF).sum(axis=1)
    if seen.max() < 6:   # fewer than six points anywhere: nothing worth showing
        return None
    score = seen + conf[lo:hi].mean(axis=1)
    return lo + int(np.argmax(score))


def _colour(a, b=None):
    sides = {k[0] for k in (a, b) if k and k[:2] in ("l_", "r_")}
    return LEFT if sides == {"l"} else RIGHT if sides == {"r"} else MID


def _crop(points, w, h):
    """The part of the frame to show: the swimmer with room around them, 16:9, so a
    swimmer filmed from across the pool isn't a speck in the picture."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    cw = max(bw * 1.5, bh * 2.0 * 16 / 9, w * 0.3)
    cw = min(cw, w, h * 16 / 9)
    ch = cw * 9 / 16
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0 = int(min(max(cx - cw / 2, 0), w - cw))
    y0 = int(min(max(cy - ch / 2, 0), h - ch))
    return x0, y0, x0 + int(cw), y0 + int(ch)


def render(video_path, track, out_path):
    """Write the still. -> True if one was written. Never raises: a report without the
    picture is still a report."""
    try:
        import cv2
        i = pick_frame(track)
        if i is None:
            return False
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            return False
        h, w = img.shape[:2]
        # The track's pixel space is the video's, unless the phone stored it rotated.
        sx, sy = w / track.W, h / track.H
        pts = track.data[i]
        seen = [(int(p[0] * sx), int(p[1] * sy)) for p in pts if p[2] > CONF]
        x0, y0, x1, y1 = _crop(seen, w, h)
        thick = max(2, round((x1 - x0) / 240))
        over = img.copy()
        lines = [(a, b) for a, b in EDGES if pts[IDX[a]][2] > CONF and pts[IDX[b]][2] > CONF]
        at = lambda k: (int(pts[IDX[k]][0] * sx), int(pts[IDX[k]][1] * sy))   # noqa: E731
        for a, b in lines:   # dark edge first, so the lines read on bright water too
            cv2.line(over, at(a), at(b), EDGE, thick + 2, cv2.LINE_AA)
        for a, b in lines:
            cv2.line(over, at(a), at(b), _colour(a, b), thick, cv2.LINE_AA)
        for k in KEYPOINTS:
            if pts[IDX[k]][2] > CONF:
                cv2.circle(over, at(k), thick + 3, EDGE, -1, cv2.LINE_AA)
                cv2.circle(over, at(k), thick + 1, _colour(k), -1, cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.9, img, 0.1, 0)[y0:y1, x0:x1]
        h, w = img.shape[:2]
        if w > MAX_WIDTH:
            img = cv2.resize(img, (MAX_WIDTH, round(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        return bool(cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 82]))
    except Exception:   # noqa: BLE001
        log.exception("could not draw the tracked frame for %s", video_path)
        return False
