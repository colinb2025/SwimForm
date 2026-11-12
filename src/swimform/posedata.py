"""
Pose track: the contract between any pose model and the rest of the pipeline.

The on-disk format is what `swimform.pose` already writes:

    {"summary": {"fps":, "size": [W, H], "view": "side" | "head_on", ...},
     "frames": [{"frame": i, "t": s, "box": [...] | None,
                 "kp": {name: [x_px, y_px, conf], ...} | None}, ...]}

`PoseTrack` loads that into a (T, 15, 3) array, matching keypoints by name so a
legacy COCO-17 file still loads (its face points are dropped and the spine points
come back missing). Anything that produces this JSON — the YOLO harness, a future
RTMPose harness, or `swimform.synthetic` — plugs into metrics without changes.
"""
import json
import numpy as np

from .schema import CONF, IDX, KEYPOINTS, NUM_KEYPOINTS, VIEWS


class PoseDataError(ValueError):
    """The pose data breaks the contract."""


class PoseTrack:
    def __init__(self, data, fps, size, view, meta=None):
        self.data = np.asarray(data, dtype=float)      # (T, 15, 3) x_px, y_px, conf
        self.fps = float(fps)
        self.W, self.H = int(size[0]), int(size[1])
        self.view = view
        self.meta = meta or {}
        self._validate()

    def _validate(self):
        d = self.data
        if d.ndim != 3 or d.shape[1:] != (NUM_KEYPOINTS, 3):
            raise PoseDataError(f"expected (T, {NUM_KEYPOINTS}, 3), got {d.shape}")
        if self.view not in VIEWS:
            raise PoseDataError(f"view must be one of {VIEWS}, got {self.view!r}")
        if self.fps <= 0 or self.W <= 0 or self.H <= 0:
            raise PoseDataError("fps and size must be positive")
        c = d[..., 2]
        if np.isnan(c).any() or (c < 0).any() or (c > 1).any():
            raise PoseDataError("confidence must be in [0, 1]")

    # -- loading ---------------------------------------------------------------
    @classmethod
    def from_json(cls, obj, view=None):
        """obj: parsed harness JSON, or a path to it. `view` overrides summary['view']."""
        if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
            with open(obj) as f:
                obj = json.load(f)
        s, frames = obj["summary"], obj["frames"]
        view = view or s.get("view")
        if view is None:
            raise PoseDataError("view unknown: pass view= or set summary['view']")
        data = np.full((len(frames), NUM_KEYPOINTS, 3), np.nan)
        data[..., 2] = 0.0
        for t, f in enumerate(frames):
            for name, v in (f.get("kp") or {}).items():
                i = IDX.get(name)
                if i is not None and v is not None and v[2] > 0:
                    data[t, i] = v
        meta = {k: v for k, v in s.items() if k not in ("fps", "size", "view")}
        return cls(data, s["fps"], s["size"], view, meta)

    def to_json(self):
        frames = []
        for t in range(self.n):
            kp = {}
            for name, i in IDX.items():
                x, y, c = self.data[t, i]
                kp[name] = [0.0, 0.0, 0.0] if c == 0 else [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
            frames.append({"frame": t, "t": round(t / self.fps, 4), "box": None, "kp": kp})
        summary = dict(self.meta, fps=self.fps, size=[self.W, self.H], view=self.view,
                       frames=self.n, keypoints=list(KEYPOINTS))
        return {"summary": summary, "frames": frames}

    # -- access ----------------------------------------------------------------
    @property
    def n(self):
        return self.data.shape[0]

    def p(self, name):
        """(T, 2) pixel coordinates, NaN where the point is missing or untrusted."""
        out = self.data[:, IDX[name], :2].copy()
        out[self.data[:, IDX[name], 2] < CONF] = np.nan
        return out

    def ok(self, *names):
        """(T,) mask: every named point trusted in that frame."""
        m = np.ones(self.n, dtype=bool)
        for k in names:
            m &= self.data[:, IDX[k], 2] >= CONF
        return m

    def detection_rates(self):
        return {k: round(float((self.data[:, i, 2] >= CONF).mean()), 3) for k, i in IDX.items()}
