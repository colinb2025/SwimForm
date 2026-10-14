"""
Upload -> pose -> clip check -> analysis -> saved report.

The pose step is pluggable (SWIMFORM_POSE):

  synthetic  default until the fine-tuned model is ready. Generates a deterministic
             synthetic swimmer from the uploaded files' bytes, so the whole flow runs
             end to end. Dev conveniences: a file name containing "head"/"front" or
             "side" decides which view is generated (to try the swapped-clip check), and
             "deck" simulates a deck-level camera.
  yolo       runs swimform.pose with the weights in SWIMFORM_WEIGHTS.
"""
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from swimform import rules as R
from swimform.analyze import DEFAULT_RULES, analyze
from swimform.posedata import PoseTrack
from swimform.schema import HEAD_ON, SIDE
from swimform.viewcheck import check_clip

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SWIMFORM_DATA", ROOT / "data"))
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_BYTES = 500 * 1024 * 1024
SLOTS = (SIDE, HEAD_ON)


class UploadProblem(Exception):
    def __init__(self, problems):
        super().__init__("; ".join(f"{k}: {v}" for k, v in problems.items()))
        self.problems = problems   # {slot: [message, ...]}


# ------------------------------------------------------------------ pose step
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _synthetic_swimmer(seed):
    from swimform.synthetic import FAULTS, Swimmer
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 4))
    picks = rng.choice(len(FAULTS), size=k, replace=False)
    return Swimmer({FAULTS[i]: float(rng.uniform(0.7, 1.0)) for i in picks},
                   cycle_s=float(rng.uniform(1.4, 1.8)))


def pose_synthetic(paths, names):
    from swimform.synthetic import generate
    digests = {s: _sha(paths[s]) for s in SLOTS}
    session_seed = int.from_bytes(hashlib.sha256(digests[SIDE] + digests[HEAD_ON]).digest()[:8], "big")
    sw = _synthetic_swimmer(session_seed)
    tracks = {}
    for slot in SLOTS:
        n = names[slot].lower()
        view = HEAD_ON if ("head" in n or "front" in n) else SIDE if "side" in n else slot
        camera = "deck" if "deck" in n else "underwater"
        seed = int.from_bytes(digests[slot][:4], "big")
        tr = generate(view, sw, camera=camera, quality="typical", seed=seed, t0=3.7 if view == HEAD_ON else 0.0)
        tr.view = slot   # the slot it was uploaded into; the clip check decides if that was right
        tracks[slot] = tr
    return tracks


def pose_yolo(paths, names):
    from ultralytics import YOLO
    from swimform import pose as harness
    model = YOLO(os.environ["SWIMFORM_WEIGHTS"])
    tracks = {}
    for slot in SLOTS:
        with tempfile.TemporaryDirectory() as out:
            harness.analyze(str(paths[slot]), model, out)
            stem = Path(paths[slot]).stem
            tracks[slot] = PoseTrack.from_json(os.path.join(out, f"{stem}_pose.json"), view=slot)
    return tracks


PROVIDERS = {"synthetic": pose_synthetic, "yolo": pose_yolo}


# ------------------------------------------------------------------ the run
def save_uploads(files):
    """files: {slot: (filename, fileobj)} -> (analysis id, {slot: path}, {slot: filename})."""
    problems = {}
    for slot in SLOTS:
        if slot not in files or not files[slot][0]:
            problems[slot] = ["Add a clip here."]
        elif Path(files[slot][0]).suffix.lower() not in VIDEO_EXT:
            problems[slot] = ["This doesn't look like a video. Use an .mp4 or .mov file straight from your phone."]
    if problems:
        raise UploadProblem(problems)
    aid = uuid.uuid4().hex[:12]
    folder = DATA / "analyses" / aid
    folder.mkdir(parents=True, exist_ok=True)
    paths, names = {}, {}
    for slot in SLOTS:
        name, fobj = files[slot]
        dest = folder / f"{slot}{Path(name).suffix.lower()}"
        size = 0
        with open(dest, "wb") as out:
            while chunk := fobj.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise UploadProblem({slot: ["This clip is over 500 MB. Trim it to 15–30 seconds of swimming."]})
                out.write(chunk)
        if size == 0:
            raise UploadProblem({slot: ["This file is empty."]})
        paths[slot], names[slot] = dest, name
    return aid, paths, names


def run(files, provider=None, rules_path=DEFAULT_RULES):
    """Full upload-to-report run. Raises UploadProblem with per-slot fixes when a clip fails its check."""
    aid, paths, names = save_uploads(files)
    provider = provider or os.environ.get("SWIMFORM_POSE", "synthetic")
    tracks = PROVIDERS[provider](paths, names)
    checks = {slot: check_clip(tracks[slot], slot) for slot in SLOTS}
    problems = {slot: c["problems"] for slot, c in checks.items() if not c["ok"]}
    if problems:
        raise UploadProblem(problems)
    rules, errors = R.load(rules_path)
    if errors:
        raise RuntimeError("rules workbook invalid: " + "; ".join(errors))
    report = analyze(tracks, rules, top_n=3)
    record = {"id": aid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "files": names, "pose_provider": provider, "report": report}
    with open(DATA / "analyses" / aid / "report.json", "w") as f:
        json.dump(record, f, default=str, indent=1)
    return aid


def load(aid):
    if not aid.isalnum():
        return None
    path = DATA / "analyses" / aid / "report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
