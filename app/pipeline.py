"""
Upload -> pose -> clip check -> analysis -> saved report.

The pose step is pluggable (SWIMFORM_POSE):

  synthetic  default until the fine-tuned model is ready. Generates a deterministic
             synthetic swimmer from the uploaded files' bytes, so the whole flow runs
             end to end. Dev conveniences: a file name containing "head"/"front" or
             "side" decides which view is generated (to try the swapped-clip check),
             "deck" simulates a deck-level camera, and the side clip swims a 50 (down
             and back) unless its name contains "oneway".
  yolo       runs swimform.pose with the weights in SWIMFORM_WEIGHTS.
  collect    no analysis yet: clips are checked for type and size, stored, and the
             analysis waits (status "waiting") until an admin re-runs it with a real
             model. For gathering beta footage before the model is ready.
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
from swimform.schema import HEAD_ON, KEYPOINTS, SIDE
from swimform.viewcheck import check_clip

from . import storage

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SWIMFORM_DATA", ROOT / "data"))
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_BYTES = 500 * 1024 * 1024
SLOTS = (SIDE, HEAD_ON)   # the two views every analysis needs
# Upload slots: which view each one is, and where the camera was. An analysis needs at
# least one clip of each view; extra clips of a view are all used (see swimform.analyze).
CLIPS = {
    "side": (SIDE, "above"), "head_on": (HEAD_ON, "above"),
    "side_uw": (SIDE, "underwater"), "head_on_uw": (HEAD_ON, "underwater"),
}
VIEW_WORDS = {SIDE: "side-view", HEAD_ON: "head-on"}


def slots_for(view):
    return [slot for slot, (v, _) in CLIPS.items() if v == view]


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
    from swimform.synthetic import ARM_FAULTS, FAULTS, Swimmer
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 4))
    picks = rng.choice(len(FAULTS), size=k, replace=False)
    faults, arm_faults = {}, {}
    for i in picks:
        f, sev = FAULTS[i], float(rng.uniform(0.7, 1.0))
        if f in ARM_FAULTS and rng.random() < 0.5:   # one-sided, as many real ones are
            arm_faults.setdefault(str(rng.choice(["r", "l"])), {})[f] = sev
        else:
            faults[f] = sev
    return Swimmer(faults, cycle_s=float(rng.uniform(1.4, 1.8)), arm_faults=arm_faults)


def pose_synthetic(paths, names):
    from swimform.synthetic import generate, generate_down_and_back
    digests = {slot: _sha(paths[slot]) for slot in paths}
    first = {view: next(slot for slot in slots_for(view) if slot in paths) for view in SLOTS}
    session_seed = int.from_bytes(hashlib.sha256(digests[first[SIDE]] + digests[first[HEAD_ON]]).digest()[:8], "big")
    sw = _synthetic_swimmer(session_seed)
    tracks = {}
    for slot in paths:
        slot_view, water = CLIPS[slot]
        n = names[slot].lower()
        view = HEAD_ON if ("head" in n or "front" in n) else SIDE if "side" in n else slot_view
        camera = "deck" if "deck" in n else "underwater"
        seed = int.from_bytes(digests[slot][:4], "big")
        if view == SIDE and "oneway" not in n:
            tr = generate_down_and_back(sw, camera=camera, quality="typical", seed=seed)
        else:
            tr = generate(view, sw, camera=camera, quality="typical", seed=seed, t0=3.7 if view == HEAD_ON else 0.0)
        tr.view = slot_view   # the slot it was uploaded into; the clip check decides if that was right
        tracks[slot] = tr
    return tracks


def pose_yolo(paths, names):
    from ultralytics import YOLO
    from swimform import pose as harness
    model = YOLO(os.environ["SWIMFORM_WEIGHTS"])
    tracks = {}
    for slot in paths:
        with tempfile.TemporaryDirectory() as out:
            summary = harness.analyze(str(paths[slot]), model, out)
            if summary.get("keypoints") not in (None, list(KEYPOINTS)):
                raise RuntimeError("SWIMFORM_WEIGHTS is not a 15-point SwimForm model (it predicts "
                                   f"{len(summary['keypoints'])} points); use the fine-tuned weights")
            stem = Path(paths[slot]).stem
            tracks[slot] = PoseTrack.from_json(os.path.join(out, f"{stem}_pose.json"), view=CLIPS[slot][0])
    return tracks


PROVIDERS = {"synthetic": pose_synthetic, "yolo": pose_yolo}


# ------------------------------------------------------------------ the run
def save_uploads(files):
    """files: {slot: (filename, fileobj)} -> (analysis id, {slot: path}, {slot: filename})."""
    files = {slot: f for slot, f in files.items() if slot in CLIPS and f[0]}
    problems = {}
    for slot, (name, _) in files.items():
        if Path(name).suffix.lower() not in VIDEO_EXT:
            problems[slot] = ["This doesn't look like a video. Use an .mp4 or .mov file straight from your phone."]
    for view in SLOTS:   # one clip of each view is the minimum; above water or under it
        if not any(slot in files for slot in slots_for(view)):
            problems.setdefault(view, []).append(f"Add a {VIEW_WORDS[view]} clip: above water, underwater, or both.")
    if problems:
        raise UploadProblem(problems)
    aid = uuid.uuid4().hex[:12]
    folder = DATA / "analyses" / aid
    folder.mkdir(parents=True, exist_ok=True)
    try:
        paths, names = _write(folder, files)
    except UploadProblem:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)   # don't keep half an upload
        raise
    return aid, paths, names


def _write(folder, files):
    paths, names = {}, {}
    for slot in files:
        name, fobj = files[slot]
        dest = folder / f"{slot}{Path(name).suffix.lower()}"
        size = 0
        with open(dest, "wb") as out:
            while chunk := fobj.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise UploadProblem({slot: ["This clip is over 500 MB. Trim it to one 50 of swimming."]})
                out.write(chunk)
        if size == 0:
            raise UploadProblem({slot: ["This file is empty."]})
        paths[slot], names[slot] = dest, name
    return paths, names


def folder(aid):
    return DATA / "analyses" / aid


def queue(files, provider=None):
    """Save the uploads and note how to process them. Fast: runs inside the request.
    Raises UploadProblem for problems visible without looking at the video."""
    aid, paths, names = save_uploads(files)
    provider = provider or os.environ.get("SWIMFORM_POSE", "synthetic")
    with open(folder(aid) / "upload.json", "w") as f:
        json.dump({"files": names, "paths": {k: v.name for k, v in paths.items()}, "provider": provider}, f)
    return aid


def process(aid, rules_path=DEFAULT_RULES):
    """Pose -> clip check -> analysis -> report.json. Slow: runs in a background job.
    Raises UploadProblem with per-slot fixes when a clip fails its check."""
    up = upload_info(aid)
    paths = {slot: folder(aid) / name for slot, name in up["paths"].items()}
    names, provider = up["files"], up["provider"]
    for slot, path in paths.items():   # a re-run after the clips moved to cloud storage
        if storage.fetch(folder(aid), aid, path.name) is None:
            raise RuntimeError(f"the {slot} clip is no longer stored")
    tracks = PROVIDERS[provider](paths, names)
    checks = {slot: check_clip(tracks[slot], CLIPS[slot][0]) for slot in tracks}
    usable = {view: [slot for slot in slots_for(view) if slot in checks and checks[slot]["ok"]] for view in SLOTS}
    if not all(usable.values()):   # a view with no usable clip: nothing honest to report
        raise UploadProblem({slot: c["problems"] for slot, c in checks.items() if not c["ok"]})
    rules, errors = R.load(rules_path)
    if errors:
        raise RuntimeError("rules workbook invalid: " + "; ".join(errors))
    report = analyze({view: [tracks[slot] for slot in usable[view]] for view in SLOTS}, rules, top_n=3)
    # What each uploaded clip gave us, for the report's footage notes. An extra clip that
    # failed its check is left out of the analysis and listed here with the reason.
    clips = []
    for slot in [sl for view in SLOTS for sl in slots_for(view) if sl in tracks]:   # grouped by view
        view, water = CLIPS[slot]
        tr, c = tracks[slot], checks[slot]
        used = slot in usable[view]
        info = report["clips"][view][usable[view].index(slot)] if used else {}
        clips.append({"slot": slot, "view": view, "water": water, "file": names[slot], "used": used,
                      "problems": c["problems"], "cycles": c["cycles"], "visible": round(c["visible"], 3),
                      "passes": len(c["passes"]) if c.get("passes") else None,
                      "fps": round(float(tr.fps), 1), "size": [int(tr.W), int(tr.H)],
                      "seconds": round(tr.n / tr.fps, 1) if tr.fps else None,
                      "measurements_used": info.get("measurements_used", 0)})
    record = {"id": aid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "files": names, "clips": clips, "pose_provider": provider, "report": report}
    with open(folder(aid) / "report.json", "w") as f:
        json.dump(record, f, default=str, indent=1)
    return record


def upload_info(aid):
    """What was uploaded and how it should be processed, or None if that's gone."""
    path = storage.fetch(folder(aid), aid, "upload.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def set_provider(aid, provider):
    up = upload_info(aid)
    up["provider"] = provider
    with open(folder(aid) / "upload.json", "w") as f:
        json.dump(up, f)


def run(files, provider=None, rules_path=DEFAULT_RULES):
    """Upload-to-report in one go (scripts and tests)."""
    aid = queue(files, provider)
    process(aid, rules_path)
    return aid


def remove(aid):
    """Delete everything stored for one analysis: clips, pose, report."""
    import shutil
    if aid.isalnum():
        shutil.rmtree(folder(aid), ignore_errors=True)
        try:
            storage.remove(aid)
        except Exception:   # noqa: BLE001 - deleting the account must not fail on a storage hiccup
            import logging
            logging.getLogger("swimform.storage").exception("could not remove %s from storage", aid)


def load(aid):
    if not aid.isalnum():
        return None
    path = storage.fetch(folder(aid), aid, "report.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)
