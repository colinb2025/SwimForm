"""
The model-improvement loop in two commands.

  # 1. Choose frames from your clips and pre-fill the current model's guesses.
  python -m swimform.train prepare --clips clips/ --round 1
  #    ...then correct them in Label Studio and export JSON (docs/labeling.md).

  # 2. Build the dataset, fine-tune, grade against held-out clips, install the weights.
  python -m swimform.train fit --round 1 --export export.json

Everything lands under training/ (kept out of git):

  training/round1/          frames, tasks.json and labeling_config.xml for Label Studio
  training/pose_r1/         the model's guesses on your clips
  training/dataset_r1/      YOLO-pose dataset: every round so far, val split by clip
  training/models/swimform-r1.pt   the fine-tuned weights
  training/scorecard_r1/    how it did on held-out clips (and against the previous round)

Round 1's guesses come from the stock COCO model; later rounds use the previous round's
weights, so each round you correct less. `fit` exits 0 when the model passes the
scorecard's bar, and prints the two settings that switch the web app over to it.
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

VIDEO_EXT = ("mp4", "mov", "m4v", "avi", "mkv")
STOCK = "yolo11m-pose.pt"


def weights_for(work: Path, rnd: int):
    """The model whose guesses seed round `rnd`: last round's weights, or the stock model."""
    prev = work / "models" / f"swimform-r{rnd - 1}.pt"
    return str(prev) if prev.exists() else STOCK


def clips_in(folder):
    found = []
    for ext in VIDEO_EXT:
        found += glob.glob(os.path.join(folder, f"*.{ext}")) + glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
    return sorted(set(found))


def image_dirs(work: Path, rnd: int):
    """Every round's frames up to this one: round N trains on all of them."""
    return [str(work / f"round{n}" / "images") for n in range(1, rnd + 1) if (work / f"round{n}" / "images").is_dir()]


def prepare(a, run_pose=None):
    from .labeling import prep
    work = Path(a.work)
    clips = clips_in(a.clips)
    if not clips:
        raise SystemExit(f"no video clips found in {a.clips}")
    weights = a.weights or weights_for(work, a.round)
    pose_dir, round_dir = work / f"pose_r{a.round}", work / f"round{a.round}"
    print(f"Round {a.round}: {len(clips)} clips, guesses from {weights}")
    if run_pose is None:
        from . import pose
        run_pose = pose.main
    run_pose([str(pose_dir), weights, *clips])
    prep.main(["--videos", a.clips, "--pose", str(pose_dir), "--out", str(round_dir),
               "--per-clip", str(a.per_clip), "--model-version", f"round{a.round - 1}"])
    print(f"""
Next:
  1. label-studio start          (pip install label-studio)
  2. New project > Labeling Setup > Custom template: paste {round_dir}/labeling_config.xml
  3. Settings > Cloud Storage > Local files: serve {round_dir}/images   (docs/labeling.md)
  4. Import {round_dir}/tasks.json, correct the points, then Export > JSON
  5. python -m swimform.train fit --round {a.round} --export <the exported file>""")
    return round_dir


def fit(a, train=None, grade=None):
    from .labeling import convert
    work = Path(a.work)
    dirs = image_dirs(work, a.round)
    if not dirs:
        raise SystemExit(f"no frames found under {work}/round*/images: run `prepare` first")
    dataset = work / f"dataset_r{a.round}"
    convert.main(["--export", a.export, "--images", *dirs, "--out", str(dataset)])
    data = str(dataset / "data.yaml")

    if train is None:
        def train(data, base, name, a):
            from ultralytics import YOLO
            result = YOLO(base).train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=25,
                                      project=str(work / "runs"), name=name, exist_ok=True, device=a.device)
            return Path(result.save_dir) / "weights" / "best.pt"
    best = train(data, a.base, f"swimform-r{a.round}", a)

    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    final = models / f"swimform-r{a.round}.pt"
    shutil.copy(best, final)

    prev = models / f"swimform-r{a.round - 1}.pt"
    card = work / f"scorecard_r{a.round}"
    if grade is None:
        from . import scorecard
        grade = scorecard.main
    code = grade(["--data", data, "--weights", *([str(prev)] if prev.exists() else []), str(final),
                  "--imgsz", str(a.imgsz), "--out", str(card), "--draw", "12"])
    print(f"\nWeights: {final}\nScorecard: {card}/scorecard.md  (worst frames drawn in {card}/frames)")
    if code == 0:
        print(f"\nPASSES the bar. To switch the web app to it, set:\n  SWIMFORM_POSE=yolo\n  SWIMFORM_WEIGHTS={final.resolve()}\n"
              "then open /admin and run the waiting analyses.")
    else:
        print(f"\nNot at the bar yet. Label another round: python -m swimform.train prepare --clips <clips> --round {a.round + 1}")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="pick frames and pre-fill guesses for labeling")
    p.add_argument("--clips", required=True, help="folder of swim clips (name them <swimmer>_<view>_<n>.mp4)")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--per-clip", type=int, default=30, help="frames to label from each clip")
    p.add_argument("--weights", help="model for the guesses (default: last round's, else the stock model)")
    p.add_argument("--work", default="training")
    f = sub.add_parser("fit", help="build the dataset, fine-tune, grade, install")
    f.add_argument("--round", type=int, required=True)
    f.add_argument("--export", required=True, help="Label Studio JSON export (all rounds labeled so far)")
    f.add_argument("--base", default=STOCK, help="model to fine-tune from")
    f.add_argument("--epochs", type=int, default=100)
    f.add_argument("--imgsz", type=int, default=960)
    f.add_argument("--batch", type=int, default=8)
    f.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu")
    f.add_argument("--work", default="training")
    a = ap.parse_args(argv)
    return (prepare(a) and 0) if a.cmd == "prepare" else fit(a)


if __name__ == "__main__":
    sys.exit(main())
