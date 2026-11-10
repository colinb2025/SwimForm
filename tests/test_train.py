"""The two-command training loop, with the GPU parts (pose, fine-tune, grading) stood in for."""
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from swimform import train
from swimform.schema import KEYPOINTS
from tests.test_labeling import _clip, _export, _guesses


def fake_pose(names):
    def run(argv):
        out, weights, clips = Path(argv[0]), argv[1], argv[2:]
        out.mkdir(parents=True, exist_ok=True)
        run.used = weights
        for c in clips:
            (out / f"{Path(c).stem}_pose.json").write_text(json.dumps(_guesses(names, seed=len(c))))
    return run


def test_prepare_then_fit_round_one_and_two(tmp_path, capsys):
    from swimform import pose
    clips, work = tmp_path / "clips", tmp_path / "training"
    clips.mkdir()
    for name in ("ann_side_1", "bob_side_1", "cy_headon_1"):
        _clip(clips / f"{name}.mp4")

    # round 1: no weights of ours yet, so the stock model seeds the guesses
    run = fake_pose(pose.COCO)
    round1 = train.prepare(train.main.__globals__["argparse"].Namespace(
        clips=str(clips), round=1, per_clip=6, weights=None, work=str(work)), run_pose=run)
    assert run.used == train.STOCK and (round1 / "tasks.json").exists()
    assert "label-studio start" in capsys.readouterr().out

    export = tmp_path / "export.json"
    export.write_text(json.dumps(_export(json.loads((round1 / "tasks.json").read_text()))))
    seen = {}

    def fake_train(data, base, name, a):
        seen.update(data=data, base=base, name=name)
        best = tmp_path / "best.pt"
        best.write_bytes(b"weights")
        return best

    def fake_grade(argv):
        seen["grade"] = argv
        return 0

    args = train.main.__globals__["argparse"].Namespace(round=1, export=str(export), base=train.STOCK, epochs=1,
                                                         imgsz=960, batch=8, device=None, work=str(work))
    assert train.fit(args, train=fake_train, grade=fake_grade) == 0
    out = capsys.readouterr().out
    final = work / "models" / "swimform-r1.pt"
    assert final.read_bytes() == b"weights" and seen["name"] == "swimform-r1"
    assert f"kpt_shape: [{len(KEYPOINTS)}, 3]" in Path(seen["data"]).read_text()
    assert "SWIMFORM_POSE=yolo" in out and str(final.resolve()) in out
    assert seen["grade"][seen["grade"].index("--weights") + 1] == str(final)       # nothing earlier to compare with

    # round 2: last round's weights seed the guesses, and grading compares the two
    assert train.weights_for(work, 2) == str(final)
    run2 = fake_pose(list(KEYPOINTS))
    train.prepare(train.main.__globals__["argparse"].Namespace(
        clips=str(clips), round=2, per_clip=6, weights=None, work=str(work)), run_pose=run2)
    assert run2.used == str(final)
    assert train.image_dirs(work, 2) == [str(work / "round1" / "images"), str(work / "round2" / "images")]
    args.round = 2
    assert train.fit(args, train=fake_train, grade=lambda argv: seen.update(grade=argv) or 1) == 1
    w = seen["grade"].index("--weights")
    assert seen["grade"][w + 1:w + 3] == [str(final), str(work / "models" / "swimform-r2.pt")]
    assert "Not at the bar yet" in capsys.readouterr().out


def test_prepare_needs_clips(tmp_path):
    import argparse
    with pytest.raises(SystemExit):
        train.prepare(argparse.Namespace(clips=str(tmp_path), round=1, per_clip=6, weights=None, work=str(tmp_path / "w")))
