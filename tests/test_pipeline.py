"""
End-to-end checks of pose -> metrics -> rules against the real workbook, using
synthetic swimmers whose faults are known. The bar these tests hold:

  - a clean swimmer triggers no fault, on clean, noisy, or deck-level footage;
  - each fault triggers its own issue when the footage can show it;
  - when the footage cannot show it, the issue is reported as not assessed.
"""
import json

import numpy as np
import pytest

from swimform import rules as R
from swimform.analyze import MissingViewError, WorkbookMismatch, _check_registry, analyze
from swimform.metrics import REGISTRY, compute
from swimform.posedata import PoseDataError, PoseTrack
from swimform.schema import FLIP_IDX, HEAD_ON, KEYPOINTS, SIDE
from swimform.synthetic import FAULTS, Swimmer, generate

WORKBOOK = "rules/swim_rules.xlsx"


@pytest.fixture(scope="module")
def rules():
    r, errors = R.load(WORKBOOK)
    assert not errors, errors
    return r


def session(faults=None, **kw):
    sw = Swimmer(faults or {})
    return {v: generate(v, sw, **kw) for v in (SIDE, HEAD_ON)}


def triggered(report):
    return {p["issue_id"] for p in report["priorities"]} | set(report["other_issues"])


# ------------------------------------------------------------------ schema
def test_schema_is_15_points_with_spine_and_no_face():
    assert len(KEYPOINTS) == 15
    assert {"top_of_head", "spine_at_neck", "spine_at_waist"} <= set(KEYPOINTS)
    assert not {"nose", "l_eye", "r_eye", "l_ear", "r_ear"} & set(KEYPOINTS)


def test_flip_index_swaps_sides_and_is_an_involution():
    assert [FLIP_IDX[i] for i in FLIP_IDX] == list(range(15))
    assert KEYPOINTS[FLIP_IDX[KEYPOINTS.index("l_wrist")]] == "r_wrist"
    assert FLIP_IDX[KEYPOINTS.index("spine_at_neck")] == KEYPOINTS.index("spine_at_neck")


# ---------------------------------------------------------------- posedata
def test_json_roundtrip():
    tr = generate(SIDE, quality="poor", seed=2)
    back = PoseTrack.from_json(json.loads(json.dumps(tr.to_json())))
    assert back.view == SIDE and back.n == tr.n
    np.testing.assert_allclose(back.data[..., 2], tr.data[..., 2], atol=1e-3)


def test_legacy_coco_file_loads_with_spine_missing():
    coco = {"summary": {"fps": 30, "size": [1920, 1080]},
            "frames": [{"frame": 0, "t": 0, "box": None,
                        "kp": {"nose": [10, 10, .9], "l_shoulder": [100, 100, .9], "r_wrist": [5, 5, .2]}},
                       {"frame": 1, "t": .03, "box": None, "kp": None}]}
    tr = PoseTrack.from_json(coco, view=SIDE)
    assert tr.ok("l_shoulder")[0] and not tr.ok("r_wrist")[0]
    assert not tr.ok("spine_at_neck").any() and not tr.ok("l_shoulder")[1]


def test_view_is_required():
    with pytest.raises(PoseDataError):
        PoseTrack.from_json({"summary": {"fps": 30, "size": [10, 10]}, "frames": []})


def test_bad_shape_rejected():
    with pytest.raises(PoseDataError):
        PoseTrack(np.zeros((3, 17, 3)), 30, (10, 10), SIDE)


# --------------------------------------------------------------- synthetic
def test_synthetic_is_deterministic_and_rejects_unknown_faults():
    a, b = generate(SIDE, seed=4), generate(SIDE, seed=4)
    np.testing.assert_array_equal(np.nan_to_num(a.data), np.nan_to_num(b.data))
    with pytest.raises(KeyError):
        Swimmer({"bad_vibes": 1})


def test_every_workbook_issue_has_a_synthetic_fault(rules):
    assert {i["issue_id"] for i in rules["issues"]} == set(FAULTS)


# ----------------------------------------------------------------- metrics
def test_every_workbook_metric_is_implemented_for_its_view(rules):
    _check_registry(rules)  # raises on any gap or view mismatch
    assert {m["metric_id"] for m in rules["metrics"]} == set(REGISTRY)


def test_view_mismatch_is_caught(rules):
    bad = {**rules, "metrics": [dict(m, view=HEAD_ON) if m["metric_id"] == "head_lift_angle" else m
                                for m in rules["metrics"]]}
    with pytest.raises(WorkbookMismatch):
        _check_registry(bad)


@pytest.mark.parametrize("faults", [{}, {"over_gliding": 1.0}])
def test_stroke_rate_matches_truth_even_with_long_glide(faults):
    r = compute(generate(SIDE, Swimmer(faults, cycle_s=1.6), quality="typical", seed=1))
    rate = np.median([v for v in r["values"]["stroke_rate"] if v])
    assert abs(rate - 60 / 1.6) < 3


def test_direction_of_travel_does_not_change_measurements():
    sw = Swimmer({"dropped_elbow": 0.8})
    right = compute(generate(SIDE, sw, quality="clean", facing="right"))
    left = compute(generate(SIDE, sw, quality="clean", facing="left"))
    for mid in right["values"]:
        a = np.nanmedian([v for v in right["values"][mid] if v is not None])
        b = np.nanmedian([v for v in left["values"][mid] if v is not None])
        assert a == pytest.approx(b, abs=0.5), mid


# --------------------------------------------------------------- end to end
CONDITIONS = [("clean", "underwater"), ("typical", "underwater"), ("poor", "underwater"),
              ("typical", "deck"), ("poor", "deck")]


@pytest.mark.parametrize("quality,camera", CONDITIONS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_swimmer_triggers_nothing(rules, quality, camera, seed):
    report = analyze(session(quality=quality, camera=camera, seed=seed), rules, top_n=20)
    assert triggered(report) == set(), report["priorities"]


@pytest.mark.parametrize("quality", ["clean", "typical", "poor"])
@pytest.mark.parametrize("fault", FAULTS)
def test_each_fault_is_caught_with_an_underwater_camera(rules, fault, quality):
    report = analyze(session({fault: 1.0}, quality=quality, camera="underwater", seed=5), rules, top_n=20)
    assert fault in triggered(report), (fault, report["priorities"])


@pytest.mark.parametrize("fault", FAULTS)
def test_deck_camera_catches_or_honestly_skips_each_fault(rules, fault):
    report = analyze(session({fault: 1.0}, quality="typical", camera="deck", seed=5), rules, top_n=20)
    assert fault in triggered(report) or fault in report["issues_not_assessed"]
    assert triggered(report) <= {fault}, "a fault the swimmer does not have was reported"


def test_deck_camera_reports_why_underwater_issues_were_skipped(rules):
    report = analyze(session(quality="typical", camera="deck"), rules)
    assert "dropped_elbow" in report["issues_not_assessed"]
    assert "readable" in report["issues_not_assessed"]["dropped_elbow"] \
        or "no stroke cycles" in report["issues_not_assessed"]["dropped_elbow"]
    # above-water issues are still assessed from the deck
    for iid in ("late_breath", "crossover_entry", "low_body_rotation", "wide_recovery_straight_arm"):
        assert iid not in report["issues_not_assessed"], iid


def test_both_views_are_required_by_default(rules):
    with pytest.raises(MissingViewError):
        analyze({SIDE: generate(SIDE)}, rules)


def test_single_view_when_explicitly_allowed_marks_the_other_views_issues(rules):
    report = analyze({SIDE: generate(SIDE)}, rules, require=(SIDE,))
    assert "no head_on video" in report["issues_not_assessed"]["crossover_entry"]
    assert "dropped_elbow" not in report["issues_not_assessed"]


def test_track_under_wrong_view_key_is_rejected(rules):
    with pytest.raises(MissingViewError):
        analyze({SIDE: generate(HEAD_ON), HEAD_ON: generate(HEAD_ON)}, rules)


def test_report_carries_confidence_and_drills(rules):
    report = analyze(session({"dropped_elbow": 1.0}, seed=5), rules)
    m = report["measurements"]["catch_upper_arm_angle"]
    assert m["view"] == SIDE and 0 < m["confidence"] <= 1 and m["usable_cycles"] >= 3
    top = report["priorities"][0]
    assert top["issue_id"] == "dropped_elbow" and top["interventions"]
