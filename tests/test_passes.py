"""
Down-and-back side clips: split at the turn, each arm measured from the length where it
faced the camera. The bar these tests hold:

  - a 50 splits into two passes, travelling opposite ways, with opposite near sides,
    and the turn is cut out;
  - a fault on ONE arm is caught on a down-and-back clip and attributed to that arm;
  - a one-way clip never claims to have checked the arm it couldn't see;
  - a clean swimmer still triggers nothing, on any footage.
"""
from pathlib import Path

import numpy as np
import pytest

from swimform import passes as P
from swimform import rules as R
from swimform import stroke3d as S3
from swimform.analyze import analyze
from swimform.schema import HEAD_ON, IDX, SIDE
from swimform.synthetic import ARM_FAULTS, Swimmer, generate, generate_down_and_back, truth3d
from swimform.viewcheck import check_clip

WORKBOOK = str(Path(__file__).resolve().parents[1] / "rules" / "swim_rules.xlsx")


@pytest.fixture(scope="module")
def rules():
    r, errors = R.load(WORKBOOK)
    assert not errors, errors
    return r


def swimmer(faults=None, arm=None, cycle_s=1.6):
    return Swimmer(faults or {}, cycle_s=cycle_s, arm_faults=arm)


def session(faults=None, arm=None, down_and_back=True, seed=0, **kw):
    sw = swimmer(faults, arm)
    side = (generate_down_and_back(sw, seed=seed, **kw) if down_and_back
            else generate(SIDE, sw, seed=seed, **kw))
    head = generate(HEAD_ON, swimmer(faults, arm, cycle_s=1.52), seed=seed + 100, t0=3.7, **kw)
    return {SIDE: side, HEAD_ON: head}


def triggered(report):
    return {p["issue_id"] for p in report["priorities"]} | {o["issue_id"] for o in report["other_issues"]}


def finding(report, iid):
    return next((p for p in report["priorities"] if p["issue_id"] == iid), None) or \
        next((o for o in report["other_issues"] if o["issue_id"] == iid), None)


# ------------------------------------------------------------------ synthetic
def test_one_arm_faults_only_change_that_arm():
    t = np.arange(0, 3.2, 1 / 60)
    clean, left = truth3d(Swimmer(), t), truth3d(Swimmer(arm_faults={"l": {"dropped_elbow": 1}}), t)
    for k in ("r_elbow", "r_wrist"):
        np.testing.assert_allclose(clean[:, IDX[k]], left[:, IDX[k]])
    assert not np.allclose(clean[:, IDX["l_wrist"]], left[:, IDX["l_wrist"]])   # forearm drops


def test_arm_fault_validation():
    with pytest.raises(KeyError):
        Swimmer(arm_faults={"left": {"dropped_elbow": 1}})
    with pytest.raises(KeyError):
        Swimmer(arm_faults={"l": {"late_breath": 1}})   # a whole-body fault, not one arm's
    assert set(ARM_FAULTS) <= set(Swimmer().faults)


# ------------------------------------------------------------------ splitting
@pytest.mark.parametrize("camera", ["underwater", "deck"])
@pytest.mark.parametrize("quality", ["clean", "typical", "poor"])
@pytest.mark.parametrize("first_near", ["r", "l"])
def test_down_and_back_splits_into_two_opposite_passes(camera, quality, first_near):
    tr = generate_down_and_back(swimmer(), camera=camera, quality=quality, first_near=first_near, seed=3)
    ps = P.split(tr)
    assert len(ps) == 2
    a, b = ps
    assert a.direction == -b.direction != 0
    assert a.near == first_near and b.near == ("l" if first_near == "r" else "r")
    turn_start, turn_end = tr.meta["passes"][0]["start"] + 300, tr.meta["passes"][1]["start"]
    assert a.end <= turn_start and b.start >= turn_end   # the turn is cut out
    assert b.start - turn_end >= int(P.TRIM_AFTER_S * tr.fps) - 1   # and so is the breakout


def test_one_way_clip_is_one_pass_covering_the_clip():
    tr = generate(SIDE, swimmer(), quality="typical", seed=1)
    ps = P.split(tr)
    assert len(ps) == 1 and (ps[0].start, ps[0].end) == (0, tr.n) and ps[0].near == "r"


def test_weak_visibility_takes_near_side_from_direction():
    """Both arms equally visible on the way back: the camera stayed on one deck, so the
    near side must be the opposite of the way down."""
    tr = generate_down_and_back(swimmer(), quality="clean", seed=2)
    back = tr.meta["passes"][1]["start"]
    c = tr.data[back:, :, 2]
    for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        both = np.maximum(c[:, IDX[f"l_{j}"]], c[:, IDX[f"r_{j}"]])
        c[:, IDX[f"l_{j}"]] = c[:, IDX[f"r_{j}"]] = both
    a, b = P.split(tr)
    assert (a.near, a.near_from) == ("r", "visibility")
    assert (b.near, b.near_from) == ("l", "direction")


def test_passes_are_head_on_proof():
    with pytest.raises(ValueError):
        P.split(generate(HEAD_ON, swimmer()))


# --------------------------------------------------------------- the payoff
@pytest.mark.parametrize("fault", ["dropped_elbow", "straight_arm_pull", "short_pull"])
@pytest.mark.parametrize("arm", ["r", "l"])
@pytest.mark.parametrize("quality", ["clean", "typical", "poor"])
def test_down_and_back_catches_a_one_arm_fault_and_names_the_arm(rules, fault, arm, quality):
    report = analyze(session(arm={arm: {fault: 1.0}}, quality=quality, seed=1), rules, top_n=20)
    hit = finding(report, fault)
    assert hit is not None, f"{fault} on the {arm} arm was missed"
    assert hit["arms"]["checked"] == ["r", "l"]
    assert hit["arms"]["affected"] == [arm]


def test_one_way_clip_misses_the_far_arm_and_says_so(rules):
    """The old single-pass clip: camera on the swimmer's right, fault on the left arm.
    It can't be seen, so it must not be cleared without saying only one arm was looked at."""
    report = analyze(session(arm={"l": {"dropped_elbow": 1.0}}, down_and_back=False, quality="typical"),
                     rules, top_n=20)
    assert "dropped_elbow" not in triggered(report)
    arms = report["issue_arms"]["dropped_elbow"]
    assert arms["checked"] == ["r"] and arms["affected"] == []
    assert "left arm never faced the camera" in arms["unseen"]["l"]


def test_both_arms_affected_is_reported_as_both(rules):
    report = analyze(session({"dropped_elbow": 1.0}, quality="typical", seed=2), rules, top_n=20)
    assert finding(report, "dropped_elbow")["arms"]["affected"] == ["r", "l"]


@pytest.mark.parametrize("camera", ["underwater", "deck"])
@pytest.mark.parametrize("quality", ["clean", "typical", "poor"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_swimmer_down_and_back_triggers_nothing(rules, camera, quality, seed):
    report = analyze(session(camera=camera, quality=quality, seed=seed), rules, top_n=20)
    assert triggered(report) == set()


@pytest.mark.parametrize("arm", ["r", "l"])
def test_deck_footage_estimates_a_one_arm_short_pull_on_the_right_arm(rules, arm):
    report = analyze(session(arm={arm: {"short_pull": 1.0}}, camera="deck", quality="typical", seed=5),
                     rules, top_n=20)
    hit = finding(report, "short_pull")
    assert hit["evidence"] == "proxy" and hit["arms"]["affected"] == [arm]


def test_measurements_carry_both_arms(rules):
    report = analyze(session(arm={"l": {"dropped_elbow": 1.0}}, quality="typical"), rules)
    m = report["measurements"]["catch_upper_arm_angle"]
    assert set(m["arms"]) == {"r", "l"} and m["arm"] == "l"
    assert m["value"] == m["arms"]["l"]["value"] > m["arms"]["r"]["value"]
    assert report["side_passes"] and len(report["side_passes"]) == 2


def test_whole_body_checks_pool_both_lengths(rules):
    one = analyze(session(down_and_back=False, quality="typical"), rules)
    both = analyze(session(quality="typical"), rules)
    assert both["measurements"]["stroke_rate"]["usable_cycles"] > one["measurements"]["stroke_rate"]["usable_cycles"]
    assert "arms" not in both["measurements"]["stroke_rate"]


# ------------------------------------------------------------- 3D and checks
def test_fused_stroke_from_down_and_back_matches_ground_truth():
    from tests.test_pipeline import _true_recovery_angle
    tracks = session({"wide_recovery_straight_arm": 0.5}, quality="typical", seed=4)
    r = S3.compute(tracks[SIDE], tracks[HEAD_ON])
    assert r["reason"] is None and r["stroke"].agreement > 0.95
    assert r["values"]["recovery_elbow_angle"] == pytest.approx(
        _true_recovery_angle(Swimmer({"wide_recovery_straight_arm": 0.5})), abs=5)


def test_clip_check_accepts_a_turn_in_the_side_clip():
    c = check_clip(generate_down_and_back(swimmer(), quality="typical"), SIDE)
    assert c["ok"] and c["cycles"] >= 8 and len(c["passes"]) == 2


def test_clip_check_explains_a_too_short_down_and_back():
    c = check_clip(generate_down_and_back(swimmer(), lap_s=2.0), SIDE)
    assert not c["ok"] and "Film a 50" in c["problems"][0]
