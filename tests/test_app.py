"""Upload -> clip check -> analysis -> report, through the web app."""
import io
import re

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from swimform.posedata import PoseTrack
from swimform.schema import HEAD_ON, SIDE
from swimform.synthetic import Swimmer, generate, truth3d
from swimform.viewcheck import check_clip, guess_view


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A logged-in browser."""
    from app import auth, db, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    from app.main import app
    c = TestClient(app)
    r = c.post("/signup", data={"name": "Test Swimmer", "email": "swimmer@example.com", "password": "correct horse 1"},
               follow_redirects=False)
    assert r.status_code == 303
    return c


def upload(client, side=("lap1.mp4", b"side clip"), head_on=("lap2.mov", b"head clip")):
    files = {}
    if side:
        files["side"] = (side[0], io.BytesIO(side[1]), "video/mp4")
    if head_on:
        files["head_on"] = (head_on[0], io.BytesIO(head_on[1]), "video/quicktime")
    return client.post("/analyses", files=files, follow_redirects=False)


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def view_for(client, response):
    from app import pipeline, presenter
    from app.main import rules
    aid = response.headers["location"].split("/")[-1]
    return presenter.build(pipeline.load(aid), rules())


# ------------------------------------------------------------------- pages
def test_upload_page_shows_both_camera_spots(client):
    r = client.get("/analyses/new")
    assert r.status_code == 200
    assert 'name="side"' in r.text and 'name="head_on"' in r.text
    assert "Where to put the two cameras" in r.text


def test_good_pair_produces_a_report(client):
    r = upload(client)
    assert r.status_code == 303
    rep = client.get(r.headers["location"])
    assert rep.status_code == 200
    t = text(rep.text)
    assert "Your stroke report" in t and "We checked 14 parts" in t
    assert "Demo data" in t                       # synthetic pose is never passed off as real
    assert rep.text.count('class="lane lane-') == 14


def test_report_puts_every_issue_in_exactly_one_lane(client):
    from app.main import rules
    names = [l["name"] for l in view_for(client, upload(client))["lanes"]]
    assert len(names) == len(set(names)) == len(rules()["issues"])


def test_proxy_findings_are_marked_as_estimates(client):
    # file names containing "deck" simulate a deck camera in the synthetic pose step
    for i in range(60):
        r = upload(client, side=("deck_side.mp4", b"s%d" % i), head_on=("deck_head.mov", b"h%d" % i))
        v = view_for(client, r)
        est = [p for p in v["priorities"] if p["estimated"]]
        if est:
            page = text(client.get(r.headers["location"]).text)
            assert "Estimated from what the cameras could see" in page
            assert "may be" in est[0]["observed"]
            return
    pytest.skip("no proxy finding in 60 synthetic sessions")


def test_measurement_names_are_sentence_case(client):
    r = upload(client, side=("deck_a.mp4", b"1"), head_on=("deck_b.mov", b"2"))
    for row in view_for(client, r)["table"]:
        assert row["name"][0].isupper(), row["name"]


def test_swapped_clips_are_sorted_out_for_the_swimmer(client):
    """A head-on clip in the side box and a side clip in the head-on box: just use them."""
    r = upload(client, side=("headon_lap.mp4", b"a"), head_on=("side_lap.mov", b"b"))
    assert r.status_code == 303
    page = text(client.get(r.headers["location"]).text)
    assert "Your stroke report" in page and "couldn't use" not in page
    from app import pipeline
    record = pipeline.load(r.headers["location"].split("/")[-1])
    assert {c["slot"]: c["view"] for c in record["clips"]} == {"side": "head_on", "head_on": "side"}


def test_two_clips_of_the_same_angle_are_explained(client):
    r = upload(client, side=("lap.mp4", b"a"), head_on=("side_again.mov", b"b"))
    assert r.status_code == 303                   # clip checks run in the job, after upload
    page = text(client.get(r.headers["location"]).text)
    assert "We couldn't use these clips" in page
    assert "Both of your clips are side views. We also need a head-on clip" in page
    assert "We've kept this clip" in page         # the side clip itself needs no change
    assert "Clips need re-filming" in text(client.get("/").text)


def test_a_turned_back_upload_is_fixed_by_adding_only_the_missing_clip(client):
    import io
    r = upload(client, side=("lap.mp4", b"a"), head_on=("side_again.mov", b"b"))
    where = r.headers["location"]
    html = client.get(where).text
    assert 'name="head_on"' in html and 'name="side"' not in html      # only the missing angle is asked for

    assert client.post(f"{where}/clips", files={}).status_code == 422                    # nothing added
    bad = client.post(f"{where}/clips", files={"head_on": ("notes.txt", io.BytesIO(b"x"), "text/plain")})
    assert bad.status_code == 422 and "look like a video" in text(bad.text)

    fixed = client.post(f"{where}/clips", files={"head_on": ("wall.mp4", io.BytesIO(b"c"), "video/mp4")},
                        follow_redirects=False)
    assert fixed.status_code == 303 and fixed.headers["location"] == where               # same analysis, now a report
    assert "Your stroke report" in text(client.get(where).text)
    from app import pipeline
    record = pipeline.load(where.split("/")[-1])
    assert record["files"] == {"side": "lap.mp4", "head_on": "wall.mp4"}                 # the good clip was kept
    assert client.post(f"{where}/clips", files={"head_on": ("x.mp4", io.BytesIO(b"d"), "video/mp4")}).status_code == 404


def test_only_the_owner_can_fix_an_upload(client):
    import io
    from fastapi.testclient import TestClient
    r = upload(client, side=("lap.mp4", b"a"), head_on=("side_again.mov", b"b"))
    other = TestClient(client.app)
    other.post("/signup", data={"email": "other@example.com", "password": "correct horse 1"})
    resp = other.post(f"{r.headers['location']}/clips", files={"head_on": ("wall.mp4", io.BytesIO(b"c"), "video/mp4")})
    assert resp.status_code == 404


def test_missing_and_non_video_files_are_rejected(client):
    r = upload(client, head_on=None)
    assert r.status_code == 422 and "Add your head-on clip." in r.text
    r = upload(client, side=("notes.txt", b"x"))
    assert r.status_code == 422 and "look like a video" in text(r.text)
    r = upload(client, side=("empty.mp4", b""))
    assert r.status_code == 422 and "empty" in r.text


def test_unknown_or_malformed_report_ids_are_404(client):
    assert client.get("/analyses/nope12345678").status_code == 404
    assert client.get("/analyses/..%2F..%2Fetc").status_code == 404


# -------------------------------------------------------------- clip check
def _rotated_side(angle_deg):
    t = np.arange(360) / 30
    J = truth3d(Swimmer(), t)
    a = np.radians(angle_deg)
    x = J[..., 0] * np.cos(a) + J[..., 2] * np.sin(a)
    d = np.zeros(J.shape[:2] + (3,))
    d[..., 0], d[..., 1], d[..., 2] = (0.5 + x * 0.4) * 1920, 430 + J[..., 1] * 0.4 * 1920, 0.9
    return PoseTrack(d, 30, (1920, 1080), SIDE)


@pytest.mark.parametrize("view", [SIDE, HEAD_ON])
@pytest.mark.parametrize("camera", ["underwater", "deck"])
def test_correct_clips_pass(view, camera):
    assert check_clip(generate(view, Swimmer(), camera=camera, quality="poor"), view)["ok"]


@pytest.mark.parametrize("angle", [45, 70])
def test_diagonal_camera_is_flagged(angle):
    res = check_clip(_rotated_side(angle), SIDE)
    assert not res["ok"] and "diagonal" in res["problems"][0]


def test_slightly_off_square_is_accepted():
    assert guess_view(_rotated_side(12))[0] == SIDE


def test_too_short_a_clip_is_flagged():
    res = check_clip(generate(SIDE, Swimmer(), duration_s=3.0), SIDE)
    assert not res["ok"] and "stroke cycle" in res["problems"][0]


def test_swimmer_mostly_missing_is_flagged():
    tr = generate(SIDE, Swimmer())
    tr.data[: int(tr.n * 0.8), :, 2] = 0.0
    tr.data[: int(tr.n * 0.8), :, :2] = np.nan
    res = check_clip(tr, SIDE)
    assert not res["ok"] and "find the swimmer" in res["problems"][0]


# --------------------------------------------------------- snapshots & pages
from swimform.analyze import analyze as _analyze
from swimform.synthetic import FAULTS


@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_gets_a_posture_snapshot(fault):
    from app import snapshots
    tracks = {SIDE: generate(SIDE, Swimmer({fault: 1.0}), seed=3),
              HEAD_ON: generate(HEAD_ON, Swimmer({fault: 1.0}, cycle_s=1.52), seed=4, t0=3.7)}
    rep = _analyze(tracks)
    out = snapshots.svg(rep["average_strokes"], fault)
    assert out is not None, fault
    markup, moment = out
    assert markup.startswith("<svg") and "hot" in markup and moment


def test_findings_carry_snapshot_and_measured_caption(client):
    v = view_for(client, upload(client, side=("a.mp4", b"7"), head_on=("b.mov", b"8")))
    for p in v["priorities"]:
        assert p["caption"] and "We flag" in p["caption"]
    assert any(p["snapshot"] for p in v["priorities"]) or not v["priorities"]


def test_snapshot_is_skipped_when_joints_were_not_seen():
    from app import snapshots
    blank = [[None] * 15 for _ in range(100)]
    assert snapshots.svg({SIDE: blank, HEAD_ON: blank}, "dropped_elbow") is None


def test_guide_lists_what_each_clip_checks(client):
    r = client.get("/guide")
    assert r.status_code == 200
    t = text(r.text)
    assert "How to film your clips" in t and "Estimated, unless you add an underwater clip" in t
    assert "Elbow at the catch" in t and "Hand entry line" in t


def test_fonts_are_self_hosted(client):
    page = client.get("/").text
    assert "googleapis" not in page and "/static/fonts/fonts.css" in page
    assert client.get("/static/fonts/fonts.css").status_code == 200
    assert client.get("/static/fonts/archivo-latin-wdth-normal.woff2").status_code == 200


def test_pages_load_their_scripts_and_work_without_them(client):
    page = client.get("/analyses/new").text
    assert "/static/upload.js" in page and 'action="/analyses"' in page   # plain form post fallback
    rep = client.get(upload(client).headers["location"]).text
    assert "/static/report.js" in rep and "data-report-id" in rep


# ------------------------------------------------------------ down and back
def test_report_names_the_arm_a_one_sided_fault_is_on(client, monkeypatch):
    from app import pipeline
    monkeypatch.setattr(pipeline, "_synthetic_swimmer",
                        lambda seed: Swimmer(cycle_s=1.6, arm_faults={"l": {"dropped_elbow": 1.0}}))
    r = upload(client, side=("side.mp4", b"50"), head_on=("head.mov", b"hd"))
    page = text(client.get(r.headers["location"]).text)
    assert "Dropped elbow" in page and "Seen on your left arm only. Your right arm looks fine." in page


def test_one_way_side_clip_says_which_arm_was_checked(client, monkeypatch):
    from app import pipeline
    monkeypatch.setattr(pipeline, "_synthetic_swimmer",
                        lambda seed: Swimmer(cycle_s=1.6, arm_faults={"l": {"dropped_elbow": 1.0}}))
    r = upload(client, side=("side_oneway.mp4", b"25"), head_on=("head.mov", b"hd"))
    v = view_for(client, r)
    assert "Dropped elbow" not in {p["name"] for p in v["priorities"]}
    assert v["one_side"]["Dropped elbow"] == "right arm only"


def test_arm_notes():
    from app.presenter import _arm_note, _checked_one_side
    both = {"limb": "arm", "checked": ["r", "l"], "affected": ["r", "l"], "unseen": {}}
    assert _arm_note(both) == "Seen on both arms."
    one = {"limb": "leg", "checked": ["r"], "affected": ["r"], "unseen": {"l": "x"}}
    assert _arm_note(one).startswith("Seen on your right leg.") and "film a 50" in _arm_note(one)
    assert _checked_one_side(one) == "right leg only"
    assert _arm_note({"limb": "arm", "checked": ["r", "l"], "affected": [], "unseen": {}}) is None


def test_guide_asks_for_a_50_from_the_side(client):
    page = text(client.get("/guide").text)
    assert "Film a 50" in page and "same side of the pool" in page


def test_report_groups_fixes_as_pool_and_mobility_and_strength(client, monkeypatch):
    from app import pipeline
    monkeypatch.setattr(pipeline, "_synthetic_swimmer", lambda seed: Swimmer({"dropped_elbow": 1.0}, cycle_s=1.6))
    r = upload(client, side=("side.mp4", b"me"), head_on=("head.mov", b"ms"))
    html = client.get(r.headers["location"]).text
    page = text(html)
    assert "Mobility and strength" in page and "On land" not in page
    assert "External rotation at 90°" in page and "Overhead kettlebell carry" in page and "Long dog" in page
    assert "This is training, not treatment" in page
    assert 'class="more-drills"' in html                             # past the first three, tucked away


def test_second_report_says_what_changed_since_the_first(client, monkeypatch):
    from app import pipeline
    swimmer = {"now": Swimmer({"dropped_elbow": 1.0, "high_head_position": 1.0}, cycle_s=1.6)}
    monkeypatch.setattr(pipeline, "_synthetic_swimmer", lambda seed: swimmer["now"])
    first = upload(client, side=("a.mp4", b"1"), head_on=("b.mov", b"2"))
    page = text(client.get(first.headers["location"]).text)
    assert "Since your last report" not in page and "What now?" in page

    swimmer["now"] = Swimmer({"dropped_elbow": 1.0, "crossover_entry": 1.0}, cycle_s=1.6)   # head fixed, new fault
    second = upload(client, side=("c.mp4", b"3"), head_on=("d.mov", b"4"))
    page = text(client.get(second.headers["location"]).text)
    assert "Since your last report" in page and "Fixed: Head position." in page
    assert "1 still to work on." in page and "New this time: Crossover entry." in page
    assert "Also in your last report" in page and "New since your last report" in page
    home = text(client.get("/").text)
    assert "2 to work on: " in home and "Dropped elbow" in home
    assert "Since your last report" not in text(client.get(first.headers["location"]).text)   # the first has nothing before it


def test_checks_list_uses_plain_names_for_what_passed(client, monkeypatch):
    from app import pipeline
    monkeypatch.setattr(pipeline, "_synthetic_swimmer", lambda seed: Swimmer({"dropped_elbow": 1.0}, cycle_s=1.6))
    r = upload(client, side=("a.mp4", b"1"), head_on=("b.mov", b"2"))
    html = client.get(r.headers["location"]).text
    checks = text(html.split('id="all-h"')[1].split("</section>")[0])
    assert "Dropped elbow" in checks                       # a fault keeps its name
    assert "Hand entry line" in checks and "Kick width" in checks
    assert "Crossover entry" not in checks and "Scissor kick" not in checks   # fault names don't sit under "Looks good"
