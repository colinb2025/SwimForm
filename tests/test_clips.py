"""
Any mix of clips, two at a minimum (one side, one head-on): each measurement comes from
the clip that read it best, a bad extra clip never sinks a good upload, and a first-time uploader
is shown how to film before they start.
"""
import io
import re
from pathlib import Path

import pytest

from swimform import rules as R
from swimform.analyze import MissingViewError, analyze
from swimform.schema import HEAD_ON, SIDE
from swimform.synthetic import Swimmer, generate, generate_down_and_back

WORKBOOK = str(Path(__file__).resolve().parents[1] / "rules" / "swim_rules.xlsx")


@pytest.fixture(scope="module")
def rules():
    r, errors = R.load(WORKBOOK)
    assert not errors
    return r


def clips(faults, side=("deck",), head=("deck",), seed=0):
    sw = Swimmer(faults)
    return {SIDE: [generate_down_and_back(sw, camera=c, quality="typical", seed=seed + i) for i, c in enumerate(side)],
            HEAD_ON: [generate(HEAD_ON, Swimmer(faults, cycle_s=1.52), camera=c, quality="typical", seed=seed + 50 + i, t0=3.7)
                      for i, c in enumerate(head)]}


def state(report, iid):
    for p in report["priorities"] + report["other_issues"]:
        if p["issue_id"] == iid:
            return "flagged:" + p["evidence"]
    if iid in report["issues_not_assessed"]:
        return "unchecked"
    return "clear:" + report["issues_clear"][iid]


# ------------------------------------------------------------------ analysis
def test_an_underwater_side_clip_turns_an_inference_into_a_measurement(rules):
    faults = {"dropped_elbow": 1.0}
    deck = analyze(clips(faults), rules, top_n=20)
    both = analyze(clips(faults, side=("deck", "underwater")), rules, top_n=20)
    assert state(deck, "dropped_elbow") == "flagged:proxy"
    assert state(both, "dropped_elbow") == "flagged:direct"
    m = both["measurements"]["catch_upper_arm_angle"]
    assert m["clip"] == 1 and both["clips"][SIDE][1]["measurements_used"] > 0     # taken from the underwater clip


def test_clip_order_does_not_matter(rules):
    faults = {"dropped_elbow": 1.0, "high_head_position": 0.9}
    a = analyze(clips(faults, side=("deck", "underwater")), rules, top_n=20)
    b = analyze(clips(faults, side=("underwater", "deck")), rules, top_n=20)
    ids = lambda r: sorted(p["issue_id"] for p in r["priorities"] + r["other_issues"])   # noqa: E731
    assert ids(a) == ids(b) and "dropped_elbow" in ids(a)


def test_a_deck_clip_never_drags_down_what_the_underwater_clip_saw(rules):
    faults = {"straight_arm_pull": 1.0}
    alone = analyze(clips(faults, side=("underwater",), head=("underwater",)), rules, top_n=20)
    mixed = analyze(clips(faults, side=("underwater", "deck"), head=("underwater", "deck")), rules, top_n=20)
    assert state(alone, "straight_arm_pull") == state(mixed, "straight_arm_pull") == "flagged:direct"
    assert set(mixed["issues_not_assessed"]) <= set(alone["issues_not_assessed"])


def test_clean_swimmer_stays_clean_with_every_clip_type(rules):
    report = analyze(clips({}, side=("deck", "underwater"), head=("deck", "underwater")), rules, top_n=20)
    assert not report["priorities"] and not report["other_issues"]
    assert report["stroke3d"]["used"] and set(report["stroke3d"]["clips"]) == {SIDE, HEAD_ON}


def test_single_tracks_still_work_and_a_view_is_still_required(rules):
    c = clips({})
    assert analyze({SIDE: c[SIDE][0], HEAD_ON: c[HEAD_ON][0]}, rules)["clips"][SIDE][0]["main"]
    with pytest.raises(MissingViewError):
        analyze({SIDE: c[SIDE], HEAD_ON: []}, rules)


# ------------------------------------------------------------------ through the web app
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import auth, db, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    for var in ("SWIMFORM_SIGNUP", "SWIMFORM_REVIEW", "SWIMFORM_POSE", "SWIMFORM_FREE_ANALYSES", "SWIMFORM_STORAGE"):
        monkeypatch.delenv(var, raising=False)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    from app.main import app
    c = TestClient(app)
    assert c.post("/signup", data={"email": "s@example.com", "password": "correct horse 1"}, follow_redirects=False).status_code == 303
    return c


def send(c, **slots):
    files = {slot: (name, io.BytesIO(slot.encode() + name.encode()), "video/mp4") for slot, name in slots.items()}
    return c.post("/analyses", files=files, follow_redirects=False)


def view(c, r):
    from app import pipeline, presenter
    from app.main import rules as app_rules
    aid = r.headers["location"].split("/")[-1]
    return presenter.build(pipeline.load(aid), app_rules()), pipeline.load(aid)


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).replace("&#39;", "'")


def test_upload_page_offers_four_slots(client):
    page = client.get("/analyses/new").text
    for name in ("side", "side_uw", "head_on", "head_on_uw"):
        assert f'name="{name}"' in page
    assert "required" not in page.split('name="side_uw"')[1][:80]


@pytest.mark.parametrize("slots", [
    {"side": "a.mp4", "head_on": "b.mp4"},
    {"side_uw": "a.mp4", "head_on_uw": "b.mp4"},                 # underwater only is a full upload
    {"side": "a_deck.mp4", "head_on_uw": "b.mp4"},
    {"side": "a_deck.mp4", "side_uw": "c.mp4", "head_on": "b_deck.mp4", "head_on_uw": "d.mp4"},
])
def test_any_mix_with_one_clip_per_view_is_analysed(client, slots):
    r = send(client, **slots)
    assert r.status_code == 303
    v, record = view(client, r)
    assert [c["slot"] for c in record["clips"]] == list(slots) and all(c["used"] for c in record["clips"])
    assert v["unused_clips"] == []


@pytest.mark.parametrize("slots, missing", [
    ({"side": "a.mp4", "side_uw": "b.mp4"}, "Add a head-on clip"),
    ({"head_on_uw": "b.mp4"}, "Add a side-view clip"),
    ({}, "Add a side-view clip"),
])
def test_two_clips_of_the_same_view_are_not_enough(client, slots, missing):
    r = send(client, **slots)
    assert r.status_code == 422 and missing in r.text


def test_a_bad_extra_clip_is_set_aside_not_fatal(client):
    """The underwater 'side' clip is really a head-on view: it's left out and the report says so."""
    r = send(client, side="lap.mp4", head_on="wall.mp4", side_uw="front_view.mp4")
    assert r.status_code == 303
    v, record = view(client, r)
    bad = next(c for c in record["clips"] if c["slot"] == "side_uw")
    assert not bad["used"] and "swapped" in bad["problems"][0]
    page = text(client.get(r.headers["location"]).text)
    assert "We left out your side view, underwater clip." in page and "Your stroke report" in page


def test_no_usable_clip_for_a_view_is_still_rejected_with_reasons(client):
    r = send(client, side="front_view.mp4", side_uw="head_cam.mp4", head_on="wall.mp4")
    page = text(client.get(r.headers["location"]).text)
    assert "We couldn't use these clips" in page
    assert "Above-water clip" in page and "Underwater clip" in page


def test_report_does_not_coach_the_next_upload(client):
    r = send(client, side="lap_deck.mp4", head_on="wall_deck.mp4")
    page = text(client.get(r.headers["location"]).text)
    assert "next upload" not in page and "We left out" not in page


def test_first_upload_goes_through_the_filming_walkthrough_once(client, monkeypatch):
    monkeypatch.setenv("SWIMFORM_FIRST_RUN", "1")
    r = client.get("/analyses/new", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/start"
    page = text(client.get("/start").text)
    assert "Before your first upload" in page and "A few clean strokes is enough" in page
    assert "Above-water clips are all you need" in page
    assert client.post("/start", follow_redirects=False).headers["location"] == "/analyses/new"
    assert client.get("/analyses/new", follow_redirects=False).status_code == 200      # never shown again
    assert client.get("/start").status_code == 200                                     # but still reachable
    from fastapi.testclient import TestClient as TC
    from app.main import app
    assert TC(app).get("/start", follow_redirects=False).headers["location"].startswith("/login")


def test_old_reports_without_clip_records_still_open(client):
    import json
    from app import pipeline
    r = send(client, side="a.mp4", head_on="b.mp4")
    aid = r.headers["location"].split("/")[-1]
    path = pipeline.folder(aid) / "report.json"
    record = json.loads(path.read_text())
    record.pop("clips")
    path.write_text(json.dumps(record))
    page = client.get(f"/analyses/{aid}")
    assert page.status_code == 200 and "We left out" not in page.text
