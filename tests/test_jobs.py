"""Background analysis: upload returns at once, the report page waits, failures are contained."""
import html
import io
import json
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

PW = "correct horse 1"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from app import auth, db, jobs, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    from app.main import app
    yield app, pipeline, jobs, db, monkeypatch
    jobs.wait_all()


def browser(app, email="a@example.com"):
    c = TestClient(app)
    c.post("/signup", data={"name": "Alex", "email": email, "password": PW}, follow_redirects=False)
    return c


def upload(c):
    return c.post("/analyses", files={"side": ("s.mp4", io.BytesIO(b"side"), "video/mp4"),
                                      "head_on": ("h.mov", io.BytesIO(b"head"), "video/quicktime")},
                  follow_redirects=False)


def test_upload_returns_before_the_analysis_and_the_page_waits(env):
    app, pipeline, jobs, db, mp = env
    mp.setenv("SWIMFORM_JOBS", "thread")
    go, started = threading.Event(), threading.Event()
    real = pipeline.PROVIDERS["synthetic"]

    def slow(paths, names):
        started.set()
        assert go.wait(10)
        return real(paths, names)

    mp.setitem(pipeline.PROVIDERS, "synthetic", slow)
    c = browser(app)
    r = upload(c)
    assert r.status_code == 303
    url = r.headers["location"]
    assert started.wait(10)

    page = c.get(url)
    assert page.status_code == 200 and "Measuring your stroke" in page.text
    assert c.get(url + "/status").json() == {"status": "running"}
    assert "Measuring" in c.get("/").text
    assert browser(app, "b@example.com").get(url + "/status").status_code == 404   # not yours

    go.set()
    jobs.wait_all()
    assert c.get(url + "/status").json() == {"status": "done"}
    done = c.get(url)
    assert "Your stroke report" in done.text
    assert c.get("/").text.count('class="seg seg-') == 14


def test_a_crash_in_the_analysis_is_contained(env):
    app, pipeline, jobs, db, mp = env

    def broken(paths, names):
        raise ValueError("decoder exploded")

    mp.setitem(pipeline.PROVIDERS, "synthetic", broken)
    c = browser(app)
    url = upload(c).headers["location"]
    page = c.get(url)
    assert page.status_code == 500 and "Something went wrong on our end" in page.text
    assert "decoder exploded" not in page.text           # internals stay in the log
    assert "Didn't finish" in html.unescape(c.get("/").text)
    mp.setitem(pipeline.PROVIDERS, "synthetic", pipeline.pose_synthetic)
    assert upload(c).status_code == 303                   # and the next one still works


def test_restart_resumes_unfinished_work(env):
    app, pipeline, jobs, db, mp = env
    c = browser(app)
    with db.session() as s:
        uid = s.scalar(db.select(db.User.id))
    aid = pipeline.queue({"side": ("s.mp4", io.BytesIO(b"x")), "head_on": ("h.mov", io.BytesIO(b"y"))})
    with db.session() as s:
        s.add(db.Analysis(id=aid, user_id=uid, status="running"))        # was mid-run when the server died
        s.add(db.Analysis(id="gone00000000", user_id=uid, status="queued"))  # its files are gone
        s.commit()
    assert sorted(jobs.resume()) == sorted([aid, "gone00000000"])
    assert c.get(f"/analyses/{aid}/status").json() == {"status": "done"}
    assert c.get("/analyses/gone00000000/status").json() == {"status": "failed"}


def test_a_job_never_runs_twice(env):
    app, pipeline, jobs, db, mp = env
    calls = []
    real = pipeline.process
    mp.setattr(pipeline, "process", lambda aid, *a, **k: calls.append(aid) or real(aid, *a, **k))
    c = browser(app)
    aid = upload(c).headers["location"].split("/")[-1]
    jobs.work(aid)                                       # already done: claim fails, nothing runs
    assert calls == [aid]
    with open(pipeline.folder(aid) / "report.json") as f:
        assert json.load(f)["id"] == aid
