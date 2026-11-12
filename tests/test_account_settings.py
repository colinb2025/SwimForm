"""Account settings: name, email, password, other devices, deleting everything."""
import html
import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

PW = "correct horse 1"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from app import account, auth, db, mailer, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    account.limiter.fails.clear()
    sent = []
    monkeypatch.setattr(mailer, "send", lambda to, subject, body: sent.append((to, subject, body)))
    from app.main import app
    return app, db, pipeline, sent


def browser(app, email="a@example.com", name="Alex"):
    c = TestClient(app)
    assert c.post("/signup", data={"name": name, "email": email, "password": PW},
                  follow_redirects=False).status_code == 303
    return c


def login(app, email="a@example.com", pw=PW):
    c = TestClient(app)
    return c, c.post("/login", data={"email": email, "password": pw}, follow_redirects=False)


def t(r):
    return html.unescape(r.text)


def test_settings_need_login_and_are_linked_from_the_menu(env):
    app, *_ = env
    assert TestClient(app).get("/account", follow_redirects=False).headers["location"].startswith("/login")
    c = browser(app)
    assert 'href="/account"' in c.get("/").text
    page = t(c.get("/account"))
    assert "Account settings" in page and "a@example.com" in page and "Delete account" in page


def test_change_name(env):
    app, *_ = env
    c = browser(app)
    r = c.post("/account/name", data={"name": "  Colin Bitz "})
    assert "Name saved." in t(r)
    assert "Colin's reports" in t(c.get("/"))


def test_change_email_needs_current_password_and_tells_the_old_address(env):
    app, db, _, sent = env
    c = browser(app)
    browser(app, "taken@example.com")
    r = c.post("/account/email", data={"email": "new@example.com", "password": "wrong password!"})
    assert r.status_code == 422 and "isn't your current password" in t(r)
    r = c.post("/account/email", data={"email": "taken@example.com", "password": PW})
    assert r.status_code == 422 and "already uses that email" in t(r)
    r = c.post("/account/email", data={"email": "not-an-email", "password": PW})
    assert r.status_code == 422 and "Enter an email address" in t(r)
    r = c.post("/account/email", data={"email": "New@Example.com", "password": PW})
    assert r.status_code == 200 and "Email changed" in t(r)
    assert sent[-1][0] == "a@example.com" and "new@example.com" in sent[-1][2]
    assert login(app, "new@example.com")[1].status_code == 303
    assert login(app, "a@example.com")[1].status_code == 401


def test_change_password_logs_out_other_devices_but_not_this_one(env):
    app, *_ = env
    c = browser(app)
    other, r = login(app)
    assert r.status_code == 303 and other.get("/", follow_redirects=False).status_code == 200
    r = c.post("/account/password", data={"current": "wrong password!", "new": "a brand new pass"})
    assert r.status_code == 422
    r = c.post("/account/password", data={"current": PW, "new": "short"})
    assert r.status_code == 422 and "at least 10" in t(r)
    r = c.post("/account/password", data={"current": PW, "new": "a brand new pass"})
    assert "Password saved" in t(r)
    assert c.get("/", follow_redirects=False).status_code == 200            # still logged in here
    assert other.get("/", follow_redirects=False).status_code == 303        # kicked out there
    assert login(app, pw="a brand new pass")[1].status_code == 303
    assert login(app)[1].status_code == 401


def test_log_out_other_devices(env):
    app, *_ = env
    c = browser(app)
    other, _ = login(app)
    assert "Every other device" in t(c.post("/account/sessions"))
    assert other.get("/", follow_redirects=False).status_code == 303
    assert c.get("/", follow_redirects=False).status_code == 200


def test_wrong_current_password_is_rate_limited(env):
    app, *_ = env
    c = browser(app)
    for _ in range(5):
        c.post("/account/password", data={"current": "nope nope nope", "new": "a brand new pass"})
    r = c.post("/account/password", data={"current": PW, "new": "a brand new pass"})
    assert r.status_code == 422 and "Too many wrong passwords" in t(r)


def test_google_only_account_can_set_a_password_without_a_current_one(env):
    app, db, *_ = env
    c = browser(app)
    with db.session() as s:
        u = db.user_by_email(s, "a@example.com")
        u.password_hash, u.google_sub = "", "google-123"
        s.commit()
    page = t(c.get("/account"))
    assert "Set a password" in page and "Google: connected" in page and "Current password" not in page
    assert "Password saved" in t(c.post("/account/password", data={"new": "now i have one"}))
    assert login(app, pw="now i have one")[1].status_code == 303


def test_delete_account_removes_everything(env):
    app, db, pipeline, _ = env
    c = browser(app)
    keep = browser(app, "b@example.com")
    r = c.post("/analyses", files={"side": ("s.mp4", io.BytesIO(b"side"), "video/mp4"),
                                   "head_on": ("h.mov", io.BytesIO(b"head"), "video/quicktime")},
               follow_redirects=False)
    aid = r.headers["location"].split("/")[-1]
    r2 = keep.post("/analyses", files={"side": ("s.mp4", io.BytesIO(b"x"), "video/mp4"),
                                       "head_on": ("h.mov", io.BytesIO(b"y"), "video/quicktime")},
                   follow_redirects=False)
    other_aid = r2.headers["location"].split("/")[-1]
    assert pipeline.folder(aid).exists()

    r = c.post("/account/delete", data={"confirm": "yes", "password": PW})
    assert r.status_code == 422 and 'Type "delete"' in t(r)
    r = c.post("/account/delete", data={"confirm": "delete", "password": "wrong password!"})
    assert r.status_code == 422
    r = c.post("/account/delete", data={"confirm": "Delete", "password": PW}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?deleted=1"
    assert "has been deleted" in t(TestClient(app).get("/login?deleted=1"))

    assert not pipeline.folder(aid).exists()                  # clips and report gone
    with db.session() as s:
        assert db.user_by_email(s, "a@example.com") is None
        assert s.get(db.Analysis, aid) is None
        assert s.scalar(db.select(db.LoginSession).join(db.User, isouter=True).where(db.User.id.is_(None))) is None
    assert login(app)[1].status_code == 401
    assert pipeline.folder(other_aid).exists()                # someone else's data untouched
    assert keep.get(f"/analyses/{other_aid}").status_code == 200
