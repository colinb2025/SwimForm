"""Accounts: sign up, log in, log out, privacy between accounts, request protections."""
import html
import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

PW = "correct horse 1"


@pytest.fixture()
def app_(tmp_path, monkeypatch):
    from app import auth, db, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    from app.main import app
    return app


def browser(app_):
    return TestClient(app_)


def signup(c, email="a@example.com", pw=PW, name="Alex"):
    return c.post("/signup", data={"name": name, "email": email, "password": pw}, follow_redirects=False)


def login(c, email="a@example.com", pw=PW):
    return c.post("/login", data={"email": email, "password": pw}, follow_redirects=False)


def analyse(c):
    return c.post("/analyses", files={"side": ("s.mp4", io.BytesIO(b"side"), "video/mp4"),
                                      "head_on": ("h.mov", io.BytesIO(b"head"), "video/quicktime")},
                  follow_redirects=False)


# ------------------------------------------------------------------ passwords
def test_passwords_are_hashed_with_salt_and_verify():
    from app.auth import hash_password, verify_password
    a, b = hash_password(PW), hash_password(PW)
    assert a != b and PW not in a and a.startswith("scrypt$")
    assert verify_password(PW, a) and not verify_password("wrong password", a)
    assert not verify_password(PW, "garbage")


# ------------------------------------------------------------------ flows
def test_anonymous_visitors_are_sent_to_login(app_):
    c = browser(app_)
    for path in ("/", "/analyses/new", "/analyses/abc123abc123"):
        r = c.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert c.get("/guide").status_code == 200       # the filming guide stays public
    assert c.get("/login").status_code == 200


def test_signup_logs_in_and_lands_on_upload(app_):
    c = browser(app_)
    r = signup(c)
    assert r.status_code == 303 and r.headers["location"] == "/analyses/new"
    assert "swimform_session" in r.headers.get("set-cookie", "")
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=lax" in r.headers["set-cookie"].lower()
    assert c.get("/", follow_redirects=False).status_code == 200


def test_signup_validation(app_):
    c = browser(app_)
    r = signup(c, email="not-an-email")
    assert r.status_code == 422 and "Enter an email address" in r.text
    r = signup(c, pw="short")
    assert r.status_code == 422 and "at least 10" in r.text
    assert signup(c).status_code == 303
    d = browser(app_)
    r = signup(d, email="A@Example.com")             # same address, different case
    assert r.status_code == 422 and "already an account" in r.text


def test_login_logout(app_):
    c = browser(app_)
    signup(c)
    c.post("/logout", follow_redirects=False)
    assert c.get("/", follow_redirects=False).status_code == 303
    r = login(c, pw="wrong password!")
    assert r.status_code == 401 and "match an account" in r.text
    r = login(c)
    assert r.status_code == 303
    assert c.get("/", follow_redirects=False).status_code == 200


def test_logout_kills_the_session_server_side(app_):
    c = browser(app_)
    signup(c)
    token = c.cookies.get("swimform_session")
    c.post("/logout")
    thief = browser(app_)
    thief.cookies.set("swimform_session", token)
    assert thief.get("/", follow_redirects=False).status_code == 303


def test_login_redirects_back_to_where_you_were_going_but_not_offsite(app_):
    c = browser(app_)
    signup(c)
    c.post("/logout")
    r = c.post("/login", data={"email": "a@example.com", "password": PW, "next": "/guide"}, follow_redirects=False)
    assert r.headers["location"] == "/guide"
    c.post("/logout")
    r = c.post("/login", data={"email": "a@example.com", "password": PW, "next": "//evil.example/x"},
               follow_redirects=False)
    assert r.headers["location"] == "/"


def test_repeated_failures_are_rate_limited(app_):
    c = browser(app_)
    signup(c)
    c.post("/logout")
    for _ in range(5):
        assert login(c, pw="wrong password!").status_code == 401
    r = login(c)                                     # even the right password waits
    assert r.status_code == 429 and "Too many attempts" in r.text


# ------------------------------------------------------------------ privacy
def test_reports_are_private_to_their_owner(app_):
    alice, bob = browser(app_), browser(app_)
    signup(alice, "alice@example.com")
    signup(bob, "bob@example.com")
    loc = analyse(alice).headers["location"]
    assert alice.get(loc).status_code == 200
    r = bob.get(loc)
    assert r.status_code == 404 and "We can't find that report" in r.text
    assert loc.split("/")[-1] not in bob.get("/").text   # not in Bob's list either


def test_reports_home_lists_your_analyses(app_):
    c = browser(app_)
    signup(c, name="Colin Bitz")
    assert "No reports yet" in c.get("/").text
    aid = analyse(c).headers["location"].split("/")[-1]
    home = html.unescape(c.get("/").text)
    assert "Colin's reports" in home and f"/analyses/{aid}" in home
    assert home.count('class="seg seg-') == 14


# ------------------------------------------------------------------ request protections
def test_cross_site_posts_are_refused(app_):
    c = browser(app_)
    signup(c)
    r = c.post("/logout", headers={"origin": "https://evil.example"}, follow_redirects=False)
    assert r.status_code == 403
    assert c.get("/", follow_redirects=False).status_code == 200   # still logged in
    r = c.post("/logout", headers={"origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303
