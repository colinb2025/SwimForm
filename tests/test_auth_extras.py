"""Password reset and Google sign-in."""
import base64
import json
import re
import time
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

PW = "correct horse 1"
CLIENT_ID = "test-client.apps.googleusercontent.com"


@pytest.fixture()
def app_(tmp_path, monkeypatch):
    from app import auth, db, mailer, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setenv("SWIMFORM_EMAIL", "console")
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    auth.limiter.fails.clear()
    auth.reset_limiter.fails.clear()
    mailer.outbox.clear()
    from app.main import app
    return app


def signup(c, email="a@example.com", name="Alex"):
    return c.post("/signup", data={"name": name, "email": email, "password": PW}, follow_redirects=False)


def logged_in(c):
    return c.get("/", follow_redirects=False).status_code == 200


def reset_link():
    from app import mailer
    return re.search(r"(/reset\?token=\S+)", mailer.outbox[-1]["text"]).group(1)


# ================================================================ password reset
def test_reset_email_is_sent_and_the_answer_never_reveals_accounts(app_):
    from app import mailer
    c = TestClient(app_)
    signup(c)
    c.post("/logout")
    known = c.post("/forgot", data={"email": "a@example.com"})
    unknown = c.post("/forgot", data={"email": "nobody@example.com"})
    assert known.status_code == unknown.status_code == 200
    assert "Check your email" in known.text and "Check your email" in unknown.text
    assert len(mailer.outbox) == 1 and mailer.outbox[0]["to"] == "a@example.com"
    assert "within the next hour" in mailer.outbox[0]["text"] and "Hi Alex" in mailer.outbox[0]["text"]


def test_reset_sets_new_password_signs_out_everywhere_and_is_single_use(app_):
    phone, laptop = TestClient(app_), TestClient(app_)
    signup(phone)
    laptop.post("/login", data={"email": "a@example.com", "password": PW})
    assert logged_in(laptop)
    phone.post("/logout")
    phone.post("/forgot", data={"email": "a@example.com"})
    link = reset_link()
    token = parse_qs(urlsplit(link).query)["token"][0]

    assert phone.get(link).status_code == 200
    r = phone.post("/reset", data={"token": token, "password": "short"})
    assert r.status_code == 422 and "at least 10" in r.text
    r = phone.post("/reset", data={"token": token, "password": "a brand new password"}, follow_redirects=False)
    assert r.status_code == 303 and logged_in(phone)
    assert not logged_in(laptop)                                    # signed out everywhere else

    assert phone.post("/reset", data={"token": token, "password": "another password 2"}).status_code == 400
    phone.post("/logout")
    assert phone.post("/login", data={"email": "a@example.com", "password": PW}).status_code == 401
    assert phone.post("/login", data={"email": "a@example.com", "password": "a brand new password"},
                      follow_redirects=False).status_code == 303


def test_expired_or_bogus_reset_links_are_refused(app_):
    from app import db
    c = TestClient(app_)
    signup(c)
    c.post("/forgot", data={"email": "a@example.com"})
    with db.session() as s:
        for row in s.query(db.PasswordReset).all():
            row.expires = db.now() - timedelta(minutes=1)
        s.commit()
    r = c.get(reset_link())
    assert r.status_code == 400 and "expired" in r.text
    assert c.get("/reset?token=made-up").status_code == 400


def test_reset_emails_are_rate_limited(app_):
    from app import mailer
    c = TestClient(app_)
    signup(c)
    for _ in range(5):
        assert c.post("/forgot", data={"email": "a@example.com"}).status_code == 200
    assert len(mailer.outbox) == 3


def test_reset_tokens_are_stored_only_as_hashes(app_):
    from app import db
    c = TestClient(app_)
    signup(c)
    c.post("/forgot", data={"email": "a@example.com"})
    token = parse_qs(urlsplit(reset_link()).query)["token"][0]
    with db.session() as s:
        assert all(token not in r.token_hash for r in s.query(db.PasswordReset).all())


# ================================================================ Google sign-in
@pytest.fixture()
def google_on(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")


def fake_google(monkeypatch, **claims):
    """Stand in for Google's token endpoint; returns an ID token with these claims."""
    from app import google
    seen = {}

    def exchange(code, verifier, redirect):
        seen.update(code=code, verifier=verifier, redirect=redirect)
        nonce = seen.get("nonce")
        body = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "exp": time.time() + 600,
                "sub": "google-123", "email": "g@example.com", "email_verified": True,
                "name": "Gail Swimmer", "nonce": nonce}
        body.update(claims)
        payload = base64.urlsafe_b64encode(json.dumps(body).encode()).rstrip(b"=").decode()
        return {"id_token": f"header.{payload}.sig"}
    monkeypatch.setattr(google, "exchange_code", exchange)
    return seen


def begin(c, seen):
    r = c.get("/auth/google/start", follow_redirects=False)
    assert r.status_code == 303
    url = urlsplit(r.headers["location"])
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert url.netloc == "accounts.google.com"
    assert q["client_id"] == CLIENT_ID and q["code_challenge_method"] == "S256" and q["scope"] == "openid email profile"
    seen["nonce"] = q["nonce"]
    return q["state"]


def test_button_and_routes_only_exist_when_configured(app_, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    c = TestClient(app_)
    assert "Continue with Google" not in c.get("/login").text
    assert c.get("/auth/google/start").status_code == 404
    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    assert "Continue with Google" in c.get("/login").text and "Continue with Google" in c.get("/signup").text


def test_google_sign_in_creates_an_account(app_, google_on, monkeypatch):
    from app import db
    seen = fake_google(monkeypatch)
    c = TestClient(app_)
    state = begin(c, seen)
    r = c.get(f"/auth/google/callback?state={state}&code=abc", follow_redirects=False)
    assert r.status_code == 303 and logged_in(c)
    assert seen["code"] == "abc" and len(seen["verifier"]) >= 43 and seen["redirect"].endswith("/auth/google/callback")
    with db.session() as s:
        u = db.user_by_email(s, "g@example.com")
        assert u.google_sub == "google-123" and u.password_hash == "" and u.name == "Gail Swimmer"
    c.post("/logout")
    assert c.post("/login", data={"email": "g@example.com", "password": ""}).status_code == 401


def test_google_links_to_an_existing_account_with_the_same_email(app_, google_on, monkeypatch):
    from app import db
    c = TestClient(app_)
    signup(c, email="g@example.com")
    c.post("/logout")
    seen = fake_google(monkeypatch)
    state = begin(c, seen)
    c.get(f"/auth/google/callback?state={state}&code=abc")
    assert logged_in(c)
    with db.session() as s:
        assert s.query(db.User).count() == 1
        assert db.user_by_email(s, "g@example.com").google_sub == "google-123"


@pytest.mark.parametrize("claims, message", [
    ({"nonce": "wrong"}, "couldn't be verified"),
    ({"aud": "someone-else"}, "couldn't be verified"),
    ({"iss": "https://evil.example"}, "couldn't be verified"),
    ({"exp": time.time() - 10}, "expired"),
    ({"email_verified": False}, "isn't verified"),
])
def test_bad_id_tokens_are_refused(app_, google_on, monkeypatch, claims, message):
    seen = fake_google(monkeypatch, **claims)
    c = TestClient(app_)
    state = begin(c, seen)
    r = c.get(f"/auth/google/callback?state={state}&code=abc")
    assert r.status_code == 400 and message in r.text.replace("&#39;", "'")
    assert not logged_in(c)


def test_unknown_or_reused_state_is_refused(app_, google_on, monkeypatch):
    seen = fake_google(monkeypatch)
    c = TestClient(app_)
    assert c.get("/auth/google/callback?state=forged&code=abc").status_code == 400
    state = begin(c, seen)
    assert c.get(f"/auth/google/callback?state={state}&code=abc", follow_redirects=False).status_code == 303
    c.post("/logout")
    r = c.get(f"/auth/google/callback?state={state}&code=abc")
    assert r.status_code == 400 and "expired" in r.text and not logged_in(c)


def test_cancelled_google_sign_in(app_, google_on):
    r = TestClient(app_).get("/auth/google/callback?error=access_denied")
    assert r.status_code == 400 and "cancelled" in r.text


@pytest.mark.sqlite_only
def test_old_databases_gain_the_new_columns(tmp_path):
    import sqlite3
    from app import db
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(320), name VARCHAR(120), "
                "password_hash VARCHAR(255), created DATETIME)")
    con.execute("CREATE TABLE analyses (id VARCHAR(32) PRIMARY KEY, user_id INTEGER, created DATETIME, "
                "lanes TEXT, flagged INTEGER, estimated INTEGER, unchecked INTEGER)")
    con.execute("INSERT INTO analyses VALUES ('abc123', 1, '2026-10-01', 'clear', 0, 0, 0)")
    con.commit(); con.close()
    db.init(f"sqlite:///{path}")
    con = sqlite3.connect(path)
    assert "google_sub" in [r[1] for r in con.execute("PRAGMA table_info(users)")]
    assert con.execute("SELECT status, problems FROM analyses").fetchone() == ("done", "")   # old rows: finished
