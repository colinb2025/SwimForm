"""
Passwords, sessions, and request protections.

- Passwords: scrypt (Python standard library), per-user random salt, constant-time check.
- Sessions: a random token in an httpOnly, SameSite=Lax cookie; only its SHA-256 is
  stored, so a database leak does not hand out working sessions. 30-day expiry.
- Cross-site requests: any state-changing request whose Origin (or Referer) names a
  different host is refused, on top of SameSite=Lax.
- Login attempts: after 5 failures for an email in 15 minutes, further attempts wait.
"""
import base64
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import timedelta
from urllib.parse import urlsplit

from sqlalchemy import delete, select

from . import db

COOKIE = "swimform_session"
SESSION_DAYS = 30
SECURE_COOKIES = os.environ.get("SWIMFORM_SECURE_COOKIES", "0") == "1"   # set to 1 behind HTTPS
MIN_PASSWORD = 10

_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(key).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, key = stored.split("$")
        if algo != "scrypt":
            return False
        want = base64.b64decode(key)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=len(want))
        return hmac.compare_digest(got, want)
    except (ValueError, TypeError):
        return False


def password_problem(password: str):
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    return None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db.session() as s:
        s.add(db.LoginSession(token_hash=_token_hash(token), user_id=user_id,
                              expires=db.now() + timedelta(days=SESSION_DAYS)))
        s.commit()
    return token


def end_session(token: str | None):
    if not token:
        return
    with db.session() as s:
        s.execute(delete(db.LoginSession).where(db.LoginSession.token_hash == _token_hash(token)))
        s.commit()


def current_user(request):
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with db.session() as s:
        row = s.get(db.LoginSession, _token_hash(token))
        if row is None:
            return None
        expires = row.expires if row.expires.tzinfo else row.expires.replace(tzinfo=db.now().tzinfo)
        if expires < db.now():
            s.delete(row)
            s.commit()
            return None
        return s.get(db.User, row.user_id)


def set_cookie(response, token: str):
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True,
                        samesite="lax", secure=SECURE_COOKIES, path="/")


def clear_cookie(response):
    response.delete_cookie(COOKIE, path="/")


def same_origin(request) -> bool:
    """False if a state-changing request plainly comes from another site."""
    src = request.headers.get("origin") or request.headers.get("referer")
    if not src or src == "null":
        return True   # no header: not a browser cross-site form post (SameSite still applies)
    return urlsplit(src).netloc == request.headers.get("host", "")


class LoginLimiter:
    def __init__(self, max_failures=5, window_s=900):
        self.max, self.window = max_failures, window_s
        self.fails = defaultdict(deque)

    def _trim(self, key):
        q, cutoff = self.fails[key], time.monotonic() - self.window
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def blocked(self, key) -> bool:
        return len(self._trim(key)) >= self.max

    def fail(self, key):
        self._trim(key).append(time.monotonic())

    def clear(self, key):
        self.fails.pop(key, None)


limiter = LoginLimiter()


def owned_analysis(user, aid):
    """The Analysis row if this user owns it, else None (which the app shows as 'not found')."""
    if user is None:
        return None
    with db.session() as s:
        row = s.get(db.Analysis, aid)
        return row if row is not None and row.user_id == user.id else None


def list_analyses(user):
    with db.session() as s:
        return list(s.scalars(select(db.Analysis).where(db.Analysis.user_id == user.id)
                              .order_by(db.Analysis.created.desc())))
