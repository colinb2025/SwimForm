"""
Sign in with Google (OpenID Connect, authorization code flow).

Security, in order of the flow:
  - state:  random, stored server-side, single use; ties the callback to the browser
            that started sign-in (stops login CSRF).
  - PKCE:   the code is useless without the verifier only this server holds.
  - nonce:  must come back inside the ID token (stops token replay).
  - ID token checks: issuer is Google, audience is our client ID, not expired, nonce
    matches, email verified. The token comes straight from Google's token endpoint over
    TLS, which OpenID Connect accepts in place of verifying its signature (Core 3.1.3.7).

Configure with GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (Google Cloud console, OAuth
client of type "Web application", redirect URI <base>/auth/google/callback). The
button only appears when both are set. SWIMFORM_BASE_URL fixes the base URL behind a
proxy; otherwise it's taken from the request.
"""
import base64
import hashlib
import json
import os
import secrets
import time
from datetime import timedelta
from urllib.parse import urlencode

from sqlalchemy import delete, select

from . import db

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
STATE_MINUTES = 10


class GoogleSignInError(Exception):
    """Sign-in could not be completed; the message is safe to show."""


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"))


def redirect_uri(request) -> str:
    base = os.environ.get("SWIMFORM_BASE_URL") or str(request.base_url).rstrip("/")
    return base.rstrip("/") + "/auth/google/callback"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def start(request, next_url: str = "/") -> str:
    """Record a pending sign-in and return the Google URL to send the browser to."""
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    with db.session() as s:
        cutoff = db.now() - timedelta(minutes=STATE_MINUTES)
        s.execute(delete(db.OAuthState).where(db.OAuthState.created < cutoff))   # tidy abandoned attempts
        s.add(db.OAuthState(state=state, nonce=nonce, verifier=verifier, next_url=next_url))
        s.commit()
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return AUTH_URL + "?" + urlencode({
        "client_id": os.environ["GOOGLE_CLIENT_ID"], "redirect_uri": redirect_uri(request),
        "response_type": "code", "scope": "openid email profile",
        "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256",
        "prompt": "select_account",
    })


def exchange_code(code: str, verifier: str, redirect: str) -> dict:
    """Swap the authorization code for tokens at Google. Separate so tests can stand in for Google."""
    import httpx
    r = httpx.post(TOKEN_URL, data={
        "code": code, "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"], "redirect_uri": redirect,
        "grant_type": "authorization_code", "code_verifier": verifier,
    }, timeout=15)
    if r.status_code != 200:
        raise GoogleSignInError("Google didn't accept the sign-in. Please try again.")
    return r.json()


def _claims(id_token: str) -> dict:
    try:
        payload = id_token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        raise GoogleSignInError("Google sent back something we couldn't read. Please try again.")


def finish(request, state: str, code: str):
    """Validate the callback and return (user, next_url). Creates or links the account."""
    if not state or not code:
        raise GoogleSignInError("Sign-in was cancelled or didn't complete.")
    with db.session() as s:
        pending = s.get(db.OAuthState, state)
        if pending is not None:
            s.delete(pending)        # single use, whatever happens next
            s.commit()
    if pending is None:
        raise GoogleSignInError("That sign-in link has expired. Please try again.")
    created = pending.created if pending.created.tzinfo else pending.created.replace(tzinfo=db.now().tzinfo)
    if created < db.now() - timedelta(minutes=STATE_MINUTES):
        raise GoogleSignInError("That sign-in link has expired. Please try again.")

    tokens = exchange_code(code, pending.verifier, redirect_uri(request))
    c = _claims(tokens.get("id_token", ""))
    if c.get("iss") not in ISSUERS or c.get("aud") != os.environ["GOOGLE_CLIENT_ID"]:
        raise GoogleSignInError("Google sign-in couldn't be verified. Please try again.")
    if float(c.get("exp", 0)) < time.time():
        raise GoogleSignInError("Google sign-in expired. Please try again.")
    if c.get("nonce") != pending.nonce:
        raise GoogleSignInError("Google sign-in couldn't be verified. Please try again.")
    if not c.get("email") or c.get("email_verified") not in (True, "true"):
        raise GoogleSignInError("Your Google account's email isn't verified, so we can't use it to sign in.")

    sub, email = str(c["sub"]), c["email"].strip().lower()
    with db.session() as s:
        user = s.scalar(select(db.User).where(db.User.google_sub == sub))
        if user is None:
            user = db.user_by_email(s, email)
            if user is not None:
                # Same verified email as an existing account: link Google to it.
                user.google_sub = sub
            else:
                user = db.User(email=email, name=(c.get("name") or "")[:120], password_hash="", google_sub=sub)
                s.add(user)
        s.commit()
        return user, pending.next_url
