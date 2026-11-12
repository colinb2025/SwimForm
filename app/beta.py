"""
Who can get in during the beta, and who runs it.

SWIMFORM_SIGNUP
  open    (default) anyone can create an account.
  invite  a valid invite code is required; everyone else can join the waitlist.
  closed  no new accounts at all (the waitlist still works).
SWIMFORM_ADMINS  comma-separated emails. These accounts see /admin, can always sign up,
                 and are never limited by plan.
"""
import os
import secrets

from sqlalchemy import select, update

from . import db

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L: codes get read aloud and typed


def mode():
    m = os.environ.get("SWIMFORM_SIGNUP", "open").strip().lower()
    return m if m in ("open", "invite", "closed") else "open"


def admins():
    return {e.strip().lower() for e in os.environ.get("SWIMFORM_ADMINS", "").split(",") if e.strip()}


def is_admin(user):
    return user is not None and user.email in admins()


def normalise(code):
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())[:32]


def new_invite(note="", max_uses=1, code=None):
    code = normalise(code) or "".join(secrets.choice(ALPHABET) for _ in range(8))
    with db.session() as s:
        s.add(db.Invite(code=code, note=note.strip()[:200], max_uses=max(0, int(max_uses))))
        s.commit()
    return code


def gate(email, code):
    """May this email create an account? -> None if yes, else the message to show."""
    if email.strip().lower() in admins() or mode() == "open":
        return None
    if mode() == "closed":
        return "SwimForm isn't taking new accounts right now. Join the waitlist and we'll email you."
    code = normalise(code)
    if not code:
        return "SwimForm is invite-only during the beta. Enter your invite code to create an account."
    with db.session() as s:
        inv = s.get(db.Invite, code)
    if inv is None:
        return "We don't recognise that invite code. Check it and try again."
    if inv.max_uses and inv.uses >= inv.max_uses:
        return "That invite code has already been used."
    return None


def redeem(s, code):
    """Count one use of an invite, inside the caller's transaction. False if it ran out
    between the check and now (two people racing for the last use)."""
    code = normalise(code)
    if not code:
        return True
    inv = s.get(db.Invite, code)
    if inv is None:
        return True
    q = update(db.Invite).where(db.Invite.code == code)
    if inv.max_uses:
        q = q.where(db.Invite.uses < db.Invite.max_uses)
    return s.execute(q.values(uses=db.Invite.uses + 1)).rowcount == 1


def join_waitlist(email, note=""):
    """Add an email to the waitlist. Joining twice is fine and changes nothing."""
    email = email.strip().lower()
    with db.session() as s:
        if s.scalar(select(db.Waitlist).where(db.Waitlist.email == email)) is None:
            s.add(db.Waitlist(email=email, note=note.strip()[:500]))
            s.commit()
