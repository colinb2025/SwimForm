"""Emails the app sends about analyses and invites. Sending never breaks the caller."""
import logging

from . import beta, db, mailer

log = logging.getLogger("swimform.notify")


def _base():
    from .web import base_url
    return base_url()


def _send(to, subject, text):
    try:
        mailer.send(to, subject, text)
    except Exception:   # noqa: BLE001 - a mail outage must not fail an analysis
        log.exception("could not email %s: %s", to, subject)


def _owner(aid):
    with db.session() as s:
        row = s.get(db.Analysis, aid)
        return (row, s.get(db.User, row.user_id)) if row else (None, None)


def _hi(user):
    return f"Hi {user.name.split()[0]}," if user.name else "Hi,"


def report_ready(aid):
    row, user = _owner(aid)
    if user is None:
        return
    note = "\n\nYour coach left you a note at the top of the report." if row.coach_note else ""
    _send(user.email, "Your SwimForm stroke report is ready",
          f"{_hi(user)}\n\nYour stroke report is ready:\n\n{_base()}/analyses/{aid}{note}\n\n"
          "At the bottom of the report you can tell us which findings matched what you or your "
          "coach see. During the beta that feedback is how the analysis gets better.\n")


def clips_rejected(aid):
    row, user = _owner(aid)
    if user is None:
        return
    _send(user.email, "We couldn't use your SwimForm clips",
          f"{_hi(user)}\n\nWe checked the clips you uploaded and couldn't analyse them as filmed. "
          f"This page says what to change:\n\n{_base()}/analyses/{aid}\n")


def review_needed(aid):
    for admin in sorted(beta.admins()):
        _send(admin, "SwimForm: a report is waiting for review", f"Review and release it here:\n\n{_base()}/admin/analyses/{aid}\n")


def invite(email, code):
    _send(email, "Your SwimForm beta invite",
          "Hi,\n\nYou're in. Create your SwimForm account with this link:\n\n"
          f"{_base()}/signup?invite={code}\n\nYour invite code is {code}.\n\n"
          "You'll need two short clips of your freestyle: one from the side and one head-on. "
          f"The filming guide shows how: {_base()}/guide\n")
