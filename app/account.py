"""
Account settings: name, email, password, signing out other devices, deleting the account.

Anything that could lock the owner out or erase their data asks for the current password
(when the account has one), so a borrowed, still-logged-in browser can't do it quietly.
"""
from sqlalchemy import delete, select

from . import auth, db, mailer, pipeline

# Separate from the login limiter: wrong current-password guesses on the settings page.
limiter = auth.LoginLimiter(max_failures=5, window_s=900)
CONFIRM_WORD = "delete"


def _password_ok(user, password, key):
    """None if fine, else the message to show. Accounts with no password (Google-only) pass."""
    if not user.password_hash:
        return None
    if limiter.blocked(key):
        return "Too many wrong passwords. Wait 15 minutes, then try again."
    if not auth.verify_password(password, user.password_hash):
        limiter.fail(key)
        return "That isn't your current password."
    limiter.clear(key)
    return None


def update_name(user, name):
    with db.session() as s:
        u = s.get(db.User, user.id)
        u.name = name.strip()[:120]
        s.commit()


def change_email(user, new_email, password):
    """-> error message or None."""
    new = new_email.strip().lower()
    if problem := auth.email_problem(new):
        return problem
    if new == user.email:
        return "That's already your email."
    if problem := _password_ok(user, password, f"acct:{user.id}"):
        return problem
    with db.session() as s:
        if db.user_by_email(s, new):
            return "Another account already uses that email."
        u = s.get(db.User, user.id)
        old, u.email = u.email, new
        s.commit()
    # Tell the old address, so a takeover through a stolen session doesn't go unnoticed.
    mailer.send(old, "Your SwimForm email was changed",
                f"The email on your SwimForm account was changed from {old} to {new}.\n\n"
                "If you didn't do this, reply to this email and we'll lock the account.\n")
    return None


def change_password(user, current, new, keep_token):
    """-> error message or None. Signs out every other device on success."""
    if problem := _password_ok(user, current, f"acct:{user.id}"):
        return problem
    if problem := auth.password_problem(new):
        return problem
    with db.session() as s:
        u = s.get(db.User, user.id)
        u.password_hash = auth.hash_password(new)
        s.execute(delete(db.PasswordReset).where(db.PasswordReset.user_id == u.id))
        s.commit()
    sign_out_others(user, keep_token)
    return None


def sign_out_others(user, keep_token):
    keep = auth._token_hash(keep_token) if keep_token else ""
    with db.session() as s:
        n = s.execute(delete(db.LoginSession).where(db.LoginSession.user_id == user.id,
                                                    db.LoginSession.token_hash != keep)).rowcount
        s.commit()
    return n


def delete_account(user, confirm, password):
    """-> error message or None. Removes clips, reports, sessions and the account itself."""
    if confirm.strip().lower() != CONFIRM_WORD:
        return f'Type "{CONFIRM_WORD}" to confirm.'
    if problem := _password_ok(user, password, f"acct:{user.id}"):
        return problem
    with db.session() as s:
        aids = list(s.scalars(select(db.Analysis.id).where(db.Analysis.user_id == user.id)))
        for table in (db.Analysis, db.LoginSession, db.PasswordReset):
            s.execute(delete(table).where(table.user_id == user.id))   # explicit: SQLite skips ON DELETE
        s.execute(delete(db.User).where(db.User.id == user.id))
        s.commit()
    for aid in aids:
        pipeline.remove(aid)
    return None
