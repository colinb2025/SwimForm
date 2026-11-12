"""
What's plugged in. One list, shown on /admin and printed by `python -m app.check`,
so "which keys are still missing?" always has one answer.

Each item: name, state ("on" working as configured, "off" not set up and the app runs
without it, "todo" needed before real users, "problem" half-configured), detail, fix.
"""
import os
from pathlib import Path

from . import beta, billing, google, jobs, storage


def _env(*names):
    return [n for n in names if not os.environ.get(n)]


def check():
    items = []

    def add(name, state, detail, fix=""):
        items.append({"name": name, "state": state, "detail": detail, "fix": fix})

    base = os.environ.get("SWIMFORM_BASE_URL", "")
    secure = os.environ.get("SWIMFORM_SECURE_COOKIES", "0") == "1"
    if not base:
        add("Public address", "todo", "SWIMFORM_BASE_URL is not set, so emailed links point at localhost.",
            "Set SWIMFORM_BASE_URL to the site's https address.")
    elif base.startswith("https://") and not secure:
        add("Public address", "problem", f"{base} is https but SWIMFORM_SECURE_COOKIES is 0.", "Set SWIMFORM_SECURE_COOKIES=1.")
    elif base.startswith("https://"):
        add("Public address", "on", base)
    else:
        add("Public address", "todo", f"{base} (not https: fine for local development only).",
            "Use the https address and SWIMFORM_SECURE_COOKIES=1 in production.")

    db_url = os.environ.get("SWIMFORM_DB", "")
    add("Database", "on", "Postgres" if db_url.startswith("postgres") else "SQLite file (fine for a single server)")

    mail = os.environ.get("SWIMFORM_EMAIL", "console")
    sender = os.environ.get("SWIMFORM_EMAIL_FROM", "")
    if mail == "console":
        add("Email", "todo", "Emails are only printed to the server log: password resets and invites won't arrive.",
            "Set SWIMFORM_EMAIL=smtp (with SMTP_HOST, SMTP_USER, SMTP_PASSWORD) or ses, and SWIMFORM_EMAIL_FROM.")
    elif mail == "smtp" and _env("SMTP_HOST"):
        add("Email", "problem", "SWIMFORM_EMAIL=smtp but SMTP_HOST is empty.", "Fill in SMTP_HOST, SMTP_USER and SMTP_PASSWORD.")
    elif not sender or "example.com" in sender or "localhost" in sender:
        add("Email", "problem", f"Sending by {mail}, but SWIMFORM_EMAIL_FROM is still a placeholder.",
            "Set SWIMFORM_EMAIL_FROM to an address on your verified domain.")
    else:
        add("Email", "on", f"{mail}, from {sender}")

    g = _env("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET")
    if google.configured():
        add("Google sign-in", "on", "The 'Continue with Google' button is shown.")
    elif len(g) == 1:
        add("Google sign-in", "problem", f"{g[0]} is missing.", "Set both GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
    else:
        add("Google sign-in", "off", "Not set up: people sign in with email and password.",
            "Optional. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (docs/SETUP.md).")

    pose = os.environ.get("SWIMFORM_POSE", "synthetic")
    weights = os.environ.get("SWIMFORM_WEIGHTS", "")
    if pose == "yolo" and weights and Path(weights).exists():
        add("Pose model", "on", f"Analysing real video with {Path(weights).name}.")
    elif pose == "yolo":
        add("Pose model", "problem", f"SWIMFORM_POSE=yolo but the weights file {weights or '(not set)'} was not found.",
            "Point SWIMFORM_WEIGHTS at the trained .pt file.")
    elif pose == "collect":
        add("Pose model", "on", "Collecting clips only: uploads are stored and wait for the model.",
            "When the model is trained, set SWIMFORM_POSE=yolo and SWIMFORM_WEIGHTS, then run the waiting analyses from /admin.")
    else:
        add("Pose model", "todo", "Demo mode: reports come from a simulated swimmer, not from the uploaded video.",
            "Set SWIMFORM_POSE=collect to gather beta clips, or yolo once the model is trained.")

    if jobs.review_on():
        add("Coach review", "on", "Finished reports are held until an admin releases them.")
    else:
        add("Coach review", "off", "Reports go straight to the swimmer.", "Optional. Set SWIMFORM_REVIEW=1 to check each report first.")

    s = _env("STRIPE_SECRET_KEY", "STRIPE_PRICE_ID", "STRIPE_WEBHOOK_SECRET")
    limit = billing.free_limit()
    limit_txt = f"Free accounts: {limit} analyses a month." if limit is not None else "No usage limit."
    if not s:
        test = os.environ["STRIPE_SECRET_KEY"].startswith("sk_test")
        add("Payments", "on", f"Stripe{' (test mode)' if test else ''}. {limit_txt}")
    elif len(s) == 3:
        add("Payments", "off", f"Not set up: everything is free. {limit_txt}",
            "Optional. Set STRIPE_SECRET_KEY, STRIPE_PRICE_ID and STRIPE_WEBHOOK_SECRET.")
    else:
        add("Payments", "problem", "Stripe is half set up: missing " + ", ".join(s) + ".", "Set all three Stripe values.")

    if storage.backend() == "s3":
        missing = _env("S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if missing:
            add("Clip storage", "problem", "SWIMFORM_STORAGE=s3 but missing " + ", ".join(missing) + ".", "Fill in the bucket settings.")
        else:
            add("Clip storage", "on", f"Bucket {os.environ['S3_BUCKET']}.")
    else:
        add("Clip storage", "off", "Clips and reports are kept on this server's disk.",
            "Fine with a persistent disk. On hosts that wipe the disk on deploy, set SWIMFORM_STORAGE=s3.")

    who = sorted(beta.admins())
    if who:
        add("Admins", "on", ", ".join(who))
    else:
        add("Admins", "todo", "Nobody can open /admin.", "Set SWIMFORM_ADMINS to your email.")

    add("Sign-up", "on", {"open": "Open: anyone can create an account.", "invite": "Invite only, with a waitlist.",
                          "closed": "Closed: waitlist only."}[beta.mode()])

    contact = os.environ.get("SWIMFORM_CONTACT_EMAIL", "")
    if contact:
        add("Terms and privacy", "on", f"Published at /terms and /privacy, contact {contact}. Have them reviewed before charging money.")
    else:
        add("Terms and privacy", "todo", "The pages are published but list no way to contact you.",
            "Set SWIMFORM_CONTACT_EMAIL (and SWIMFORM_COMPANY if you trade under another name).")

    if os.environ.get("SENTRY_DSN"):
        add("Error reports", "on", "Crashes are sent to Sentry.")
    else:
        add("Error reports", "off", "Crashes are only written to the server log.", "Optional. Set SENTRY_DSN.")
    return items


def ready(items=None):
    """Nothing half-configured and nothing still to do before real users?"""
    return all(i["state"] in ("on", "off") for i in (items or check()))
