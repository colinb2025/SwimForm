"""
Outgoing email, behind one function so the provider is a config choice.

SWIMFORM_EMAIL selects the backend:
  console  (default) prints the message and keeps it in `outbox`, for development and tests
  smtp     SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, STARTTLS
  ses      Amazon SES via boto3, region from AWS_REGION (credentials from the usual AWS chain)
SWIMFORM_EMAIL_FROM is the sender, e.g. "SwimForm <no-reply@yourdomain.com>".
"""
import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("swimform.mail")
outbox = []   # console backend keeps sent messages here


def _sender():
    return os.environ.get("SWIMFORM_EMAIL_FROM", "SwimForm <no-reply@localhost>")


def send(to: str, subject: str, text: str):
    backend = os.environ.get("SWIMFORM_EMAIL", "console")
    if backend == "console":
        outbox.append({"to": to, "subject": subject, "text": text})
        log.warning("EMAIL (console backend) to=%s subject=%s\n%s", to, subject, text)
        print(f"\n--- email to {to}: {subject} ---\n{text}\n---\n", flush=True)
        return
    if backend == "smtp":
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = _sender(), to, subject
        msg.set_content(text)
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return
    if backend == "ses":
        import boto3   # optional dependency, only needed for this backend
        boto3.client("ses").send_email(
            Source=_sender(), Destination={"ToAddresses": [to]},
            Message={"Subject": {"Data": subject}, "Body": {"Text": {"Data": text}}})
        return
    raise RuntimeError(f"unknown SWIMFORM_EMAIL backend {backend!r}")
