"""
Outgoing email, behind one function so the provider is a config choice.

SWIMFORM_EMAIL selects the backend:
  console  (default) prints the message and keeps it in `outbox`, for development and tests
  smtp     SMTP_HOST, SMTP_PORT (587 = STARTTLS, 465 = TLS), SMTP_USER, SMTP_PASSWORD
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


def render_html(subject: str, text: str) -> str:
    """The plain-text message as a simple branded email. Links on their own line become
    buttons; everything is escaped, so message text can never inject markup."""
    import html
    import re
    blocks = []
    for para in [p.strip() for p in text.strip().split("\n\n") if p.strip()]:
        if re.fullmatch(r"https?://\S+", para):
            label = ("Open your report" if "/analyses/" in para else "Create your account" if "/signup" in para
                     else "Choose a new password" if "/reset" in para else "Review it" if "/admin" in para else "Open SwimForm")
            url = html.escape(para, quote=True)
            blocks.append(f'<p style="margin:24px 0"><a href="{url}" style="background:#000000;color:#ffffff;'
                          f'text-decoration:none;font-weight:600;padding:12px 22px;border-radius:999px;display:inline-block">{label}</a></p>'
                          f'<p style="margin:0 0 16px;font-size:13px;color:#69696E">Or paste this into your browser: {url}</p>')
        else:
            body = html.escape(para).replace("\n", "<br>")
            body = re.sub(r"(https?://[^\s<]+)", r'<a href="\1" style="color:#0A5FDB">\1</a>', body)
            blocks.append(f'<p style="margin:0 0 16px">{body}</p>')
    return ('<!doctype html><html><body style="margin:0;background:#F5F5F7;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#000000">'
            '<div style="max-width:560px;margin:0 auto;padding:24px 16px">'
            '<div style="background:#000000;color:#ffffff;font-weight:700;font-size:18px;padding:16px 24px;border-radius:12px 12px 0 0">SwimForm</div>'
            '<div style="background:#ffffff;padding:28px 24px;border-radius:0 0 12px 12px;font-size:16px;line-height:1.55">'
            f'<h1 style="font-size:20px;margin:0 0 18px">{html.escape(subject)}</h1>{"".join(blocks)}</div>'
            '<p style="font-size:12px;color:#69696E;text-align:center;margin:16px 0 0">You are getting this because you have a SwimForm account or asked for an invite.</p>'
            '</div></body></html>')


def send(to: str, subject: str, text: str):
    backend = os.environ.get("SWIMFORM_EMAIL", "console")
    if backend == "console":
        outbox.append({"to": to, "subject": subject, "text": text, "html": render_html(subject, text)})
        log.warning("EMAIL (console backend) to=%s subject=%s\n%s", to, subject, text)
        print(f"\n--- email to {to}: {subject} ---\n{text}\n---\n", flush=True)
        return
    if backend == "smtp":
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = _sender(), to, subject
        msg.set_content(text)
        msg.add_alternative(render_html(subject, text), subtype="html")
        port = int(os.environ.get("SMTP_PORT", "587"))
        conn = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP   # 465 = TLS from the start
        with conn(os.environ["SMTP_HOST"], port, timeout=15) as s:
            if port != 465:
                s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return
    if backend == "ses":
        import boto3   # optional dependency, only needed for this backend
        boto3.client("ses").send_email(
            Source=_sender(), Destination={"ToAddresses": [to]},
            Message={"Subject": {"Data": subject},
                     "Body": {"Text": {"Data": text}, "Html": {"Data": render_html(subject, text)}}})
        return
    raise RuntimeError(f"unknown SWIMFORM_EMAIL backend {backend!r}")
