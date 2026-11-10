"""
Plans, usage limits and Stripe.

Nothing here is switched on until its settings exist, so the beta can start free and
unlimited and gain limits and payments later without code changes.

SWIMFORM_FREE_ANALYSES  how many analyses a free account can run per calendar month.
                        Unset = no limit. Clips that were rejected or that failed on our
                        side don't count.
STRIPE_SECRET_KEY       sk_test_... / sk_live_...
STRIPE_PRICE_ID         price_... of the recurring price that makes an account "pro".
STRIPE_WEBHOOK_SECRET   whsec_... for the endpoint <base url>/webhooks/stripe
SWIMFORM_PRICE_LABEL    what the upgrade button says the price is, e.g. "$12 a month".

One paid plan ("pro" = unlimited) is a placeholder until the real tiers are decided:
the plan lives in users.plan, so more tiers are more values there.

Stripe is called over its REST API with httpx: no SDK to keep in step.
"""
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone

from sqlalchemy import func, select

from . import beta, db

log = logging.getLogger("swimform.billing")
API = "https://api.stripe.com/v1"
NOT_COUNTED = ("rejected", "failed")
PAID_STATES = ("active", "trialing", "past_due")   # past_due: Stripe is still retrying the card


class BillingError(Exception):
    pass


def configured():
    return bool(os.environ.get("STRIPE_SECRET_KEY") and os.environ.get("STRIPE_PRICE_ID"))


def free_limit():
    raw = os.environ.get("SWIMFORM_FREE_ANALYSES", "").strip()
    return int(raw) if raw.isdigit() else None


def visible():
    """Show plan and usage to users at all?"""
    return configured() or free_limit() is not None


def price_label():
    return os.environ.get("SWIMFORM_PRICE_LABEL", "").strip()


def usage(user):
    """Analyses this calendar month (UTC) that count toward the free limit."""
    start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    with db.session() as s:
        return s.scalar(select(func.count()).select_from(db.Analysis).where(
            db.Analysis.user_id == user.id, db.Analysis.created >= start,
            db.Analysis.status.notin_(NOT_COUNTED))) or 0


def status(user):
    """Everything the UI needs about this account's plan."""
    limit = free_limit()
    unlimited = user.plan == "pro" or beta.is_admin(user) or limit is None
    used = usage(user)
    return {"plan": user.plan, "pro": user.plan == "pro", "limit": None if unlimited else limit, "free_limit": limit,
            "used": used, "left": None if unlimited else max(0, limit - used),
            "can_analyse": unlimited or used < limit,
            "can_upgrade": configured() and user.plan != "pro",
            "can_manage": configured() and bool(user.stripe_customer),
            "price": price_label()}


# ------------------------------------------------------------------ Stripe
def _stripe(method, path, data=None):
    """One Stripe API call. Separate so tests can stand in for Stripe."""
    import httpx
    r = httpx.request(method, API + path, data=data, auth=(os.environ["STRIPE_SECRET_KEY"], ""), timeout=20)
    if r.status_code >= 400:
        log.error("stripe %s %s -> %s %s", method, path, r.status_code, r.text[:300])
        raise BillingError("The payment service didn't respond as expected. Please try again in a minute.")
    return r.json()


def checkout_url(user, base):
    data = {"mode": "subscription", "line_items[0][price]": os.environ["STRIPE_PRICE_ID"],
            "line_items[0][quantity]": "1", "client_reference_id": str(user.id),
            "success_url": f"{base}/billing?done=1", "cancel_url": f"{base}/billing",
            "allow_promotion_codes": "true"}
    if user.stripe_customer:
        data["customer"] = user.stripe_customer
    else:
        data["customer_email"] = user.email
    return _stripe("POST", "/checkout/sessions", data)["url"]


def portal_url(user, base):
    return _stripe("POST", "/billing_portal/sessions",
                   {"customer": user.stripe_customer, "return_url": f"{base}/billing"})["url"]


def cancel(user):
    """Stop billing a deleted account. Best effort: never blocks the deletion."""
    if not (configured() and user.stripe_subscription):
        return
    try:
        _stripe("DELETE", f"/subscriptions/{user.stripe_subscription}")
    except Exception:   # noqa: BLE001
        log.exception("could not cancel subscription %s for deleted user %s", user.stripe_subscription, user.id)


def verify(payload: bytes, header: str, secret=None, tolerance=300, now=None):
    """Check Stripe's signature on a webhook and return the event. Raises BillingError."""
    secret = secret or os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        raise BillingError("webhook secret not configured")
    parts = dict(p.split("=", 1) for p in (header or "").split(",") if "=" in p)
    sigs = [v for k, v in (p.split("=", 1) for p in (header or "").split(",") if "=" in p) if k == "v1"]
    try:
        ts = int(parts.get("t", ""))
    except ValueError:
        raise BillingError("bad signature header")
    want = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(want, s) for s in sigs):
        raise BillingError("signature mismatch")
    if abs((now or time.time()) - ts) > tolerance:
        raise BillingError("signature too old")
    try:
        return json.loads(payload)
    except ValueError:
        raise BillingError("bad payload")


def handle(event):
    """Apply a Stripe event to the account it concerns. Unknown events are ignored."""
    kind, obj = event.get("type", ""), event.get("data", {}).get("object", {})
    with db.session() as s:
        if kind == "checkout.session.completed":
            ref = obj.get("client_reference_id") or ""
            user = s.get(db.User, int(ref)) if ref.isdigit() else None
            if user is None:
                return "no such user"
            user.stripe_customer = obj.get("customer") or user.stripe_customer
            user.stripe_subscription = obj.get("subscription") or user.stripe_subscription
            user.plan = "pro"
        elif kind in ("customer.subscription.updated", "customer.subscription.deleted"):
            user = s.scalar(select(db.User).where(db.User.stripe_customer == obj.get("customer")))
            if user is None:
                return "no such customer"
            paid = kind.endswith("updated") and obj.get("status") in PAID_STATES
            user.plan = "pro" if paid else "free"
            user.stripe_subscription = obj.get("id") if paid else None
        else:
            return "ignored"
        s.commit()
    return "ok"
