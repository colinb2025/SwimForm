"""
Everything a beta needs around the analysis: who gets in, what they agree to, feedback,
the review hold, collecting clips before the model exists, plans and Stripe, cloud
storage, and the admin console. Outside services (Stripe, the bucket, Google) are
stood in for; nothing here touches the network.
"""
import hashlib
import hmac
import io
import json
import re
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

PW = "correct horse 1"
ADMIN = "coach@example.com"


@pytest.fixture()
def app_(tmp_path, monkeypatch):
    from app import auth, db, mailer, pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    for var in ("SWIMFORM_SIGNUP", "SWIMFORM_REVIEW", "SWIMFORM_POSE", "SWIMFORM_FREE_ANALYSES", "SWIMFORM_STORAGE",
                "STRIPE_SECRET_KEY", "STRIPE_PRICE_ID", "STRIPE_WEBHOOK_SECRET", "SWIMFORM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SWIMFORM_EMAIL", "console")
    monkeypatch.setenv("SWIMFORM_ADMINS", ADMIN)
    db.init(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    for lim in (auth.limiter, auth.reset_limiter, auth.waitlist_limiter):
        lim.fails.clear()
    mailer.outbox.clear()
    from app.main import app
    return app


def signup(c, email="swimmer@example.com", name="Sam", **extra):
    return c.post("/signup", data={"name": name, "email": email, "password": PW, **extra}, follow_redirects=False)


def browser(app_, email="swimmer@example.com", **extra):
    c = TestClient(app_)
    assert signup(c, email, **extra).status_code == 303
    return c


def upload(c, side="lap1.mp4", head="lap2.mov", **data):
    files = {"side": (side, io.BytesIO(b"side clip"), "video/mp4"),
             "head_on": (head, io.BytesIO(b"head clip"), "video/quicktime")}
    return c.post("/analyses", files=files, data=data, follow_redirects=False)


def aid_of(r):
    return r.headers["location"].split("/")[-1]


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).replace("&#39;", "'")


def status_of(aid):
    from app import db
    with db.session() as s:
        return s.get(db.Analysis, aid).status


def outbox():
    from app import mailer
    return mailer.outbox


# ================================================================ who gets in
def test_open_signup_needs_no_code_and_records_terms(app_):
    from app import db
    browser(app_)
    with db.session() as s:
        u = db.user_by_email(s, "swimmer@example.com")
        assert u.terms_accepted is not None and u.plan == "free" and u.invite_code is None


def test_invite_mode_refuses_without_a_valid_code(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    c = TestClient(app_)
    assert "Invite code" in c.get("/signup").text
    r = signup(c)
    assert r.status_code == 422 and "invite-only" in text(r.text)
    r = signup(c, invite="NOPE1234")
    assert r.status_code == 422 and "don't recognise" in text(r.text)
    assert c.get("/account", follow_redirects=False).status_code == 303   # still logged out


def test_invite_code_works_once(app_, monkeypatch):
    from app import beta, db
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    code = beta.new_invite("tri club")
    c = TestClient(app_)
    assert f'value="{code}"' in c.get(f"/signup?invite={code.lower()}").text      # link pre-fills, any case
    assert signup(c, invite=f" {code.lower()} ").status_code == 303
    with db.session() as s:
        assert s.get(db.Invite, code).uses == 1
        assert db.user_by_email(s, "swimmer@example.com").invite_code == code
    r = signup(TestClient(app_), email="second@example.com", invite=code)
    assert r.status_code == 422 and "already been used" in text(r.text)


def test_multi_use_and_unlimited_codes(app_, monkeypatch):
    from app import beta
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    two, any_ = beta.new_invite(max_uses=2), beta.new_invite(max_uses=0, code="tri-club")
    assert any_ == "TRICLUB"
    assert [signup(TestClient(app_), email=f"{i}@example.com", invite=two).status_code for i in range(3)] == [303, 303, 422]
    assert all(signup(TestClient(app_), email=f"u{i}@example.com", invite=any_).status_code == 303 for i in range(4))


def test_a_failed_signup_does_not_burn_the_invite(app_, monkeypatch):
    from app import beta, db
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    code = beta.new_invite()
    assert signup(TestClient(app_), invite=code, email="not-an-email").status_code == 422
    with db.session() as s:
        assert s.get(db.Invite, code).uses == 0
    assert signup(TestClient(app_), invite=code).status_code == 303


def test_admins_can_always_sign_up(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_SIGNUP", "closed")
    assert signup(TestClient(app_)).status_code == 422
    assert "Join the waitlist" in TestClient(app_).get("/signup").text
    assert signup(TestClient(app_), email=ADMIN).status_code == 303


def test_waitlist_collects_emails_without_revealing_or_duplicating(app_):
    from app import db
    c = TestClient(app_)
    assert c.post("/waitlist", data={"email": "nope"}).status_code == 422
    for _ in range(2):
        r = c.post("/waitlist", data={"email": "Wait@Example.com", "note": "first 70.3"})
        assert r.status_code == 200 and "on the list" in text(r.text)
    with db.session() as s:
        rows = s.query(db.Waitlist).all()
        assert [(w.email, w.note) for w in rows] == [("wait@example.com", "first 70.3")]


def test_google_signups_go_through_the_same_door(app_, monkeypatch):
    from app import beta
    from tests.test_auth_extras import CLIENT_ID, fake_google
    from urllib.parse import parse_qs, urlsplit
    monkeypatch.setenv("GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")

    def go(c, invite=""):
        seen = fake_google(monkeypatch)
        q = parse_qs(urlsplit(c.get(f"/auth/google/start?invite={invite}", follow_redirects=False).headers["location"]).query)
        seen["nonce"] = q["nonce"][0]
        return c.get(f"/auth/google/callback?state={q['state'][0]}&code=abc", follow_redirects=False)

    r = go(TestClient(app_))
    assert r.status_code == 400 and "invite-only" in text(r.text)
    assert go(TestClient(app_), beta.new_invite()).status_code == 303


def test_terms_and_privacy_are_public_and_linked(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_CONTACT_EMAIL", "hello@swimform.test")
    c = TestClient(app_)
    terms, privacy = c.get("/terms"), c.get("/privacy")
    assert terms.status_code == privacy.status_code == 200
    assert "18 or older" in text(terms.text) and "not medical advice" in text(terms.text)
    assert "only with your permission" in text(privacy.text) and "hello@swimform.test" in privacy.text
    for path in ("/login", "/signup"):
        assert 'href="/terms"' in c.get(path).text and 'href="/privacy"' in c.get(path).text


def test_security_headers(app_):
    c = browser(app_)
    r = c.get("/")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert "no-store" in r.headers["cache-control"]
    assert "cache-control" not in TestClient(app_).get("/login").headers
    assert TestClient(app_).get("/docs").status_code == 404     # no public API explorer


# ================================================================ consent to train
def test_training_consent_is_off_unless_ticked_per_upload(app_):
    from app import db
    c = browser(app_)
    assert 'name="train_ok" value="1" >' in c.get("/analyses/new").text.replace("  ", " ")
    a = aid_of(upload(c))
    b = aid_of(upload(c, train_ok="1"))
    with db.session() as s:
        assert s.get(db.Analysis, a).train_ok is False and s.get(db.Analysis, b).train_ok is True
        assert db.user_by_email(s, "swimmer@example.com").train_consent is True     # remembered as the default
    assert "checked" in c.get("/analyses/new").text.split('name="train_ok"')[1][:40]
    c.post("/account/consent", data={})
    assert "checked" not in c.get("/analyses/new").text.split('name="train_ok"')[1][:40]


# ================================================================ feedback
def test_swimmer_can_say_whether_each_finding_was_right(app_):
    from app import feedback
    c = browser(app_)
    aid = aid_of(upload(c))
    page = c.get(f"/analyses/{aid}").text
    issue = re.search(r'name="issue_id" value="([a-z_]+)"', page).group(1)
    assert "Does this match what you or your coach see?" in page

    r = c.post(f"/analyses/{aid}/feedback", data={"issue_id": issue, "verdict": "wrong"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith(f"#fb-{issue}")
    c.post(f"/analyses/{aid}/feedback", data={"issue_id": issue, "verdict": "right"})        # changed their mind
    c.post(f"/analyses/{aid}/feedback", data={"issue_id": "", "verdict": "4", "comment": "Missed my crossover"})
    got = feedback.for_analysis(aid)
    assert got[issue]["verdict"] == "right" and got[""] == {"verdict": "4", "comment": "Missed my crossover"}
    page = c.get(f"/analyses/{aid}").text
    assert "Thanks, noted." in page and "Missed my crossover" in page and 'aria-pressed="true"' in page

    assert not feedback.save(c, aid, issue, "banana")     # nonsense verdicts are dropped


def test_feedback_only_on_your_own_finished_report(app_):
    owner, other = browser(app_), browser(app_, "other@example.com")
    aid = aid_of(upload(owner))
    assert other.post(f"/analyses/{aid}/feedback", data={"issue_id": "", "verdict": "5"}).status_code == 404
    assert TestClient(app_).post(f"/analyses/{aid}/feedback", data={"issue_id": "", "verdict": "5"}).status_code == 404


def test_agreement_table_puts_the_least_trusted_finding_first():
    from app import feedback
    rows = [{"issue_id": i, "verdict": v} for i, v in
            [("a", "right"), ("a", "right"), ("a", "wrong"), ("b", "wrong"), ("b", "unsure"), ("c", "unsure"), ("", "5")]]
    table = feedback.by_issue(rows)
    assert [(t["issue_id"], t["agree_pct"]) for t in table] == [("b", 0), ("a", 67), ("c", None)]


# ================================================================ review hold
def test_review_mode_holds_the_report_until_an_admin_releases_it(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_REVIEW", "1")
    monkeypatch.setenv("SWIMFORM_BASE_URL", "https://swimform.test")
    swimmer, admin = browser(app_), browser(app_, ADMIN)
    aid = aid_of(upload(swimmer))
    assert status_of(aid) == "review"
    page = text(swimmer.get(f"/analyses/{aid}").text)
    assert "being checked" in page and "Start here" not in page and "Work on this" not in page
    assert "Being checked" in text(swimmer.get("/").text)
    assert swimmer.post(f"/analyses/{aid}/feedback", data={"issue_id": "", "verdict": "5"}).status_code == 404
    assert outbox()[-1]["to"] == ADMIN and f"https://swimform.test/admin/analyses/{aid}" in outbox()[-1]["text"]

    review = admin.get(f"/admin/analyses/{aid}").text
    assert "Release report to swimmer" in review and "Your stroke report" in review   # the admin sees the report itself
    r = admin.post(f"/admin/analyses/{aid}/release", data={"coach_note": "Nice body line. Start with the catch drill."},
                   follow_redirects=False)
    assert r.status_code == 303 and status_of(aid) == "done"
    mail = outbox()[-1]
    assert mail["to"] == "swimmer@example.com" and f"https://swimform.test/analyses/{aid}" in mail["text"]
    assert "coach left you a note" in mail["text"]
    page = text(swimmer.get(f"/analyses/{aid}").text)
    assert "A note from your coach" in page and "Start with the catch drill." in page


def test_swimmers_cannot_release_reports(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_REVIEW", "1")
    swimmer = browser(app_)
    aid = aid_of(upload(swimmer))
    assert swimmer.post(f"/admin/analyses/{aid}/release", data={}).status_code == 404
    assert status_of(aid) == "review"


def test_fast_reports_do_not_email_but_slow_ones_do(app_, monkeypatch):
    from app import jobs
    c = browser(app_)
    upload(c)
    assert outbox() == []
    monkeypatch.setattr(jobs, "NOTIFY_AFTER_S", -1)       # every job now counts as slow
    aid = aid_of(upload(c))
    assert "report is ready" in outbox()[-1]["subject"] and aid in outbox()[-1]["text"]
    bad = aid_of(upload(c, side="lap.mp4", head="side_view.mov"))            # two side views, no head-on
    assert status_of(bad) == "rejected" and "couldn't use" in outbox()[-1]["subject"]


# ================================================================ collecting clips before there's a model
def test_collect_mode_stores_clips_and_waits_then_reruns(app_, monkeypatch, tmp_path):
    monkeypatch.setenv("SWIMFORM_POSE", "collect")
    swimmer, admin = browser(app_), browser(app_, ADMIN)
    page = text(swimmer.get("/analyses/new").text)
    assert "Early beta" in page and "Send my clips" in page
    aid = aid_of(upload(swimmer, train_ok="1"))
    assert status_of(aid) == "waiting"
    assert (tmp_path / "analyses" / aid / "side.mp4").exists()
    assert "still finishing the analysis" in text(swimmer.get(f"/analyses/{aid}").text)
    assert "report is coming" in text(swimmer.get("/").text)

    assert admin.post("/admin/rerun-waiting", follow_redirects=False).headers["location"].endswith("note=norerun")
    assert status_of(aid) == "waiting"                    # still no model: nothing to run with

    monkeypatch.setenv("SWIMFORM_POSE", "synthetic")      # the model arrives
    assert admin.post("/admin/rerun-waiting", follow_redirects=False).headers["location"].endswith("note=waiting")
    assert status_of(aid) == "done"
    assert outbox()[-1]["to"] == "swimmer@example.com" and "report is ready" in outbox()[-1]["subject"]
    assert 'id="work-h"' in swimmer.get(f"/analyses/{aid}").text


def test_rerun_one_analysis_and_refuse_when_clips_are_gone(app_):
    from app import pipeline
    swimmer, admin = browser(app_), browser(app_, ADMIN)
    aid = aid_of(upload(swimmer))
    r = admin.post(f"/admin/analyses/{aid}/rerun", follow_redirects=False)
    assert r.headers["location"].endswith("note=rerun") and status_of(aid) == "done"
    pipeline.remove(aid)
    assert admin.post(f"/admin/analyses/{aid}/rerun", follow_redirects=False).headers["location"].endswith("note=norerun")


# ================================================================ plans and limits
def test_no_limit_and_no_plan_pages_until_configured(app_):
    c = browser(app_)
    assert c.get("/billing").status_code == 404 and 'href="/billing"' not in c.get("/").text
    assert all(upload(c).status_code == 303 for _ in range(3))


def test_free_limit_counts_only_analyses_that_ran(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_FREE_ANALYSES", "2")
    c = browser(app_)
    assert "2 of 2 free analyses left this month" in text(c.get("/analyses/new").text)
    upload(c, side="lap.mp4", head="side_view.mov")                 # rejected: doesn't count
    assert upload(c).status_code == 303 and upload(c).status_code == 303
    r = upload(c)
    assert r.status_code == 402 and "used this month's 2 free analyses" in text(r.text)
    page = text(c.get("/analyses/new").text)
    assert "resets on the 1st" in page and "Analyse my stroke" not in page
    assert "2 of 2 used this month" in text(c.get("/billing").text)


def test_pro_and_admin_accounts_are_not_limited(app_, monkeypatch):
    from app import billing, db
    monkeypatch.setenv("SWIMFORM_FREE_ANALYSES", "0")
    swimmer, admin = browser(app_), browser(app_, ADMIN)
    assert upload(swimmer).status_code == 402 and upload(admin).status_code == 303
    with db.session() as s:
        uid = db.user_by_email(s, "swimmer@example.com").id
    assert admin.post(f"/admin/users/{uid}/plan", data={"plan": "pro"}, follow_redirects=False).status_code == 303
    assert upload(swimmer).status_code == 303
    with db.session() as s:
        assert billing.status(s.get(db.User, uid))["limit"] is None


# ================================================================ Stripe
@pytest.fixture()
def stripe_on(monkeypatch):
    from app import billing
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setenv("STRIPE_PRICE_ID", "price_123")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setenv("SWIMFORM_BASE_URL", "https://swimform.test")
    monkeypatch.setenv("SWIMFORM_PRICE_LABEL", "$12 a month")
    calls = []

    def fake(method, path, data=None):
        calls.append((method, path, data or {}))
        return {"url": f"https://stripe.test{path}"}
    monkeypatch.setattr(billing, "_stripe", fake)
    return calls


def signed(event, secret="whsec_test", t=None):
    body = json.dumps(event).encode()
    t = int(t or time.time())
    sig = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return body, {"stripe-signature": f"t={t},v1={sig}", "content-type": "application/json"}


def test_upgrade_goes_to_stripe_checkout(app_, stripe_on):
    from app import db
    c = browser(app_)
    page = text(c.get("/billing").text)
    assert "$12 a month" in page and "Upgrade to Pro" in page
    r = c.post("/billing/checkout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "https://stripe.test/checkout/sessions"
    method, path, data = stripe_on[-1]
    with db.session() as s:
        uid = db.user_by_email(s, "swimmer@example.com").id
    assert data["mode"] == "subscription" and data["line_items[0][price]"] == "price_123"
    assert data["client_reference_id"] == str(uid) and data["customer_email"] == "swimmer@example.com"
    assert data["success_url"] == "https://swimform.test/billing?done=1"
    assert TestClient(app_).post("/billing/checkout", follow_redirects=False).headers["location"].startswith("/login")


def test_webhooks_move_the_account_between_plans(app_, stripe_on):
    from app import db
    c = browser(app_)
    with db.session() as s:
        uid = db.user_by_email(s, "swimmer@example.com").id

    def plan():
        with db.session() as s:
            u = s.get(db.User, uid)
            return u.plan, u.stripe_customer, u.stripe_subscription

    def send(event):
        body, headers = signed(event)
        return c.post("/webhooks/stripe", content=body, headers=headers)

    r = send({"type": "checkout.session.completed", "data": {"object": {
        "client_reference_id": str(uid), "customer": "cus_1", "subscription": "sub_1"}}})
    assert r.status_code == 200 and plan() == ("pro", "cus_1", "sub_1")
    assert "Manage or cancel" in c.get("/billing").text
    assert c.post("/billing/portal", follow_redirects=False).headers["location"] == "https://stripe.test/billing_portal/sessions"
    assert stripe_on[-1][2]["customer"] == "cus_1"

    send({"type": "customer.subscription.updated", "data": {"object": {"id": "sub_1", "customer": "cus_1", "status": "past_due"}}})
    assert plan()[0] == "pro"                                  # Stripe is still retrying the card
    send({"type": "customer.subscription.updated", "data": {"object": {"id": "sub_1", "customer": "cus_1", "status": "unpaid"}}})
    assert plan() == ("free", "cus_1", None)
    send({"type": "customer.subscription.updated", "data": {"object": {"id": "sub_2", "customer": "cus_1", "status": "active"}}})
    assert plan() == ("pro", "cus_1", "sub_2")
    send({"type": "customer.subscription.deleted", "data": {"object": {"id": "sub_2", "customer": "cus_1", "status": "canceled"}}})
    assert plan()[0] == "free"
    assert send({"type": "invoice.paid", "data": {"object": {}}}).json() == {"result": "ignored"}


@pytest.mark.parametrize("tamper", ["secret", "body", "old", "missing"])
def test_forged_webhooks_are_refused(app_, stripe_on, tamper):
    from app import db
    c = browser(app_)
    with db.session() as s:
        uid = db.user_by_email(s, "swimmer@example.com").id
    event = {"type": "checkout.session.completed", "data": {"object": {"client_reference_id": str(uid), "customer": "cus_x"}}}
    body, headers = signed(event, secret="whsec_wrong" if tamper == "secret" else "whsec_test",
                           t=time.time() - 3600 if tamper == "old" else None)
    if tamper == "body":
        body = body.replace(b"cus_x", b"cus_y")
    if tamper == "missing":
        headers.pop("stripe-signature")
    assert TestClient(app_).post("/webhooks/stripe", content=body, headers=headers).status_code == 400
    with db.session() as s:
        assert s.get(db.User, uid).plan == "free"


def test_deleting_an_account_cancels_its_subscription_and_feedback(app_, stripe_on):
    from app import db
    c = browser(app_)
    aid = aid_of(upload(c))
    c.post(f"/analyses/{aid}/feedback", data={"issue_id": "", "verdict": "3"})
    with db.session() as s:
        u = db.user_by_email(s, "swimmer@example.com")
        u.stripe_customer, u.stripe_subscription, u.plan = "cus_1", "sub_9", "pro"
        s.commit()
    assert c.post("/account/delete", data={"confirm": "delete", "password": PW}, follow_redirects=False).status_code == 303
    assert ("DELETE", "/subscriptions/sub_9", {}) in stripe_on
    with db.session() as s:
        assert s.query(db.User).count() == 0 and s.query(db.Feedback).count() == 0


# ================================================================ cloud storage
class FakeBucket:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = open(path, "rb").read()

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise KeyError(key)
        open(path, "wb").write(self.objects[key])

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            del self.objects[o["Key"]]


@pytest.fixture()
def bucket(monkeypatch):
    from app import storage
    b = FakeBucket()
    monkeypatch.setenv("SWIMFORM_STORAGE", "s3")
    monkeypatch.setenv("S3_BUCKET", "swimform-clips")
    monkeypatch.setattr(storage, "_client", lambda: b)
    return b


def test_finished_analyses_move_to_the_bucket_and_survive_a_wiped_disk(app_, bucket, tmp_path):
    import shutil
    c = browser(app_)
    aid = aid_of(upload(c))
    keys = {k.split("/")[-1] for k in bucket.objects}
    assert keys == {"side.mp4", "head_on.mov", "upload.json", "report.json"}
    local = tmp_path / "analyses" / aid
    assert not (local / "side.mp4").exists() and (local / "report.json").exists()   # clips gone, report cached

    shutil.rmtree(tmp_path / "analyses")                                             # the host redeploys
    assert "Your stroke report" in c.get(f"/analyses/{aid}").text
    admin = browser(app_, ADMIN)
    assert admin.post(f"/admin/analyses/{aid}/rerun", follow_redirects=False).headers["location"].endswith("note=rerun")
    assert status_of(aid) == "done"


def test_deleting_the_account_empties_the_bucket(app_, bucket):
    c = browser(app_)
    upload(c)
    assert bucket.objects
    c.post("/account/delete", data={"confirm": "delete", "password": PW})
    assert bucket.objects == {}


# ================================================================ admin console
def test_admin_console_is_invisible_to_everyone_else(app_):
    swimmer = browser(app_)
    for path in ("/admin", "/admin/analyses", "/admin/users", "/admin/invites", "/admin/feedback", "/admin/feedback.csv"):
        assert swimmer.get(path).status_code == 404
        assert TestClient(app_).get(path).status_code == 404
    assert swimmer.post("/admin/invites", data={"note": "me"}).status_code == 404
    assert "Beta console" not in swimmer.get("/").text


def test_admin_console_pages(app_, monkeypatch):
    from app import db
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    admin = browser(app_, ADMIN)
    assert "Beta console" in admin.get("/").text

    TestClient(app_).post("/waitlist", data={"email": "keen@example.com", "note": "ironman in june"})
    page = admin.get("/admin/invites").text
    assert "keen@example.com" in page and "ironman in june" in page
    with db.session() as s:
        wid = s.query(db.Waitlist).one().id
    assert admin.post(f"/admin/waitlist/{wid}/invite", follow_redirects=False).status_code == 303
    mail = outbox()[-1]
    code = re.search(r"invite=([A-Z0-9]+)", mail["text"]).group(1)
    assert mail["to"] == "keen@example.com"
    swimmer = browser(app_, "keen@example.com", invite=code)

    admin.post("/admin/invites", data={"note": "Tuesday squad", "max_uses": "20"})
    assert "Tuesday squad" in admin.get("/admin/invites").text and "0 of 20" in text(admin.get("/admin/invites").text)

    aid = aid_of(upload(swimmer, train_ok="1"))
    issue = re.search(r'name="issue_id" value="([a-z_]+)"', swimmer.get(f"/analyses/{aid}").text).group(1)
    swimmer.post(f"/analyses/{aid}/feedback", data={"issue_id": issue, "verdict": "wrong", "comment": "=cmd() my coach disagrees"})

    overview = text(admin.get("/admin").text)
    assert "What's connected" in overview and "Pose model" in overview and "0%" in overview
    assert "keen@example.com" in admin.get("/admin/analyses").text
    assert "keen@example.com" in admin.get("/admin/analyses?status=done").text
    assert "keen@example.com" not in admin.get("/admin/analyses?status=failed").text
    report = text(admin.get(f"/admin/analyses/{aid}").text)
    assert "Admin view" in report and "OK to train on" in report and "my coach disagrees" in report
    assert "Does this match what you" not in report            # admins don't answer for the swimmer
    assert code in admin.get("/admin/users").text
    assert "my coach disagrees" in admin.get("/admin/feedback").text
    csv_ = admin.get("/admin/feedback.csv")
    assert csv_.headers["content-type"].startswith("text/csv") and "keen@example.com" in csv_.text
    assert "'=cmd()" in csv_.text                              # can't run as a spreadsheet formula
    assert admin.post("/admin/test-email", follow_redirects=False).headers["location"].endswith("note=mail")
    assert outbox()[-1]["to"] == ADMIN


def test_connections_report_what_is_missing(app_, monkeypatch, tmp_path):
    from app import connections
    state = lambda: {i["name"]: i["state"] for i in connections.check()}   # noqa: E731
    s = state()
    assert s["Email"] == "todo" and s["Pose model"] == "todo" and s["Payments"] == "off" and s["Admins"] == "on"
    assert not connections.ready()

    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_1")
    monkeypatch.setenv("SWIMFORM_POSE", "yolo")
    monkeypatch.setenv("SWIMFORM_WEIGHTS", str(tmp_path / "missing.pt"))
    s = state()
    assert s["Payments"] == "problem" and s["Pose model"] == "problem"

    weights = tmp_path / "swimform.pt"
    weights.write_bytes(b"x")
    for k, v in {"STRIPE_PRICE_ID": "price_1", "STRIPE_WEBHOOK_SECRET": "whsec_1", "SWIMFORM_WEIGHTS": str(weights),
                 "SWIMFORM_BASE_URL": "https://swimform.test", "SWIMFORM_SECURE_COOKIES": "1", "SWIMFORM_EMAIL": "smtp",
                 "SMTP_HOST": "smtp.test", "SWIMFORM_EMAIL_FROM": "SwimForm <hi@swimform.test>",
                 "SWIMFORM_CONTACT_EMAIL": "hi@swimform.test"}.items():
        monkeypatch.setenv(k, v)
    assert connections.ready(), [i for i in connections.check() if i["state"] in ("todo", "problem")]


def test_check_command_exit_code(app_, capsys):
    from app import check
    assert check.main() == 1
    assert "Not ready yet" in capsys.readouterr().out


# ================================================================ upgrades from an older database
@pytest.mark.sqlite_only
def test_a_database_from_before_the_beta_is_upgraded_in_place(tmp_path, monkeypatch):
    import sqlite3
    from app import db
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(320), name VARCHAR(120), password_hash VARCHAR(255),
                            google_sub VARCHAR(255), created DATETIME);
        CREATE TABLE analyses (id VARCHAR(32) PRIMARY KEY, user_id INTEGER, created DATETIME, lanes TEXT,
                               flagged INTEGER, estimated INTEGER, unchecked INTEGER);
        INSERT INTO users VALUES (1, 'old@example.com', 'Old', 'x', NULL, '2026-10-01 00:00:00');
        INSERT INTO analyses VALUES ('abc', 1, '2026-10-01 00:00:00', '', 1, 0, 0);
    """)
    con.commit()
    con.close()
    db.init(f"sqlite:///{path}")
    with db.session() as s:
        u, a = s.get(db.User, 1), s.get(db.Analysis, "abc")
        assert (u.plan, u.train_consent, u.terms_accepted, u.stripe_customer) == ("free", False, None, None)
        assert (a.status, a.train_ok, a.coach_note, a.problems) == ("done", False, "", "")
        assert s.query(db.Invite).count() == 0


# ================================================================ beta clips -> labeling
def test_only_consented_clips_are_exported_for_labeling(app_, tmp_path):
    from app import export_clips
    yes, no = browser(app_, "yes@example.com"), browser(app_, "no@example.com")
    a = aid_of(upload(yes, train_ok="1"))
    upload(no)
    out = tmp_path / "for_labeling"
    copied, total = export_clips.export(out)
    names = sorted(p.name for p in out.glob("*.m*"))
    assert (copied, total) == (2, 2) and all(a in n for n in names) and len(names) == 2
    assert names[0].startswith("u1_headon_") and names[1].startswith("u1_side_")
    assert a in (out / "manifest.csv").read_text()
    assert export_clips.export(out) == (0, 2)                    # running it again copies nothing new

    yes.post("/account/delete", data={"confirm": "delete", "password": PW})
    assert export_clips.export(tmp_path / "later") == (0, 0)     # a deleted account's clips are never exported


# ================================================================ ship polish
def test_unknown_addresses_get_a_real_page(app_):
    c = TestClient(app_)
    r = c.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404 and "We can't find that page" in text(r.text) and "SwimForm" in r.text
    assert c.get("/no-such-page", headers={"accept": "application/json"}).json() == {"detail": "Not Found"}
    r = c.get("/logout", headers={"accept": "text/html"})                 # a POST-only address, visited by hand
    assert r.status_code == 405 and "That didn't work" in text(r.text)


def test_a_crash_shows_an_apology_not_a_stack_trace(app_, monkeypatch):
    from app import presenter
    c = TestClient(app_, raise_server_exceptions=False)
    signup(c)
    aid = aid_of(upload(c))
    monkeypatch.setattr(presenter, "build", lambda *a, **k: 1 / 0)
    r = c.get(f"/analyses/{aid}")
    assert r.status_code == 500 and "Something went wrong on our side" in r.text and "ZeroDivision" not in r.text


def test_sample_report_is_public_and_asks_for_nothing(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    c = TestClient(app_)
    r = c.get("/sample")
    page = text(r.text)
    assert r.status_code == 200 and "Sample report" in page and "This is a sample report" in page
    assert "Dropped elbow" in page and "Seen on your left arm only" in page
    assert "/feedback" not in r.text and "Analyse new clips" not in page
    assert "I have an invite" in page and "Join the waitlist" in page
    assert 'href="/sample"' in c.get("/login").text and 'href="/sample"' in c.get("/waitlist").text


def test_icons_share_card_and_robots(app_):
    c = TestClient(app_)
    assert c.get("/favicon.ico").headers["content-type"] == "image/png"
    for path in ("/static/favicon.svg", "/static/apple-touch-icon.png", "/static/share.png"):
        assert c.get(path).status_code == 200
    head = c.get("/login").text
    assert 'property="og:image"' in head and 'name="description"' in head
    assert "Disallow: /" in c.get("/robots.txt").text


def test_emails_go_out_as_branded_html_with_a_button(app_, monkeypatch):
    from app import jobs, mailer
    monkeypatch.setenv("SWIMFORM_BASE_URL", "https://swimform.test")
    monkeypatch.setattr(jobs, "NOTIFY_AFTER_S", -1)
    c = browser(app_, name="<b>Sam</b>")
    aid = aid_of(upload(c))
    html_ = outbox()[-1]["html"]
    assert f'href="https://swimform.test/analyses/{aid}"' in html_ and "Open your report" in html_
    assert "&lt;b&gt;Sam&lt;/b&gt;" in html_ and "<b>Sam</b>" not in html_     # names can't inject markup
    assert "Create your account" in mailer.render_html("x", "Hi,\n\nhttps://swimform.test/signup?invite=ABC\n")


def test_invite_error_still_offers_the_waitlist(app_, monkeypatch):
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    r = signup(TestClient(app_))
    assert r.status_code == 422 and 'href="/waitlist"' in r.text.split('id="invite-err"')[1][:400]


# ------------------------------------------------------------------ tracked frames
def _film(track, path, size=(480, 270)):
    """A real video of a pose track: water-blue frames with the swimmer's points as dots."""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), track.fps, size)
    sx, sy = size[0] / track.W, size[1] / track.H
    for pts in track.data:
        img = np.full((size[1], size[0], 3), (150, 110, 40), np.uint8)
        for x, y, c in pts:
            if c > 0.5:
                cv2.circle(img, (int(x * sx), int(y * sy)), 3, (255, 255, 255), -1)
        w.write(img)
    w.release()


def _camera_provider(monkeypatch):
    """Stand in for the trained model: keypoints that really do belong to the uploaded video."""
    from app import pipeline

    def fake(paths, names):
        tracks = pipeline.pose_synthetic(paths, names)
        for slot, tr in tracks.items():
            _film(tr, paths[slot])
        return tracks
    monkeypatch.setitem(pipeline.PROVIDERS, "yolo", fake)
    monkeypatch.setenv("SWIMFORM_POSE", "yolo")
    monkeypatch.setenv("SWIMFORM_WEIGHTS", __file__)   # only has to exist


def test_report_shows_the_swimmers_own_frames_with_the_tracked_points(app_, monkeypatch):
    import cv2
    from app import tracked
    client = browser(app_)
    _camera_provider(monkeypatch)
    aid = aid_of(upload(client))
    page_ = client.get(f"/analyses/{aid}")
    assert "What we tracked" in page_.text and f"/analyses/{aid}/tracked/side.jpg" in page_.text
    for slot in ("side", "head_on"):
        img = client.get(f"/analyses/{aid}/tracked/{slot}.jpg")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert "private" in img.headers["cache-control"]
        still = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
        assert still.shape[1] <= tracked.MAX_WIDTH
        # the skeleton was drawn: the right side's coral is in the picture
        b, g, rd = (still[..., i].astype(int) for i in range(3))
        assert ((rd > 200) & (b < 130) & (g < 130)).sum() > 20
    assert client.get(f"/analyses/{aid}/tracked/side_uw.jpg").status_code == 404     # no such clip
    assert client.get(f"/analyses/{aid}/tracked/nonsense.jpg").status_code == 404


def test_tracked_frames_are_private_and_absent_in_demo_mode(app_, monkeypatch):
    client = browser(app_)
    aid = aid_of(upload(client))                         # demo mode: simulated swimmer
    assert "What we tracked" not in client.get(f"/analyses/{aid}").text
    assert client.get(f"/analyses/{aid}/tracked/side.jpg").status_code == 404
    _camera_provider(monkeypatch)
    aid = aid_of(upload(client))
    assert client.get(f"/analyses/{aid}/tracked/side.jpg").status_code == 200
    client.post("/logout")
    assert client.get(f"/analyses/{aid}/tracked/side.jpg").status_code == 404
    assert browser(app_, "other@example.com").get(f"/analyses/{aid}/tracked/side.jpg").status_code == 404
    assert browser(app_, ADMIN).get(f"/analyses/{aid}/tracked/side.jpg").status_code == 200   # coach review


def test_tracked_frame_gives_up_quietly(tmp_path):
    from app import tracked
    from swimform.schema import SIDE
    from swimform.synthetic import Swimmer, generate
    tr = generate(SIDE, Swimmer(), quality="typical")
    assert tracked.render(tmp_path / "not-a-video.mp4", tr, tmp_path / "out.jpg") is False
    tr.data[:, :, 2] = 0.1                                # nothing seen anywhere
    assert tracked.pick_frame(tr) is None


def test_front_door_follows_the_signup_mode(app_, monkeypatch):
    from starlette.testclient import TestClient
    from app.main import app as web
    c = TestClient(web)                                       # signed out
    monkeypatch.setenv("SWIMFORM_SIGNUP", "closed")
    page_ = c.get("/").text
    assert 'href="/waitlist"' in page_ and "Create your account" not in page_
    monkeypatch.setenv("SWIMFORM_SIGNUP", "invite")
    page_ = c.get("/").text
    assert "invite-only" in page_ and 'href="/signup"' in page_
    assert page_.count('class="seg ') >= 28                    # the sample's 14 checks, shown twice
