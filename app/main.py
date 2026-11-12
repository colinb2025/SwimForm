"""
SwimForm web app.

  uvicorn app.main:app --reload

Public:   GET /login, /signup, /forgot, /reset, /guide, /auth/google/*
          POST /login, /signup, /forgot, /reset
          /waitlist, /terms, /privacy; POST /webhooks/stripe (signed by Stripe)
Account:  GET / -> your reports, /analyses/new, /analyses/{id}, /analyses/{id}/status, /account, /billing
          POST /analyses, /analyses/{id}/feedback, /logout, /billing/{checkout,portal},
          /account/{name,email,password,consent,sessions,delete}
Admins:   /admin/... (app/admin.py), for the emails in SWIMFORM_ADMINS
A report is visible only to the account that made it; anyone else gets "not found".
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import account, admin, auth, beta, billing, db, feedback, google, jobs, mailer, pipeline, presenter, storage, tracked
from .web import base_url, login_redirect, page, rules, safe_next as _safe_next, templates  # noqa: F401

HERE = Path(__file__).resolve().parent


def _sentry():
    """Send crashes to Sentry when SENTRY_DSN is set and sentry-sdk is installed."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0, send_default_pii=False)
    except ImportError:
        import logging
        logging.getLogger("swimform").warning("SENTRY_DSN is set but sentry-sdk is not installed")


@asynccontextmanager
async def lifespan(_app):
    _sentry()
    db.init()
    jobs.resume()
    yield


app = FastAPI(title="SwimForm", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.include_router(admin.router)


from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402


def _wants_html(request):
    return "text/html" in request.headers.get("accept", "") or "*/*" in request.headers.get("accept", "*/*")


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """Unknown addresses and refused methods get a page, not a line of JSON."""
    if not _wants_html(request) or request.url.path.startswith("/webhooks/"):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    if not hasattr(request.state, "user"):
        request.state.user = None
    if exc.status_code == 404:
        return page(request, "error.html", {"title": "We can't find that page",
                                            "text": "The link may be mistyped, or the page may have moved."}, 404)
    return page(request, "error.html", {"title": "That didn't work",
                                        "text": "Go back and try again. If it keeps happening, let us know."}, exc.status_code)


@app.exception_handler(Exception)
async def crash(request: Request, exc: Exception):
    import logging
    logging.getLogger("swimform").exception("unhandled error on %s %s", request.method, request.url.path)
    request.state.user = getattr(request.state, "user", None)
    try:
        return page(request, "error.html", {"title": "Something went wrong on our side",
                                            "text": "We've logged it. Please try again in a minute."}, 500)
    except Exception:   # noqa: BLE001 - never let the error page itself take the response down
        return HTMLResponse("<h1>Something went wrong</h1><p>Please try again in a minute.</p>", status_code=500)


def summarise(record):
    """The small summary the reports list shows, from a finished analysis."""
    view = presenter.build(record, rules())
    return {"lanes": ",".join(l["status"] for l in view["lanes"]), "flagged": view["counts"]["flagged"],
            "headline": view["headline"][:300],
            "estimated": view["counts"]["estimated"], "unchecked": view["counts"]["unchecked"]}


jobs.summarise = summarise


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and not auth.same_origin(request):
        return HTMLResponse("Request refused: it came from another site.", status_code=403)
    request.state.user = auth.current_user(request)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if auth.SECURE_COOKIES:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if request.state.user is not None:   # never let a shared cache keep someone's private pages
        response.headers.setdefault("Cache-Control", "private, no-store")
    return response


# ------------------------------------------------------------------ accounts
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", deleted: str = ""):
    if request.state.user:
        return RedirectResponse(_safe_next(next), status_code=303)
    return page(request, "login.html", {"next": _safe_next(next), "email": "", "error": None,
                                        "notice": "Your account and everything in it has been deleted." if deleted else None})


@app.post("/login")
def login(request: Request, email: str = Form(""), password: str = Form(""), next: str = Form("/")):
    key = email.strip().lower()
    nxt = _safe_next(next)
    if auth.limiter.blocked(key):
        return page(request, "login.html", {"next": nxt, "email": email,
                    "error": "Too many attempts. Wait 15 minutes, then try again."}, 429)
    with db.session() as s:
        user = db.user_by_email(s, key)
    if user is None or not auth.verify_password(password, user.password_hash):
        auth.limiter.fail(key)
        return page(request, "login.html", {"next": nxt, "email": email,
                    "error": "That email and password don't match an account."}, 401)
    auth.limiter.clear(key)
    resp = RedirectResponse(nxt, status_code=303)
    auth.set_cookie(resp, auth.start_session(user.id))
    return resp


def _signup_ctx(**kw):
    ctx = {"name": "", "email": "", "invite": "", "errors": {}, "signup_mode": beta.mode()}
    ctx.update(kw)
    return ctx


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, invite: str = ""):
    if request.state.user:
        return RedirectResponse("/", status_code=303)
    return page(request, "signup.html", _signup_ctx(invite=beta.normalise(invite)))


@app.post("/signup")
def signup(request: Request, name: str = Form(""), email: str = Form(""), password: str = Form(""),
           invite: str = Form("")):
    email_n = email.strip().lower()
    errors = {}
    if problem := auth.email_problem(email_n):
        errors["email"] = problem
    if problem := auth.password_problem(password):
        errors["password"] = problem
    if not errors and (problem := beta.gate(email_n, invite)):
        errors["invite"] = problem
    with db.session() as s:
        if not errors and db.user_by_email(s, email_n):
            errors["email"] = "There's already an account with this email. Log in instead."
        if not errors and not beta.redeem(s, invite):
            errors["invite"] = "That invite code has already been used."
        if errors:
            s.rollback()
            return page(request, "signup.html", _signup_ctx(name=name, email=email, invite=invite, errors=errors), 422)
        user = db.User(email=email_n, name=name.strip()[:120], password_hash=auth.hash_password(password),
                       terms_accepted=db.now(), invite_code=beta.normalise(invite) or None)
        s.add(user)
        s.commit()
        uid = user.id
    resp = RedirectResponse("/analyses/new", status_code=303)
    auth.set_cookie(resp, auth.start_session(uid))
    return resp


@app.get("/waitlist", response_class=HTMLResponse)
def waitlist_page(request: Request):
    return page(request, "waitlist.html", {"joined": False, "email": "", "error": None})


@app.post("/waitlist")
def waitlist(request: Request, email: str = Form(""), note: str = Form("")):
    key = email.strip().lower()
    if problem := auth.email_problem(key):
        return page(request, "waitlist.html", {"joined": False, "email": email, "error": problem}, 422)
    if not auth.waitlist_limiter.blocked(_client_ip(request)):
        auth.waitlist_limiter.fail(_client_ip(request))
        beta.join_waitlist(key, note)
    return page(request, "waitlist.html", {"joined": True, "email": email, "error": None})


def _client_ip(request):
    return request.client.host if request.client else "unknown"


@app.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return page(request, "terms.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return page(request, "privacy.html")


# ------------------------------------------------------------------ password reset
@app.get("/forgot", response_class=HTMLResponse)
def forgot_page(request: Request):
    return page(request, "forgot.html", {"sent": False, "email": ""})


@app.post("/forgot")
def forgot(request: Request, email: str = Form("")):
    key = email.strip().lower()
    with db.session() as s:
        user = db.user_by_email(s, key) if key else None
    # Same answer whether or not the account exists, so this page can't be used to
    # find out who has one. Rate-limited per address so it can't flood an inbox.
    if user is not None and not auth.reset_limiter.blocked(key):
        auth.reset_limiter.fail(key)
        link = f"{base_url(request)}/reset?token={auth.create_reset(user.id)}"
        greeting = f"Hi {user.name.split()[0]}," if user.name else "Hi,"
        mailer.send(user.email, "Reset your SwimForm password",
                    f"{greeting}\n\nSomeone asked to reset the password for your SwimForm account. "
                    f"If it was you, use this link within the next hour:\n\n{link}\n\n"
                    "If it wasn't you, you can ignore this email. Your password won't change.\n")
    return page(request, "forgot.html", {"sent": True, "email": email})


@app.get("/reset", response_class=HTMLResponse)
def reset_page(request: Request, token: str = ""):
    valid = auth.reset_user(token) is not None
    return page(request, "reset.html", {"token": token, "valid": valid, "error": None}, 200 if valid else 400)


@app.post("/reset")
def reset(request: Request, token: str = Form(""), password: str = Form("")):
    if auth.reset_user(token) is None:
        return page(request, "reset.html", {"token": "", "valid": False, "error": None}, 400)
    if problem := auth.password_problem(password):
        return page(request, "reset.html", {"token": token, "valid": True, "error": problem}, 422)
    user = auth.complete_reset(token, password)
    if user is None:
        return page(request, "reset.html", {"token": "", "valid": False, "error": None}, 400)
    resp = RedirectResponse("/", status_code=303)
    auth.set_cookie(resp, auth.start_session(user.id))
    return resp


# ------------------------------------------------------------------ Google sign-in
@app.get("/auth/google/start")
def google_start(request: Request, next: str = "/", invite: str = ""):
    if not google.configured():
        return page(request, "missing.html", {}, 404)
    return RedirectResponse(google.start(request, _safe_next(next), beta.normalise(invite)), status_code=303)


@app.get("/auth/google/callback")
def google_callback(request: Request, state: str = "", code: str = "", error: str = ""):
    if not google.configured():
        return page(request, "missing.html", {}, 404)
    if error:
        msg = "Google sign-in was cancelled." if error == "access_denied" else "Google sign-in didn't complete. Please try again."
        return page(request, "login.html", {"next": "/", "email": "", "error": msg}, 400)
    try:
        user, nxt = google.finish(request, state, code)
    except google.GoogleSignInError as e:
        return page(request, "login.html", {"next": "/", "email": "", "error": str(e)}, 400)
    resp = RedirectResponse(_safe_next(nxt), status_code=303)
    auth.set_cookie(resp, auth.start_session(user.id))
    return resp


@app.post("/logout")
def logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    resp = RedirectResponse("/login", status_code=303)
    auth.clear_cookie(resp)
    return resp


# ------------------------------------------------------------------ account settings
SAVED = {"consent": "Saved.", "name": "Name saved.", "email": "Email changed. We sent a note to your old address.",
         "password": "Password saved. Any other devices have been logged out.",
         "sessions": "Every other device has been logged out."}


def account_page(request, errors=None, status_code=200, saved=None):
    user = request.state.user
    return page(request, "account.html", {"errors": errors or {}, "saved": SAVED.get(saved or ""),
                "has_password": bool(user.password_hash), "google_linked": bool(user.google_sub),
                "plan": billing.status(user),
                "confirm_word": account.CONFIRM_WORD}, status_code)


def _saved(what):
    return RedirectResponse(f"/account?saved={what}", status_code=303)


@app.get("/account", response_class=HTMLResponse)
def account_settings(request: Request, saved: str = ""):
    if request.state.user is None:
        return login_redirect(request)
    return account_page(request, saved=saved)


@app.post("/account/name")
def account_name(request: Request, name: str = Form("")):
    if request.state.user is None:
        return login_redirect(request)
    account.update_name(request.state.user, name)
    return _saved("name")


@app.post("/account/consent")
def account_consent(request: Request, train_consent: str = Form("")):
    if request.state.user is None:
        return login_redirect(request)
    account.set_train_consent(request.state.user, bool(train_consent))
    return _saved("consent")


@app.post("/account/email")
def account_email(request: Request, email: str = Form(""), password: str = Form("")):
    if request.state.user is None:
        return login_redirect(request)
    if err := account.change_email(request.state.user, email, password):
        return account_page(request, {"email": err}, 422)
    return _saved("email")


@app.post("/account/password")
def account_password(request: Request, current: str = Form(""), new: str = Form("")):
    if request.state.user is None:
        return login_redirect(request)
    if err := account.change_password(request.state.user, current, new, request.cookies.get(auth.COOKIE)):
        return account_page(request, {"password": err}, 422)
    return _saved("password")


@app.post("/account/sessions")
def account_sessions(request: Request):
    if request.state.user is None:
        return login_redirect(request)
    account.sign_out_others(request.state.user, request.cookies.get(auth.COOKIE))
    return _saved("sessions")


@app.post("/account/delete")
def account_delete(request: Request, confirm: str = Form(""), password: str = Form("")):
    if request.state.user is None:
        return login_redirect(request)
    if err := account.delete_account(request.state.user, confirm, password):
        return account_page(request, {"delete": err}, 422)
    resp = RedirectResponse("/login?deleted=1", status_code=303)
    auth.clear_cookie(resp)
    return resp


# ------------------------------------------------------------------ reports
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    rows = auth.list_analyses(user)
    items = [{"id": a.id, "date": f"{a.created.day} {a.created:%b %Y}",
              "iso": (a.created if a.created.tzinfo else a.created.replace(tzinfo=db.now().tzinfo)).isoformat(),
              "lanes": a.lanes.split(",") if a.lanes else [],
              "flagged": a.flagged, "estimated": a.estimated, "unchecked": a.unchecked,
              "headline": a.headline, "status": a.status} for a in rows]
    return page(request, "reports.html", {"items": items})


def _upload_ctx(user, problems=None):
    return {"problems": problems or {}, "plan": billing.status(user), "train_consent": user.train_consent,
            "collecting": os.environ.get("SWIMFORM_POSE", "synthetic") == "collect"}


def first_run_on():
    """Show the filming walkthrough before a first upload? SWIMFORM_FIRST_RUN=0 turns it off."""
    return os.environ.get("SWIMFORM_FIRST_RUN", "1") != "0"


@app.get("/analyses/new", response_class=HTMLResponse)
def upload_page(request: Request):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    if first_run_on() and user.trained_at is None:
        return RedirectResponse("/start", status_code=303)
    return page(request, "upload.html", _upload_ctx(user))


@app.get("/start", response_class=HTMLResponse)
def start_page(request: Request):
    if request.state.user is None:
        return login_redirect(request)
    return page(request, "start.html")


@app.post("/start")
def start_done(request: Request):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    with db.session() as s:
        row = s.get(db.User, user.id)
        if row.trained_at is None:
            row.trained_at = db.now()
            s.commit()
    return RedirectResponse("/analyses/new", status_code=303)


@app.post("/analyses")
async def create_analysis(request: Request, side: UploadFile | None = None, head_on: UploadFile | None = None,
                          side_uw: UploadFile | None = None, head_on_uw: UploadFile | None = None,
                          train_ok: str = Form("")):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    if not billing.status(user)["can_analyse"]:
        return page(request, "upload.html", _upload_ctx(user), 402)
    files = {}
    for slot, up in (("side", side), ("head_on", head_on), ("side_uw", side_uw), ("head_on_uw", head_on_uw)):
        if up is not None and up.filename:
            files[slot] = (up.filename, up.file)
    try:
        aid = pipeline.queue(files)
    except pipeline.UploadProblem as e:
        return page(request, "upload.html", _upload_ctx(user, e.problems), 422)
    account.set_train_consent(user, bool(train_ok))   # remembered as the default for next time
    with db.session() as s:
        s.add(db.Analysis(id=aid, user_id=user.id, status="queued", train_ok=bool(train_ok)))
        s.commit()
    jobs.submit(aid)
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


def render_analysis(request, row, admin_view=None):
    """The page for one analysis in whatever state it's in (also used by /admin)."""
    aid = row.id
    if row.status in ("queued", "running"):
        return page(request, "processing.html", {"aid": aid, "status": row.status, "admin_view": admin_view})
    if row.status == "rejected":
        return page(request, "rejected.html", {"aid": aid, "problems": json.loads(row.problems or "{}"),
                                               "admin_view": admin_view})
    if row.status == "failed":
        return page(request, "failed.html", {"admin_view": admin_view}, 200 if admin_view else 500)
    if row.status == "waiting" or (row.status == "review" and not admin_view):
        return page(request, "held.html", {"status": row.status, "admin_view": admin_view})
    record = pipeline.load(aid)
    if record is None:
        return page(request, "missing.html", {}, 404)
    view = presenter.build(record, rules())
    since = None
    previous = auth.previous_report(row)
    if previous is not None and (old := pipeline.load(previous.id)) is not None:
        since = presenter.since_last(view, presenter.build(old, rules()))
    return page(request, "report.html", {"r": view, "coach_note": row.coach_note, "since": since,
                                         "feedback": feedback.for_analysis(aid), "admin_view": admin_view,
                                         "saved": request.query_params.get("saved")})


admin.render_analysis = render_analysis


@app.get("/sample", response_class=HTMLResponse)
def sample_report(request: Request):
    """What a report looks like, for people deciding whether to sign up. Public."""
    return page(request, "report.html", {"r": presenter.build(pipeline.sample_record(), rules()), "sample": True,
                                         "coach_note": "", "feedback": {}, "since": None, "admin_view": None})


@app.get("/analyses/{aid}", response_class=HTMLResponse)
def report_page(request: Request, aid: str):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    row = auth.owned_analysis(user, aid)
    if row is None:
        return page(request, "missing.html", {}, 404)
    return render_analysis(request, row)


@app.get("/analyses/{aid}/tracked/{slot}.jpg")
def tracked_frame(request: Request, aid: str, slot: str):
    """A still from the swimmer's own clip with the tracked points drawn on. Private:
    the owner and admins only, like the report it belongs to."""
    user = request.state.user
    row = auth.owned_analysis(user, aid)
    if row is None and beta.is_admin(user):
        with db.session() as s:
            row = s.get(db.Analysis, aid)
    path = None
    if row is not None and slot in pipeline.CLIPS:
        path = storage.fetch(pipeline.folder(aid), aid, tracked.name(slot))
    if path is None:
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/analyses/{aid}/status")
def report_status(request: Request, aid: str):
    row = auth.owned_analysis(request.state.user, aid)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"status": row.status}


@app.post("/analyses/{aid}/clips")
async def replace_clips(request: Request, aid: str, side: UploadFile | None = None, head_on: UploadFile | None = None,
                        side_uw: UploadFile | None = None, head_on_uw: UploadFile | None = None):
    """Fix an upload that was turned back: send only the clips that need replacing."""
    user = request.state.user
    row = auth.owned_analysis(user, aid)
    if row is None or row.status != "rejected":
        return page(request, "missing.html", {}, 404)
    files = {slot: (up.filename, up.file) for slot, up in
             (("side", side), ("head_on", head_on), ("side_uw", side_uw), ("head_on_uw", head_on_uw))
             if up is not None and up.filename}
    was = json.loads(row.problems or "{}")
    if not files:
        return page(request, "rejected.html", {"aid": aid, "problems": was, "nothing_added": True}, 422)
    try:
        pipeline.replace_clips(aid, files)
    except pipeline.UploadProblem as e:
        return page(request, "rejected.html", {"aid": aid, "problems": dict(was, **e.problems)}, 422)
    with db.session() as s:
        s.get(db.Analysis, aid).status = "queued"
        s.commit()
    jobs.submit(aid)
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


@app.post("/analyses/{aid}/feedback")
def report_feedback(request: Request, aid: str, issue_id: str = Form(""), verdict: str = Form(""),
                    comment: str = Form("")):
    user = request.state.user
    row = auth.owned_analysis(user, aid)
    if row is None or row.status != "done":
        return page(request, "missing.html", {}, 404)
    feedback.save(user, aid, issue_id, verdict, comment)
    if "application/json" in request.headers.get("accept", ""):
        return {"ok": True}
    anchor = f"fb-{issue_id}" if issue_id else "fb-overall"
    return RedirectResponse(f"/analyses/{aid}?saved=1#{anchor}", status_code=303)


# ------------------------------------------------------------------ plan and payments
@app.get("/billing", response_class=HTMLResponse)
def billing_page(request: Request, done: str = ""):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    if not billing.visible():
        return page(request, "missing.html", {}, 404)
    return page(request, "billing.html", {"plan": billing.status(user), "done": bool(done), "error": None})


def _to_stripe(request, make_url):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    if not billing.configured():
        return page(request, "missing.html", {}, 404)
    try:
        return RedirectResponse(make_url(user, base_url(request)), status_code=303)
    except billing.BillingError as e:
        return page(request, "billing.html", {"plan": billing.status(user), "done": False, "error": str(e)}, 502)


@app.post("/billing/checkout")
def billing_checkout(request: Request):
    return _to_stripe(request, billing.checkout_url)


@app.post("/billing/portal")
def billing_portal(request: Request):
    return _to_stripe(request, billing.portal_url)


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    try:
        event = billing.verify(await request.body(), request.headers.get("stripe-signature", ""))
    except billing.BillingError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"result": billing.handle(event)}


@app.get("/guide", response_class=HTMLResponse)
def guide_page(request: Request):
    groups = [g for g in presenter.coverage(rules()) if g["issues"]]
    return page(request, "guide.html", {"coverage": groups})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import FileResponse
    return FileResponse(HERE / "static" / "favicon.png", media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})


@app.get("/robots.txt", include_in_schema=False)
def robots():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("User-agent: *\nAllow: /$\nAllow: /sample\nAllow: /guide\nAllow: /terms\nAllow: /privacy\nDisallow: /\n")


@app.get("/healthz")
def healthz():
    from sqlalchemy import text
    try:
        with db.session() as s:
            s.execute(text("SELECT 1"))
    except Exception:   # noqa: BLE001
        return JSONResponse({"ok": False, "database": "unreachable"}, status_code=503)
    return {"ok": True}
