"""
SwimForm web app.

  uvicorn app.main:app --reload

Public:   GET /login, /signup, /forgot, /reset, /guide, /auth/google/*
          POST /login, /signup, /forgot, /reset
Account:  GET / -> your reports, /analyses/new, /analyses/{id}
          POST /analyses, /logout
A report is visible only to the account that made it; anyone else gets "not found".
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
import os

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from swimform import rules as R
from swimform.analyze import DEFAULT_RULES

from . import auth, db, google, mailer, pipeline, presenter

HERE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app):
    db.init()
    yield


app = FastAPI(title="SwimForm", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

_rules_cache = {}


def rules():
    if "r" not in _rules_cache:
        r, errors = R.load(DEFAULT_RULES)
        if errors:
            raise RuntimeError("rules workbook invalid: " + "; ".join(errors))
        _rules_cache["r"] = r
    return _rules_cache["r"]


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and not auth.same_origin(request):
        return HTMLResponse("Request refused: it came from another site.", status_code=403)
    request.state.user = auth.current_user(request)
    return await call_next(request)


def page(request, name, ctx=None, status_code=200):
    ctx = dict(ctx or {})
    ctx.setdefault("user", request.state.user)
    ctx.setdefault("google_enabled", google.configured())
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def login_redirect(request):
    nxt = request.url.path
    return RedirectResponse(f"/login?next={nxt}" if nxt not in ("/", "") else "/login", status_code=303)


def _safe_next(nxt):
    return nxt if nxt and nxt.startswith("/") and not nxt.startswith("//") else "/"


# ------------------------------------------------------------------ accounts
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if request.state.user:
        return RedirectResponse(_safe_next(next), status_code=303)
    return page(request, "login.html", {"next": _safe_next(next), "email": "", "error": None})


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


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    if request.state.user:
        return RedirectResponse("/", status_code=303)
    return page(request, "signup.html", {"name": "", "email": "", "errors": {}})


@app.post("/signup")
def signup(request: Request, name: str = Form(""), email: str = Form(""), password: str = Form("")):
    email_n = email.strip().lower()
    errors = {}
    if "@" not in email_n or "." not in email_n.split("@")[-1] or len(email_n) > 320:
        errors["email"] = "Enter an email address like name@example.com."
    if problem := auth.password_problem(password):
        errors["password"] = problem
    with db.session() as s:
        if not errors and db.user_by_email(s, email_n):
            errors["email"] = "There's already an account with this email. Log in instead."
        if errors:
            return page(request, "signup.html", {"name": name, "email": email, "errors": errors}, 422)
        user = db.User(email=email_n, name=name.strip()[:120], password_hash=auth.hash_password(password))
        s.add(user)
        s.commit()
        uid = user.id
    resp = RedirectResponse("/analyses/new", status_code=303)
    auth.set_cookie(resp, auth.start_session(uid))
    return resp


# ------------------------------------------------------------------ password reset
def _base_url(request):
    return (os.environ.get("SWIMFORM_BASE_URL") or str(request.base_url)).rstrip("/")


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
        link = f"{_base_url(request)}/reset?token={auth.create_reset(user.id)}"
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
def google_start(request: Request, next: str = "/"):
    if not google.configured():
        return page(request, "missing.html", {}, 404)
    return RedirectResponse(google.start(request, _safe_next(next)), status_code=303)


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
              "flagged": a.flagged, "estimated": a.estimated, "unchecked": a.unchecked} for a in rows]
    return page(request, "reports.html", {"items": items})


@app.get("/analyses/new", response_class=HTMLResponse)
def upload_page(request: Request):
    if request.state.user is None:
        return login_redirect(request)
    return page(request, "upload.html", {"problems": {}})


@app.post("/analyses")
async def create_analysis(request: Request, side: UploadFile | None = None, head_on: UploadFile | None = None):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    files = {}
    for slot, up in (("side", side), ("head_on", head_on)):
        if up is not None and up.filename:
            files[slot] = (up.filename, up.file)
    try:
        aid = pipeline.run(files)
    except pipeline.UploadProblem as e:
        return page(request, "upload.html", {"problems": e.problems}, 422)
    view = presenter.build(pipeline.load(aid), rules())
    with db.session() as s:
        s.add(db.Analysis(id=aid, user_id=user.id, lanes=",".join(l["status"] for l in view["lanes"]),
                          flagged=view["counts"]["flagged"], estimated=view["counts"]["estimated"],
                          unchecked=view["counts"]["unchecked"]))
        s.commit()
    return RedirectResponse(f"/analyses/{aid}", status_code=303)


@app.get("/analyses/{aid}", response_class=HTMLResponse)
def report_page(request: Request, aid: str):
    user = request.state.user
    if user is None:
        return login_redirect(request)
    record = pipeline.load(aid) if auth.owned_analysis(user, aid) else None
    if record is None:
        return page(request, "missing.html", {}, 404)
    return page(request, "report.html", {"r": presenter.build(record, rules())})


@app.get("/guide", response_class=HTMLResponse)
def guide_page(request: Request):
    groups = [g for g in presenter.coverage(rules()) if g["issues"]]
    return page(request, "guide.html", {"coverage": groups})


@app.get("/healthz")
def healthz():
    return {"ok": True}
