"""Things every route module shares: templates, page rendering, the rules workbook."""
import os
from pathlib import Path

from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from swimform import rules as R
from swimform.analyze import DEFAULT_RULES

from . import beta, billing, google

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")
_rules_cache = {}


def rules():
    if "r" not in _rules_cache:
        r, errors = R.load(DEFAULT_RULES)
        if errors:
            raise RuntimeError("rules workbook invalid: " + "; ".join(errors))
        _rules_cache["r"] = r
    return _rules_cache["r"]


def company():
    return {"name": os.environ.get("SWIMFORM_COMPANY", "SwimForm"),
            "contact": os.environ.get("SWIMFORM_CONTACT_EMAIL", ""),
            "law": os.environ.get("SWIMFORM_GOVERNING_LAW", "the Commonwealth of Virginia, USA")}


def page(request, name, ctx=None, status_code=200):
    ctx = dict(ctx or {})
    user = request.state.user
    ctx.setdefault("user", user)
    ctx.setdefault("google_enabled", google.configured())
    ctx.setdefault("is_admin", beta.is_admin(user))
    ctx.setdefault("billing_enabled", billing.visible())
    ctx.setdefault("company", company())
    ctx.setdefault("signup_mode", beta.mode())
    ctx.setdefault("share_image", base_url(request) + "/static/share.png")
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def login_redirect(request):
    nxt = request.url.path
    return RedirectResponse(f"/login?next={nxt}" if nxt not in ("/", "") else "/login", status_code=303)


def safe_next(nxt):
    return nxt if nxt and nxt.startswith("/") and not nxt.startswith("//") else "/"


def base_url(request=None):
    """The public address, for links in emails and redirects handed to other services."""
    env = os.environ.get("SWIMFORM_BASE_URL")
    if env:
        return env.rstrip("/")
    return str(request.base_url).rstrip("/") if request is not None else "http://localhost:8000"
