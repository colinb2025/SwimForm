"""
The beta console at /admin, for the accounts listed in SWIMFORM_ADMINS.

  /admin            what's happening and what's connected
  /admin/analyses   every analysis; open one, release a held report, re-run it
  /admin/users      accounts; give or remove the pro plan
  /admin/invites    invite codes and the waitlist
  /admin/feedback   what swimmers said about their findings (also as CSV)

Everyone else gets "not found", so the console's existence isn't advertised.
Admins can open any swimmer's report: the privacy page says so.
"""
import csv
import io

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import func, select

from . import beta, connections, db, feedback, jobs, mailer, notify, pipeline
from .web import page, rules

router = APIRouter(prefix="/admin")
render_analysis = None   # set by app.main (it owns the report page)

STATUSES = ("review", "waiting", "queued", "running", "done", "rejected", "failed")
PAGE = 200


def _admin(request):
    user = request.state.user
    return user if beta.is_admin(user) else None


def _missing(request):
    return page(request, "missing.html", {}, 404)


def _back(path, note=""):
    return RedirectResponse(f"{path}{'?note=' + note if note else ''}", status_code=303)


NOTES = {"released": "Report released. The swimmer has been emailed.",
         "rerun": "Queued to run again.", "norerun": "Couldn't re-run: the clips are gone, or the pose model is still set to collect.",
         "waiting": "Waiting analyses queued.", "invite": "Invite created.", "sent": "Invite emailed.",
         "plan": "Plan changed.", "mail": "Test email sent. Check your inbox.",
         "mailfail": "The test email failed to send. Check the email settings and the server log."}


def _ctx(request, tab, **kw):
    with db.session() as s:
        review = s.scalar(select(func.count()).select_from(db.Analysis).where(db.Analysis.status == "review")) or 0
    return dict(kw, tab=tab, review_count=review, note=NOTES.get(request.query_params.get("note", "")))


def _iso(dt):
    return (dt if dt.tzinfo else dt.replace(tzinfo=db.now().tzinfo)).isoformat()


# ------------------------------------------------------------------ overview
@router.get("", response_class=HTMLResponse)
def overview(request: Request):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        by_status = dict(s.execute(select(db.Analysis.status, func.count()).group_by(db.Analysis.status)).all())
        stats = {
            "users": s.scalar(select(func.count()).select_from(db.User)) or 0,
            "pro": s.scalar(select(func.count()).select_from(db.User).where(db.User.plan == "pro")) or 0,
            "analyses": sum(by_status.values()),
            "waitlist": s.scalar(select(func.count()).select_from(db.Waitlist).where(db.Waitlist.invited.is_(None))) or 0,
            "consented": s.scalar(select(func.count()).select_from(db.Analysis).where(db.Analysis.train_ok.is_(True))) or 0,
        }
    rows = feedback.everything()
    judged = [r for r in rows if r["issue_id"] and r["verdict"] in ("right", "wrong")]
    stats["feedback"] = len(rows)
    stats["agree_pct"] = round(100 * sum(r["verdict"] == "right" for r in judged) / len(judged)) if judged else None
    items = connections.check()
    return page(request, "admin/overview.html", _ctx(
        request, "overview", stats=stats, by_status=[(st, by_status.get(st, 0)) for st in STATUSES],
        connections=items, ready=connections.ready(items)))


@router.post("/test-email")
def test_email(request: Request):
    user = _admin(request)
    if not user:
        return _missing(request)
    try:
        mailer.send(user.email, "SwimForm test email", "If you're reading this, SwimForm can send email.\n")
    except Exception:   # noqa: BLE001 - the point of the button is to find this out
        return _back("/admin", "mailfail")
    return _back("/admin", "mail")


# ------------------------------------------------------------------ analyses
@router.get("/analyses", response_class=HTMLResponse)
def analyses(request: Request, status: str = ""):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        q = select(db.Analysis, db.User.email).join(db.User, db.User.id == db.Analysis.user_id)
        if status in STATUSES:
            q = q.where(db.Analysis.status == status)
        rows = s.execute(q.order_by(db.Analysis.created.desc()).limit(PAGE)).all()
    items = [{"id": a.id, "email": email, "iso": _iso(a.created), "status": a.status, "flagged": a.flagged,
              "estimated": a.estimated, "unchecked": a.unchecked, "train_ok": a.train_ok,
              "problem": a.problems if a.status == "failed" else ""} for a, email in rows]
    return page(request, "admin/analyses.html", _ctx(request, "analyses", items=items, status=status, statuses=STATUSES))


@router.get("/analyses/{aid}", response_class=HTMLResponse)
def analysis(request: Request, aid: str):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        row = s.get(db.Analysis, aid)
        owner = s.get(db.User, row.user_id) if row else None
    if row is None:
        return _missing(request)
    issues = {i["issue_id"]: i["name"] for i in rules()["issues"]}
    fb = [{"what": issues.get(k, k) if k else "Whole report", "verdict": v["verdict"], "comment": v["comment"]}
          for k, v in feedback.for_analysis(aid).items()]
    up = pipeline.upload_info(aid) or {}
    return render_analysis(request, row, admin_view={
        "aid": aid, "email": owner.email if owner else "(deleted)", "status": row.status, "coach_note": row.coach_note,
        "problem": row.problems if row.status == "failed" else "", "feedback": fb, "train_ok": row.train_ok,
        "files": up.get("files", {}), "provider": up.get("provider", ""), "note": NOTES.get(request.query_params.get("note", ""))})


@router.post("/analyses/{aid}/release")
def release(request: Request, aid: str, coach_note: str = Form("")):
    if not _admin(request):
        return _missing(request)
    jobs.release(aid, coach_note)
    return _back("/admin/analyses", "released")


@router.post("/analyses/{aid}/rerun")
def rerun(request: Request, aid: str):
    if not _admin(request):
        return _missing(request)
    return _back(f"/admin/analyses/{aid}", "rerun" if jobs.rerun(aid) else "norerun")


@router.post("/rerun-waiting")
def rerun_waiting(request: Request):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        waiting = list(s.scalars(select(db.Analysis.id).where(db.Analysis.status == "waiting")))
    queued = sum(1 for aid in waiting if jobs.rerun(aid))
    return _back("/admin/analyses", "waiting" if queued else "norerun")


# ------------------------------------------------------------------ users
@router.get("/users", response_class=HTMLResponse)
def users(request: Request):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        counts = dict(s.execute(select(db.Analysis.user_id, func.count()).group_by(db.Analysis.user_id)).all())
        rows = list(s.scalars(select(db.User).order_by(db.User.created.desc()).limit(PAGE)))
    items = [{"id": u.id, "email": u.email, "name": u.name, "iso": _iso(u.created), "plan": u.plan,
              "analyses": counts.get(u.id, 0), "invite": u.invite_code or "", "google": bool(u.google_sub),
              "paying": bool(u.stripe_subscription), "admin": u.email in beta.admins()} for u in rows]
    return page(request, "admin/users.html", _ctx(request, "users", items=items))


@router.post("/users/{uid}/plan")
def set_plan(request: Request, uid: int, plan: str = Form("")):
    if not _admin(request):
        return _missing(request)
    if plan in ("free", "pro"):
        with db.session() as s:
            u = s.get(db.User, uid)
            if u is not None:
                u.plan = plan
                s.commit()
    return _back("/admin/users", "plan")


# ------------------------------------------------------------------ invites and waitlist
@router.get("/invites", response_class=HTMLResponse)
def invites(request: Request):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        codes = list(s.scalars(select(db.Invite).order_by(db.Invite.created.desc()).limit(PAGE)))
        waiting = list(s.scalars(select(db.Waitlist).order_by(db.Waitlist.invited.is_not(None), db.Waitlist.created).limit(PAGE)))
    return page(request, "admin/invites.html", _ctx(
        request, "invites", mode=beta.mode(),
        codes=[{"code": c.code, "note": c.note, "uses": c.uses, "max_uses": c.max_uses, "iso": _iso(c.created)} for c in codes],
        waitlist=[{"id": w.id, "email": w.email, "note": w.note, "iso": _iso(w.created), "invited": bool(w.invited)} for w in waiting]))


@router.post("/invites")
def create_invite(request: Request, note: str = Form(""), max_uses: str = Form("1")):
    if not _admin(request):
        return _missing(request)
    beta.new_invite(note, int(max_uses) if max_uses.isdigit() else 1)
    return _back("/admin/invites", "invite")


@router.post("/waitlist/{wid}/invite")
def invite_from_waitlist(request: Request, wid: int):
    if not _admin(request):
        return _missing(request)
    with db.session() as s:
        w = s.get(db.Waitlist, wid)
        if w is None:
            return _missing(request)
        email = w.email
        w.invited = db.now()
        s.commit()
    notify.invite(email, beta.new_invite(f"waitlist: {email}", 1))
    return _back("/admin/invites", "sent")


# ------------------------------------------------------------------ feedback
def _feedback_rows():
    issues = {i["issue_id"]: i["name"] for i in rules()["issues"]}
    rows = feedback.everything()
    for r in rows:
        r["what"] = issues.get(r["issue_id"], r["issue_id"]) if r["issue_id"] else "Whole report"
        r["iso"] = _iso(r["created"])
    return rows, issues


@router.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    if not _admin(request):
        return _missing(request)
    rows, issues = _feedback_rows()
    table = feedback.by_issue(rows)
    for t in table:
        t["name"] = issues.get(t["issue_id"], t["issue_id"])
    ratings = [int(r["verdict"]) for r in rows if not r["issue_id"] and r["verdict"].isdigit()]
    return page(request, "admin/feedback.html", _ctx(
        request, "feedback", rows=rows[:PAGE], table=table,
        rating=round(sum(ratings) / len(ratings), 1) if ratings else None, n_ratings=len(ratings)))


def _cell(v):
    """Text typed by users, made safe to open in a spreadsheet (no formulas)."""
    v = str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v


@router.get("/feedback.csv")
def feedback_csv(request: Request):
    if not _admin(request):
        return _missing(request)
    rows, _ = _feedback_rows()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["when", "analysis", "email", "finding", "issue_id", "verdict", "comment"])
    for r in rows:
        w.writerow([r["iso"], r["analysis_id"], _cell(r["email"]), r["what"], r["issue_id"], r["verdict"], _cell(r["comment"])])
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="swimform-feedback.csv"'})
