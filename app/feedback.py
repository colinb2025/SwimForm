"""
What swimmers say about their reports. During the beta this is the accuracy signal:
for each finding, did it match what they or their coach see?
"""
from sqlalchemy import select

from . import db

VERDICTS = ("right", "wrong", "unsure")


def save(user, aid, issue_id, verdict, comment=""):
    """Record (or change) one answer. issue_id "" = the whole report, verdict "1".."5"."""
    issue_id = issue_id.strip()[:64]
    ok = verdict in VERDICTS if issue_id else verdict in ("", "1", "2", "3", "4", "5")
    if not ok:
        return False
    with db.session() as s:
        row = s.scalar(select(db.Feedback).where(db.Feedback.analysis_id == aid, db.Feedback.issue_id == issue_id))
        if row is None:
            row = db.Feedback(analysis_id=aid, user_id=user.id, issue_id=issue_id)
            s.add(row)
        row.verdict, row.comment, row.created = verdict, comment.strip()[:4000], db.now()
        s.commit()
    return True


def for_analysis(aid):
    """{issue_id: {"verdict":, "comment":}} for one report."""
    with db.session() as s:
        rows = s.scalars(select(db.Feedback).where(db.Feedback.analysis_id == aid))
        return {r.issue_id: {"verdict": r.verdict, "comment": r.comment} for r in rows}


def everything():
    with db.session() as s:
        q = (select(db.Feedback, db.User.email).join(db.User, db.User.id == db.Feedback.user_id)
             .order_by(db.Feedback.created.desc()))
        return [{"analysis_id": f.analysis_id, "email": email, "issue_id": f.issue_id, "verdict": f.verdict,
                 "comment": f.comment, "created": f.created} for f, email in s.execute(q)]


def by_issue(rows):
    """Per finding: how often swimmers agreed. -> [{issue_id, right, wrong, unsure, agree_pct}]"""
    out = {}
    for r in rows:
        if r["issue_id"] and r["verdict"] in VERDICTS:
            out.setdefault(r["issue_id"], dict.fromkeys(VERDICTS, 0))[r["verdict"]] += 1
    table = []
    for iid, c in out.items():
        judged = c["right"] + c["wrong"]
        table.append(dict(c, issue_id=iid, agree_pct=round(100 * c["right"] / judged) if judged else None))
    return sorted(table, key=lambda t: (t["agree_pct"] is None, t["agree_pct"] or 0))
