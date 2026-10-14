"""
Turns a saved analysis into what the report page shows, in the athlete's language.

Every one of the workbook's issues ends up in exactly one place:
  needs work        flagged from a direct measurement
  may need work     flagged from an estimate (proxy), worded as such
  looks good        checked and fine (noting when the check was only an estimate)
  couldn't check    with the reason and what footage would fix it
"""
from swimform.schema import BOTH, HEAD_ON, SIDE

STATUS = {
    "flagged":   "Needs work",
    "estimated": "May need work",
    "clear":     "Looks good",
    "clear_est": "Looks good (estimated)",
    "unchecked": "Couldn't check",
}
ORDER = ["flagged", "estimated", "unchecked", "clear_est", "clear"]

UNITS = {"deg": "°", "ratio": "", "cycle_frac": "", "cyc/min": " strokes/min"}


def _advice(reason, below_water):
    r = reason.lower()
    if "no side video" in r or "no head_on video" in r or "needs both" in r:
        return "Add both clips to check this."
    if "no breaths" in r:
        return "Breathe as you normally would during the clip, and we can check this."
    if "do not line up" in r:
        return "Film both clips at the same steady pace, and we can check this."
    if "could not be computed" in r:
        return "Something went wrong measuring this. Try the clips again, or a different pair."
    if below_water:
        return "This happens underwater. An underwater side view would let us check it."
    return "Film closer, with the whole swimmer in frame, and we can check this."


def _sentence(text):
    return text[:1].upper() + text[1:]


def _fmt(value, unit):
    if unit == "cycle_frac":
        return f"{round(100 * value)}% of each stroke"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "deg":
        return f"{round(value)}°"
    return f"{round(value, 1)}{UNITS.get(unit, '')}"


def build(record, rules):
    rep = record["report"]
    metrics = {m["metric_id"]: m for m in rules["metrics"]}
    issues = {i["issue_id"]: i for i in rules["issues"]}
    issue_metrics = {}
    for r in rules["issue_rules"]:
        issue_metrics.setdefault(r["issue_id"], set()).add(r["metric_id"])

    status = {}
    for p in rep["priorities"]:
        status[p["issue_id"]] = "estimated" if p["evidence"] == "proxy" else "flagged"
    for o in rep["other_issues"]:
        status[o["issue_id"]] = "estimated" if o["evidence"] == "proxy" else "flagged"
    for iid, ev in rep["issues_clear"].items():
        status[iid] = "clear_est" if ev == "proxy" else "clear"
    for iid in rep["issues_not_assessed"]:
        status[iid] = "unchecked"

    lanes = sorted(
        [{"issue_id": iid, "name": issues[iid]["name"], "status": st, "label": STATUS[st]}
         for iid, st in status.items() if iid in issues],
        key=lambda l: (ORDER.index(l["status"]), l["name"]))

    priorities = []
    for p in rep["priorities"]:
        priorities.append({
            "name": p["name"], "estimated": p["evidence"] == "proxy",
            "observed": p["what_we_observed"], "why": p["why_it_matters"],
            "drills": [d for d in p["interventions"] if d["type"] == "drill"],
            "cues": [d for d in p["interventions"] if d["type"] == "cue"],
            "exercises": [d for d in p["interventions"] if d["type"] == "exercise"],
        })
    shown = {p["issue_id"] for p in rep["priorities"]}
    also = [{"name": issues[o["issue_id"]]["name"], "estimated": o["evidence"] == "proxy",
             "observed": (issues[o["issue_id"]].get("proxy_observed") if o["evidence"] == "proxy"
                          else issues[o["issue_id"]]["what_we_observed"])}
            for o in rep["other_issues"] if o["issue_id"] not in shown]

    unchecked = []
    for iid, reason in rep["issues_not_assessed"].items():
        below = any(metrics[m]["water"] == "below" for m in issue_metrics.get(iid, ()) if m in metrics)
        unchecked.append({"name": issues[iid]["name"], "advice": _advice(reason, below)})
    unchecked.sort(key=lambda u: u["name"])

    good = sorted([{"name": issues[iid]["name"], "estimated": ev == "proxy"}
                   for iid, ev in rep["issues_clear"].items()], key=lambda g: (g["estimated"], g["name"]))

    view_label = {SIDE: "Side view", HEAD_ON: "Head-on view", BOTH: "3D, from both clips"}
    table = []
    for mid, m in rep["measurements"].items():
        meta = metrics.get(mid, {})
        table.append({
            "name": _sentence(meta.get("name", mid).replace("Proxy: ", "")),
            "value": _fmt(m["value"], meta.get("unit")),
            "score": m.get("score"),
            "source": view_label.get(m["view"], m["view"]),
            "estimate": m["method"].startswith("proxy"),
            "confidence": round(100 * m["confidence"]),
        })
    table.sort(key=lambda t: (t["estimate"], t["source"], t["name"]))

    counts = {k: sum(1 for l in lanes if l["status"] == k) for k in STATUS}
    rate = rep["measurements"].get("stroke_rate", {}).get("value")
    s3 = rep.get("stroke3d") or {}
    return {
        "id": record["id"], "created": record["created"][:10], "lanes": lanes, "counts": counts,
        "priorities": priorities, "also": also, "good": good, "unchecked": unchecked,
        "table": table, "stroke_rate": round(rate) if rate else None,
        "fused": bool(s3.get("used")), "fused_agreement": s3.get("view_agreement"),
        "synthetic": record.get("pose_provider") == "synthetic",
    }
