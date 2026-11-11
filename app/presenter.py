"""
Turns a saved analysis into what the report page shows, in the athlete's language.

Every one of the workbook's issues ends up in exactly one place:
  needs work        flagged from a direct measurement
  may need work     flagged from an estimate (proxy), worded as such
  looks good        checked and fine (noting when the check was only an estimate)
  couldn't check    with the reason and what footage would fix it
"""
from swimform.schema import BOTH, HEAD_ON, SIDE

from . import snapshots

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


SIDE_NAME = {"r": "right", "l": "left"}
OTHER = {"r": "l", "l": "r"}


def _arm_note(arms):
    """Which arm (or leg) a finding was seen on, and whether the other one was checked."""
    if not arms or not arms.get("checked"):
        return None
    limb, checked, affected = arms["limb"], arms["checked"], arms["affected"]
    if len(checked) == 2:
        if len(affected) == 1:
            sd = affected[0]
            return f"Seen on your {SIDE_NAME[sd]} {limb} only. Your {SIDE_NAME[OTHER[sd]]} {limb} looks fine."
        if len(affected) == 2:
            return f"Seen on both {limb}s."
        return None
    sd = checked[0]
    if affected:
        return (f"Seen on your {SIDE_NAME[sd]} {limb}. Your {SIDE_NAME[OTHER[sd]]} {limb} didn't face the camera, "
                f"so we couldn't check it: film a 50, down the pool and back, and we'll see both.")
    return None


def _checked_one_side(arms):
    """'right arm only' when an arm-specific check could only look at one side."""
    if arms and len(arms.get("checked", [])) == 1:
        return f"{SIDE_NAME[arms['checked'][0]]} {arms['limb']} only"
    return None


def _sentence(text):
    return text[:1].upper() + text[1:]


def _fmt(value, unit):
    if unit == "cycle_frac":
        return f"{round(100 * value)}% of each stroke"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "deg":
        return f"{round(value)}°"
    if unit == "s":
        return f"{value:.2f} s"
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
        [{"issue_id": iid, "name": issues[iid]["name"], "status": st, "label": STATUS[st],
          # What was checked, in neutral words ("Hand entry line"), for the rows that aren't faults.
          "check": issues[iid].get("check_name") or issues[iid]["name"]}
         for iid, st in status.items() if iid in issues],
        key=lambda l: (ORDER.index(l["status"]), l["name"]))

    def caption(iid, evidence):
        """The measurement behind a finding, and the threshold the workbook flags at."""
        for r in rules["issue_rules"]:
            if r["issue_id"] != iid or (r.get("evidence") or "direct") != evidence:
                continue
            m = rep["measurements"].get(r["metric_id"])
            meta = metrics.get(r["metric_id"])
            if not m or not meta:
                continue
            unit = meta.get("unit")
            op = {"<": "below", "<=": "at or below", ">": "above", ">=": "at or above"}.get(r["operator"])
            limit = f" We flag {op} {_fmt(r['value'], unit)}." if op else ""
            return f"{_sentence(meta['name'].replace('Proxy: ', ''))}: {_fmt(m['value'], unit)}.{limit}"
        return None

    strokes = rep.get("average_strokes") or {}
    issue_arms = rep.get("issue_arms") or {}
    priorities = []
    for p in rep["priorities"]:
        arms = p.get("arms") or issue_arms.get(p["issue_id"])
        one_arm = arms["affected"][0] if arms and len(arms["affected"]) == 1 else None
        snap = snapshots.svg(strokes, p["issue_id"], arm=one_arm) if strokes else None
        priorities.append({
            "snapshot": snap[0] if snap else None,
            "snapshot_moment": snap[1] if snap else None,
            "snapshot_view": snapshots.SPECS[p["issue_id"]][0] if snap else None,
            "caption": caption(p["issue_id"], p["evidence"]),
            "id": p["issue_id"],
            "name": p["name"], "estimated": p["evidence"] == "proxy", "arm_note": _arm_note(arms),
            "observed": p["what_we_observed"], "why": p["why_it_matters"],
            "drills": [d for d in p["interventions"] if d["type"] == "drill"],
            "cues": [d for d in p["interventions"] if d["type"] == "cue"],
            "exercises": [d for d in p["interventions"] if d["type"] == "exercise"],
        })
    shown = {p["issue_id"] for p in rep["priorities"]}
    also = [{"name": issues[o["issue_id"]]["name"], "estimated": o["evidence"] == "proxy",
             "arm_note": _arm_note(o.get("arms") or issue_arms.get(o["issue_id"])),
             "observed": (issues[o["issue_id"]].get("proxy_observed") if o["evidence"] == "proxy"
                          else issues[o["issue_id"]]["what_we_observed"])}
            for o in rep["other_issues"] if o["issue_id"] not in shown]

    unchecked = []
    for iid, reason in rep["issues_not_assessed"].items():
        below = any(metrics[m]["water"] == "below" for m in issue_metrics.get(iid, ()) if m in metrics)
        unchecked.append({"name": issues[iid]["name"], "advice": _advice(reason, below)})
    unchecked.sort(key=lambda u: u["name"])

    good = sorted([{"name": issues[iid]["name"], "estimated": ev == "proxy",
                    "one_side": _checked_one_side(issue_arms.get(iid))}
                   for iid, ev in rep["issues_clear"].items()], key=lambda g: (g["estimated"], g["name"]))
    one_side = {g["name"]: g["one_side"] for g in good if g["one_side"]}

    view_label = {SIDE: "Side view", HEAD_ON: "Head-on view", BOTH: "3D, from both clips"}
    table = []
    for mid, m in rep["measurements"].items():
        meta = metrics.get(mid, {})
        arms = m.get("arms") or {}
        per_arm = None
        if len(arms) == 2:
            per_arm = f"right {_fmt(arms['r']['value'], meta.get('unit'))}, left {_fmt(arms['l']['value'], meta.get('unit'))}"
        elif len(arms) == 1:
            per_arm = f"{SIDE_NAME[next(iter(arms))]} {m.get('limb', 'arm')} only"
        table.append({
            "name": _sentence(meta.get("name", mid).replace("Proxy: ", "")),
            "value": _fmt(m["value"], meta.get("unit")),
            "per_arm": per_arm,
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
        "id": record["id"], "created": record["created"][:10], "created_iso": record["created"],
        "lanes": lanes, "counts": counts, "one_side": one_side,
        "both_sides": any(len(a.get("checked", [])) == 2 for a in issue_arms.values()),
        "priorities": priorities, "also": also, "good": good, "unchecked": unchecked,
        "table": table, "stroke_rate": round(rate) if rate else None,
        "fused": bool(s3.get("used")), "fused_agreement": s3.get("view_agreement"),
        "synthetic": record.get("pose_provider") == "synthetic",
        "unused_clips": unused_clips(record),
        "headline": ", ".join(p["name"] for p in priorities),
    }


FLAGGED = ("flagged", "estimated")
CLEAR = ("clear", "clear_est")


def since_last(view, previous):
    """What changed between the previous report and this one. Marks each finding on
    `view` as new or repeated and returns the summary line's parts."""
    before = {l["issue_id"]: l["status"] for l in previous["lanes"]}
    fixed = [l["check"] for l in view["lanes"] if l["status"] in CLEAR and before.get(l["issue_id"]) in FLAGGED]
    new, still = [], 0
    for l in view["lanes"]:
        if l["status"] not in FLAGGED:
            continue
        if before.get(l["issue_id"]) in FLAGGED:
            still += 1
        elif before.get(l["issue_id"]) in CLEAR:
            new.append(l["name"])
    for p in view["priorities"]:
        was = before.get(p["id"])
        p["since"] = "repeat" if was in FLAGGED else "new" if was in CLEAR else None
    return {"iso": previous["created_iso"], "date": previous["created"], "id": previous["id"],
            "fixed": fixed, "new": new, "still": still}



CLIP_NAMES = {("side", "above"): "side-view", ("side", "underwater"): "underwater side-view",
              ("head_on", "above"): "head-on", ("head_on", "underwater"): "underwater head-on"}


def unused_clips(record):
    """Clips that failed their check and were left out of the analysis, with why."""
    return [{"name": CLIP_NAMES[(c["view"], c["water"])], "problems": c["problems"]}
            for c in record.get("clips") or [] if not c["used"]]


def coverage(rules):
    """Which clip lets us check which issue, from the workbook's direct rules, for the guide."""
    metrics = {m["metric_id"]: m for m in rules["metrics"]}
    groups = {}
    for r in rules["issue_rules"]:
        key = (r["issue_id"], r.get("rule_group") or 1)
        groups.setdefault(key, {"rows": [], "proxy": False})
        groups[key]["rows"].append(r["metric_id"])
        groups[key]["proxy"] |= r.get("evidence") == "proxy"
    names = {i["issue_id"]: i.get("check_name") or i["name"] for i in rules["issues"]}
    out = {"side": [], "head_on": [], "both": [], "underwater": [], "underwater_only": []}
    for iid, name in names.items():
        direct = [g["rows"] for (i, _), g in groups.items() if i == iid and not g["proxy"]]
        proxy = [g["rows"] for (i, _), g in groups.items() if i == iid and g["proxy"]]

        def above(rows):
            return all(metrics[m]["water"] != "below" for m in rows if m in metrics)

        def views(rows):
            return {metrics[m]["view"] for m in rows if m in metrics}

        def where(rows):
            v = views(rows)
            if BOTH in v or v == {SIDE, HEAD_ON}:
                return "both"
            return next(iter(v)) if len(v) == 1 else None

        placed = False
        for key in ("side", "head_on", "both"):
            if any(above(rows) and where(rows) == key for rows in direct):
                out[key].append(name)
                placed = True
                break
        if not placed:
            estimable = any(above(rows) for rows in proxy)
            out["underwater" if estimable else "underwater_only"].append(name)
    return [
        {"key": "side", "title": "From your side clip", "note": "Measured from above the water.", "issues": sorted(out["side"])},
        {"key": "head_on", "title": "From your head-on clip", "note": "Measured from above the water.", "issues": sorted(out["head_on"])},
        {"key": "both", "title": "From both clips together", "note": "Combined into a 3D picture of your typical stroke.",
         "issues": sorted(out["both"])},
        {"key": "underwater", "title": "Estimated, unless you add an underwater clip",
         "note": "These happen under the surface. From above the water we estimate them and mark them \"estimated\" in your report; an optional underwater clip measures them directly.",
         "issues": sorted(out["underwater"])},
        {"key": "underwater", "title": "Only with an underwater clip",
         "note": "We can't judge these from above the water, and the report will say so.",
         "issues": sorted(out["underwater_only"])},
    ]
