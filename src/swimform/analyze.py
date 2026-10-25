"""
One analysis = one side-view clip + one head-on clip.

Each metric is measured from the view the workbook assigns it (side, head_on, or
both: the 3D average stroke fused from the two clips, see stroke3d.py), so a flaw is
judged from the angle that can actually see it. Every measurement says how it was taken.

Where a fault cannot be measured directly (usually the catch, from a deck camera), the
workbook may allow a proxy: an indirect clue. Direct evidence always decides when it
exists. A finding that rests on a proxy is marked `evidence: proxy`, ranked lower, and
uses the workbook's hedged `proxy_observed` wording ("may be ... would confirm it"). A metric whose clip had too few
readable stroke cycles is reported as not assessed, with the reason, and any issue
that depends on it is reported as not assessed too. The athlete is never told about
a fault the footage could not show.

Arms are measured one at a time. From the side, only the arm (and leg) nearest the
camera is seen clearly, so a side clip is split into passes (see passes.py): film a 50,
down and back, and each arm gets a pass where it faces the camera. Each one-limb metric
is aggregated per arm; the worse arm decides the finding, and every arm-specific finding
says which arm it was seen on, which arms were checked, and which could not be.

  python -m swimform.analyze --side side_pose.json --head-on head_on_pose.json
  python -m swimform.analyze --synthetic dropped_elbow=1 late_breath=0.7
"""
import argparse, json
from pathlib import Path

import numpy as np

from . import metrics as M
from . import passes as P
from . import rules as R
from . import stroke3d as S3
from .posedata import PoseTrack
from .schema import BOTH, HEAD_ON, SIDE, VIEWS

# Resolved from the package location so the default works from any directory.
DEFAULT_RULES = str(Path(__file__).resolve().parents[2] / "rules" / "swim_rules.xlsx")
MIN_CYCLES = 3          # fewer readable cycles than this and a metric is not assessed
MIN_USABLE_FRAC = 0.5   # ...or if under half the detected cycles were readable

METHOD = {
    SIDE: "direct (side view)",
    HEAD_ON: "direct (head-on view)",
    BOTH: "3D stroke fused from side + head-on",
}

AGGREGATES = {
    "median": np.median, "min": np.min, "max": np.max,
    "p25": lambda v: np.percentile(v, 25), "p75": lambda v: np.percentile(v, 75),
}


class MissingViewError(ValueError):
    pass


class WorkbookMismatch(ValueError):
    pass


def _check_registry(rules):
    """Every workbook metric must be implemented, for the view the workbook names."""
    errs = []
    for m in rules["metrics"]:
        mid = m["metric_id"]
        if m["view"] == BOTH:
            if mid not in S3.FUSED_REGISTRY:
                errs.append(f"{mid}: workbook says view=both, but stroke3d.py has no 3D version")
        elif mid not in M.REGISTRY:
            errs.append(f"{mid}: in the workbook but not implemented in metrics.py")
        elif M.REGISTRY[mid][0] != m["view"]:
            errs.append(f"{mid}: workbook says view={m['view']}, metrics.py measures it from {M.REGISTRY[mid][0]}")
        if m.get("aggregate") not in AGGREGATES:
            errs.append(f"{mid}: unsupported aggregate {m.get('aggregate')!r}")
    if errs:
        raise WorkbookMismatch("; ".join(errs))


ARM_NAMES = {"r": "right", "l": "left"}
LEG_METRICS = {"kick_knee_angle", "heel_rise"}


def _compute_side(passes):
    """Metrics for each pass of a side clip, merged: cycles in clip frames, and each
    cycle tagged with the pass's near side (the arm and leg its sided metrics measured)."""
    merged = {"view": SIDE, "cycles": [], "values": {}, "errors": {}, "cycle_side": []}
    for p in passes:
        r = M.compute(p.track)
        near = r["near"] or p.near
        merged["cycles"] += [[a + p.start, b + p.start] for a, b in r["cycles"]]
        merged["cycle_side"] += [near] * len(r["cycles"])
        for mid, vals in r["values"].items():
            merged["values"].setdefault(mid, []).extend(vals)
        for mid, msg in r["errors"].items():
            merged["errors"].setdefault(mid, msg)
    if not passes:   # every pass too short: nothing measurable, but the metric list stays complete
        merged["values"] = {mid: [] for mid, (view, _) in M.REGISTRY.items() if view == SIDE}
    merged["near_sides"] = sorted({s for s in merged["cycle_side"]})
    return merged


def _worse_direction(rules):
    """metric_id -> "high" | "low": which way a reading is worse, from the operators of
    the rules that use it. None when the rules disagree or there are none."""
    seen = {}
    for r in rules["issue_rules"]:
        d = {">": "high", ">=": "high", "<": "low", "<=": "low"}.get(r["operator"])
        if d:
            seen.setdefault(r["metric_id"], set()).add(d)
    return {mid: next(iter(d)) for mid, d in seen.items() if len(d) == 1}


def _sided(m, res, worse, min_cycles, min_usable_frac):
    """Aggregate a one-limb metric per arm. -> reason (str) if not assessable, else
    (value of the worse arm, its usable per-cycle values, detail fields)."""
    mid, view = m["metric_id"], m["view"]
    vals, sides = res["values"].get(mid, []), res["cycle_side"]
    limb = "leg" if mid in LEG_METRICS else "arm"
    if not res["cycles"]:
        return f"no stroke cycles detected in the {view} video"
    arms, unseen = {}, {}
    for sd in ("r", "l"):
        applicable = [v for v, s in zip(vals, sides) if s == sd and v != M.SKIP]
        usable = [v for v in applicable if v is not None]
        if sd not in res["near_sides"]:
            unseen[sd] = f"your {ARM_NAMES[sd]} {limb} never faced the camera"
        elif not applicable:
            unseen[sd] = f"did not occur on the {ARM_NAMES[sd]} side"
        elif len(usable) < min_cycles or len(usable) / len(applicable) < min_usable_frac:
            unseen[sd] = f"only {len(usable)} of {len(applicable)} stroke cycles readable on the {ARM_NAMES[sd]} side"
        else:
            arms[sd] = {"value": round(float(AGGREGATES[m["aggregate"]](usable)), 3),
                        "usable_cycles": len(usable), "cycles": len(applicable), "_usable": usable}
    if not arms:
        applicable = [v for v in vals if v != M.SKIP]
        if not applicable:
            return f"did not occur in the {view} video"
        # Reported the way an unreadable two-sided metric always was: the clip's totals.
        n_use = sum(1 for v in applicable if v is not None)
        return f"only {n_use} of {len(applicable)} stroke cycles readable in the {view} video"
    if worse == "high":
        arm = max(arms, key=lambda k: arms[k]["value"])
    elif worse == "low":
        arm = min(arms, key=lambda k: arms[k]["value"])
    else:
        arm = None
    if arm is None and len(arms) == 2:
        value = float(np.mean([a["value"] for a in arms.values()]))
        usable = arms["r"]["_usable"] + arms["l"]["_usable"]
    else:
        arm = arm or next(iter(arms))
        value, usable = arms[arm]["value"], arms[arm]["_usable"]
    n_use = sum(a["usable_cycles"] for a in arms.values())
    n_all = sum(a["cycles"] for a in arms.values())
    d = {"usable_cycles": n_use, "cycles": n_all, "confidence": round(n_use / n_all, 2),
         "limb": limb, "arm": arm,
         "arms": {sd: {k: v for k, v in a.items() if k != "_usable"} for sd, a in arms.items()},
         "arms_unseen": unseen}
    return value, usable, d


def _deciding_groups(rules, iid, measurements):
    """The rule groups that decide an issue, as rules.evaluate picks them: measurable
    direct groups if any, else measurable proxy groups."""
    groups = {}
    for r in rules["issue_rules"]:
        if r["issue_id"] == iid:
            groups.setdefault(r.get("rule_group") or 1, []).append(r)
    measurable = [g for g in groups.values() if all(r["metric_id"] in measurements for r in g)]
    direct = [g for g in measurable if not any(r.get("evidence") == "proxy" for r in g)]
    return direct or measurable


def _issue_arms(rules, measurements, per_cycle, detail):
    """For issues decided by one-limb metrics: which arms were checked, and which show it.

    Re-runs the rules once per arm with that arm's readings, so "affected" means the
    same thing as a finding: the workbook rule fires on that arm's own measurements."""
    sided = {mid for mid, d in detail.items() if d.get("arms")}
    if not sided:
        return {}
    body = {k: v for k, v in measurements.items() if k not in sided}
    body_pc = {k: v for k, v in per_cycle.items() if k not in sided}
    per_arm = {}
    for sd in ("r", "l"):
        m, pc = dict(body), dict(body_pc)
        for mid in sided:
            a = detail[mid]["arms"].get(sd)
            if a:
                m[mid] = a["value"]
                pc.pop(mid, None)   # per-arm cycle lists aren't kept; judge on the arm's aggregate
        ev = R.evaluate(rules, m, pc, top_n=0)
        per_arm[sd] = (set(ev["other_issues"]), ev["assessed_with"])
    out = {}
    for i in rules["issues"]:
        iid = i["issue_id"]
        groups = _deciding_groups(rules, iid, measurements)
        # Only when every deciding rule is one-limb: a rule measured on the whole body
        # (or on the 3D stroke, which pools both arms) can't say which arm it saw.
        if not groups or any(not ({r["metric_id"] for r in g} <= sided) for g in groups):
            continue
        used = {r["metric_id"] for g in groups for r in g} & sided
        checked = [sd for sd in ("r", "l") if iid in per_arm[sd][1]
                   and all(sd in detail[mid]["arms"] for mid in used)]
        unseen = {}
        for mid in used:
            unseen.update({sd: why for sd, why in detail[mid]["arms_unseen"].items() if sd not in checked})
        out[iid] = {"limb": detail[next(iter(used))]["limb"],
                    "checked": checked,
                    "affected": [sd for sd in checked if iid in per_arm[sd][0]],
                    "unseen": unseen}
    return out


def analyze(tracks, rules=DEFAULT_RULES, require=VIEWS, top_n=3,
            min_cycles=MIN_CYCLES, min_usable_frac=MIN_USABLE_FRAC):
    """
    tracks: {"side": PoseTrack, "head_on": PoseTrack}
    rules:  path to the workbook, or an already-loaded rules dict
    require: views that must be present; analysis refuses to run without them
    """
    if isinstance(rules, (str, Path)):
        rules, errors = R.load(str(rules))
        if errors:
            raise WorkbookMismatch("workbook invalid: " + "; ".join(errors))
    _check_registry(rules)
    missing = [v for v in require if v not in tracks]
    if missing:
        raise MissingViewError(f"missing required view(s): {', '.join(missing)}")
    for v, tr in tracks.items():
        if tr.view != v:
            raise MissingViewError(f"track supplied as {v!r} is a {tr.view!r} track")

    proxy_metrics = {r["metric_id"] for r in rules["issue_rules"] if r.get("evidence") == "proxy"}

    def method(view, mid):
        return METHOD[view].replace("direct", "proxy", 1) if mid in proxy_metrics else METHOD[view]

    per_view, side_passes = {}, None
    for v, tr in tracks.items():
        if v == SIDE:
            side_passes = P.split(tr)
            per_view[v] = _compute_side(side_passes)
        else:
            per_view[v] = M.compute(tr)
    worse = _worse_direction(rules)
    fused = S3.compute(tracks[SIDE], tracks[HEAD_ON]) if SIDE in tracks and HEAD_ON in tracks else None

    measurements, per_cycle, detail, not_assessed = {}, {}, {}, {}
    for m in rules["metrics"]:
        mid, view = m["metric_id"], m["view"]
        if view == BOTH:
            if fused is None:
                not_assessed[mid] = "needs both a side and a head-on video"
            elif fused["reason"]:
                not_assessed[mid] = fused["reason"]
            elif fused["values"].get(mid) is None:
                not_assessed[mid] = "not visible in the 3D stroke fused from both videos"
            else:
                st = fused["stroke"]
                value = fused["values"][mid]
                measurements[mid] = round(value, 3)
                detail[mid] = {"value": round(value, 3), "view": BOTH, "method": method(BOTH, mid),
                               "usable_cycles": min(st.cycles.values()), "cycles": min(st.cycles.values()),
                               "confidence": round(st.agreement, 2), "view_agreement": st.agreement}
            continue
        if view not in per_view:
            not_assessed[mid] = f"no {view} video"
            continue
        res = per_view[view]
        if mid in res["errors"]:
            not_assessed[mid] = f"could not be computed ({res['errors'][mid]})"
            continue
        if mid in M.SIDED:
            out = _sided(m, res, worse.get(mid), min_cycles, min_usable_frac)
            if isinstance(out, str):
                not_assessed[mid] = out
                continue
            value, usable, d = out
            measurements[mid] = round(value, 3)
            per_cycle[mid] = usable
            detail[mid] = dict(d, value=round(value, 3), view=view, method=method(view, mid))
            continue
        vals = res["values"].get(mid, [])
        applicable = [v for v in vals if v != M.SKIP]
        usable = [v for v in applicable if v is not None]
        n = len(applicable)
        if not res["cycles"]:
            not_assessed[mid] = f"no stroke cycles detected in the {view} video"
            continue
        if n == 0:
            not_assessed[mid] = f"no breaths seen in the {view} video" if m.get("phase") == "breath" \
                else f"did not occur in the {view} video"
            continue
        if len(usable) < min_cycles or len(usable) / n < min_usable_frac:
            not_assessed[mid] = f"only {len(usable)} of {n} stroke cycles readable in the {view} video"
            continue
        value = float(AGGREGATES[m["aggregate"]](usable))
        measurements[mid] = round(value, 3)
        per_cycle[mid] = usable
        detail[mid] = {"value": round(value, 3), "view": view, "method": method(view, mid),
                       "usable_cycles": len(usable), "cycles": n, "confidence": round(len(usable) / n, 2)}

    result = R.evaluate(rules, measurements, per_cycle, top_n=top_n)
    for mid, s in result["scores"].items():
        detail[mid]["score"] = s

    # An issue is assessable when at least one of its rule groups is fully measured.
    groups = {}
    for r in rules["issue_rules"]:
        groups.setdefault(r["issue_id"], {}).setdefault(r.get("rule_group") or 1, []).append(r["metric_id"])
    issues_not_assessed = {}
    for i in rules["issues"]:
        iid = i["issue_id"]
        g = groups.get(iid, {})
        if not any(all(mid in measurements for mid in mids) for mids in g.values()):
            reasons = sorted({not_assessed.get(mid, "not measured") for mids in g.values() for mid in mids
                              if mid not in measurements})
            issues_not_assessed[iid] = "; ".join(reasons)

    issue_arms = _issue_arms(rules, measurements, per_cycle, detail)
    for f in result["priorities"]:
        if f["issue_id"] in issue_arms:
            f["arms"] = issue_arms[f["issue_id"]]

    flagged = {f["issue_id"] for f in result["priorities"]} | set(result["other_issues"])
    # Checked and fine, with how it was checked: a clear bill from a proxy is weaker.
    issues_clear = {i["issue_id"]: result["assessed_with"].get(i["issue_id"], "direct") for i in rules["issues"]
                    if i["issue_id"] not in flagged and i["issue_id"] not in issues_not_assessed}
    other_issues = [dict({"issue_id": iid, "evidence": result["assessed_with"].get(iid, "direct")},
                         **({"arms": issue_arms[iid]} if iid in issue_arms else {}))
                    for iid in result["other_issues"]]

    # Each view's average stroke cycle, for drawing posture snapshots in the report:
    # {view: [[ [x, y] | null per keypoint ] per phase bin]}, pixels relative to the
    # shoulder midpoint, side-view x pointing in the direction of travel.
    average_strokes = {}
    for v, tr in tracks.items():
        try:
            mc, _, _ = S3.mean_cycle(tr)
        except Exception:
            mc = None
        if mc is not None:
            average_strokes[v] = [[None if np.isnan(pt).any() else [round(float(pt[0]), 1), round(float(pt[1]), 1)]
                                   for pt in frame] for frame in mc]

    return {
        "average_strokes": average_strokes,
        "views": {v: {"frames": tracks[v].n, "fps": tracks[v].fps, "cycles": len(per_view[v]["cycles"]),
                      "detection_rates": tracks[v].detection_rates()} for v in tracks},
        "measurements": detail,
        "metrics_not_assessed": not_assessed,
        "priorities": result["priorities"],
        "other_issues": other_issues,
        "issues_not_assessed": issues_not_assessed,
        "issues_clear": issues_clear,  # {issue_id: "direct" | "proxy"}: checked, and not a problem
        # Arm-specific issues: {issue_id: {"limb", "checked": [...], "affected": [...], "unseen": {...}}}
        "issue_arms": issue_arms,
        "side_passes": P.summary(side_passes) if side_passes is not None else None,
        "errors": {mid: msg for v in per_view.values() for mid, msg in v["errors"].items()},
        "stroke3d": None if fused is None else {
            "used": fused["reason"] is None,
            "reason": fused["reason"],
            "view_agreement": fused["stroke"].agreement if fused["stroke"] else None,
            "cycles_averaged": fused["stroke"].cycles if fused["stroke"] else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side")
    ap.add_argument("--head-on")
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--synthetic", nargs="*", metavar="ISSUE=SEVERITY",
                    help="analyze a synthetic swimmer instead of pose files")
    ap.add_argument("--quality", default="typical")
    ap.add_argument("--camera", default="underwater")
    a = ap.parse_args()
    if a.synthetic is not None:
        from .synthetic import Swimmer, generate
        sw = Swimmer({k: float(v) for k, v in (s.split("=") for s in a.synthetic)})
        tracks = {v: generate(v, sw, quality=a.quality, camera=a.camera) for v in VIEWS}
    else:
        tracks = {}
        if a.side:
            tracks[SIDE] = PoseTrack.from_json(a.side, view=SIDE)
        if a.head_on:
            tracks[HEAD_ON] = PoseTrack.from_json(a.head_on, view=HEAD_ON)
    print(json.dumps(analyze(tracks, a.rules), indent=2, default=str))


if __name__ == "__main__":
    main()
