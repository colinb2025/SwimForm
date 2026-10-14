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

  python -m swimform.analyze --side side_pose.json --head-on head_on_pose.json
  python -m swimform.analyze --synthetic dropped_elbow=1 late_breath=0.7
"""
import argparse, json
from pathlib import Path

import numpy as np

from . import metrics as M
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

    per_view = {v: M.compute(tr) for v, tr in tracks.items()}
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

    flagged = {f["issue_id"] for f in result["priorities"]} | set(result["other_issues"])
    # Checked and fine, with how it was checked: a clear bill from a proxy is weaker.
    issues_clear = {i["issue_id"]: result["assessed_with"].get(i["issue_id"], "direct") for i in rules["issues"]
                    if i["issue_id"] not in flagged and i["issue_id"] not in issues_not_assessed}
    other_issues = [{"issue_id": iid, "evidence": result["assessed_with"].get(iid, "direct")}
                    for iid in result["other_issues"]]

    return {
        "views": {v: {"frames": tracks[v].n, "fps": tracks[v].fps, "cycles": len(per_view[v]["cycles"]),
                      "detection_rates": tracks[v].detection_rates()} for v in tracks},
        "measurements": detail,
        "metrics_not_assessed": not_assessed,
        "priorities": result["priorities"],
        "other_issues": other_issues,
        "issues_not_assessed": issues_not_assessed,
        "issues_clear": issues_clear,  # {issue_id: "direct" | "proxy"}: checked, and not a problem
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
