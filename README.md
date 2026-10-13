# swimform

Freestyle swim technique analysis. A video goes in; pose landmarks, biomechanical
measurements, scored metrics, a short list of prioritised faults, and the drills
that address them come out.

Status: **early**. The rules engine runs end to end on synthetic measurements.
The pose model does not work yet — see [docs/baseline-findings.md](docs/baseline-findings.md).
Every threshold in `rules/swim_rules.xlsx` is a draft awaiting expert review.

## The pipeline

```
video ──▶ pose ──▶ stroke cycles ──▶ metrics ──▶ scores ──▶ faults ──▶ drills
         (model)    (metrics.py)                     (rules.py, driven by the workbook)
```

Two layers, and only the first needs training:

- **Pose** (`swimform.pose`) turns video into per-frame landmarks. This is a trained
  model, and the open problem.
- **Rules** (`swimform.rules`) turn landmarks into faults and drills. Plain geometry
  plus thresholds. Nothing is learned here; it is calibrated against expert judgement.

## Install

```bash
pip install -e ".[pose]"
```

## Use

```bash
# 1. landmarks + overlay video + per-joint detection rates
python -m swimform.pose out/ yolo11m-pose.pt clips/*.mov

# 2. stroke cycles and side-view measurements
python -m swimform.metrics out/*_pose.json

# 3. validate the rules workbook and export it as JSON for the app
python -m swimform.rules rules/swim_rules.xlsx --demo
```

## The rules workbook

`rules/swim_rules.xlsx` is the single source of truth for what gets measured and what
it means. It is edited in Excel and read by `swimform.rules`; the engine has no
thresholds of its own.

| Sheet | Holds |
|---|---|
| `Metrics` | each measurement: required camera view, above or below water, stroke phase, landmarks, formula, ideal and fault ranges |
| `Issues` | athlete-facing "what we observed" and "why it matters", plus a 1–5 priority weight |
| `Issue_Rules` | the threshold logic. Same `rule_group` is ANDed; different groups are ORed |
| `Interventions` | drills, exercises and cues |
| `Issue_Interventions` | which interventions address which fault, ranked |

Scoring: 100 inside `[ideal_min, ideal_max]`, falling linearly to 0 at the fault bound.
Ranking: `priority_weight × (1 + how far past the threshold)`, so a mild fault on a
critical metric can outrank a severe one on a minor metric. The top 2–3 reach the athlete.

`swimform.rules` validates every cross-reference and enum before evaluating, so a typo
in a `metric_id` fails loudly instead of silently disabling a fault.

Rows are marked `draft` or `verified`. **Everything currently reads `draft`.**

## Labeling

The pose model has to be fine-tuned on labeled swim frames. See
[docs/labeling.md](docs/labeling.md) for the model-in-the-loop workflow:
the current model guesses, you correct, it retrains on your corrections.

## Repo layout

```
src/swimform/
  pose.py              video  -> landmarks, overlay, detection rates
  metrics.py           landmarks -> stroke cycles -> measurements
  rules.py             workbook + measurements -> scores, faults, drills
  labeling/prep.py     pick frames worth labeling, pre-fill model guesses
  labeling/convert.py  Label Studio export -> YOLO-pose dataset
rules/swim_rules.xlsx  metric, fault and drill definitions
labeling/              Label Studio labeling config
docs/                  baseline findings, labeling guide
```

Clips, frames, weights and pipeline output are gitignored. Nothing in this repo is
large, and nothing in it is personal data.

## Open decisions

- **Camera position.** Several defined metrics (the catch, mid-pull, hip rotation,
  kick) are underwater and cannot be measured from a deck-level camera. Either the
  recording spec requires an underwater view or those metrics come out of V1.
- **Pose backbone.** `ultralytics` is AGPL-3.0. Fine for research, not for shipping —
  RTMPose (Apache-2.0) is the likely replacement.
- **Repeatability.** Longitudinal tracking only means something if the same stroke
  filmed twice scores roughly the same. Not yet measured, and it belongs in the
  success criteria.
