# swimform

Freestyle swim technique analysis. A video goes in; pose landmarks, biomechanical
measurements, scored metrics, a short list of prioritised faults, and the drills
that address them come out.

Status: **early**. Metrics and rules run end to end on synthetic swimmers in the 15-point
schema, for both camera views, with every workbook fault covered by tests. The pose model
does not work yet — see [docs/baseline-findings.md](docs/baseline-findings.md).
Every threshold in `rules/swim_rules.xlsx` is a draft awaiting expert review.

## The pipeline

```
side video ────▶ pose ──▶ cycles ──▶ side metrics ─────┐
                                                        ├─▶ scores ──▶ faults ──▶ drills
head-on video ─▶ pose ──▶ cycles ──▶ head-on metrics ──┘
               (model)   (metrics.py)              (analyze.py + rules.py, driven by the workbook)
```

An analysis takes **both** a side-view and a head-on clip. Each metric is measured from the
view the workbook assigns it: `side`, `head_on`, or `both`, meaning the 3D average stroke
fused from the two clips (below). Every measurement in the report says how it was taken. If a clip shows too few readable stroke cycles for a metric,
that metric — and any fault depending on it — comes back **not assessed**, with the reason,
rather than guessed. A confident score from bad keypoints is worse than no score.

The report splits every workbook issue three ways: **flagged** (`priorities`, `other_issues`),
**clear** (checked and fine), and **not assessed** (with the reason). A metric that fails on
an odd clip is reported as not assessed under `errors`; the rest of the analysis still runs.

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

# 2. per-cycle measurements for one clip (pass --view if the JSON has no summary.view)
python -m swimform.metrics out/clip_pose.json --view side

# 3. full analysis: both views -> measurements, faults, drills, not-assessed reasons
python -m swimform.analyze --side out/side_pose.json --head-on out/head_on_pose.json

# validate the rules workbook and export it as JSON for the app
python -m swimform.rules rules/swim_rules.xlsx --demo
```

## The 3D average stroke

The two clips are separate laps, so no instant is filmed from both angles and joints can't
be triangulated frame by frame. Freestyle repeats, so `swimform.stroke3d` instead:

1. cuts each clip into stroke cycles and averages them into one typical cycle, by phase;
2. lines the two typical cycles up using depth, the one axis both cameras see: the head-on
   cycle is shifted in phase and scaled until its depth curves best match the side view's;
3. takes forward position from the side, lateral position from head-on, and depth from both.

How well the depth curves agree after alignment (r²) is reported with every 3D measurement;
below 0.8 the fused stroke is not used and its metrics come back not assessed. It describes
a *typical* stroke, so it suits faults that repeat every cycle, not one-offs.

On synthetic swimmers whose true 3D motion is known, the fused recovery elbow angle comes
back within ~1–2° of the truth on poor footage, from laps with different start points and
stroke rates.

## Web app

```bash
pip install -e ".[app]"
uvicorn app.main:app --reload        # then open http://127.0.0.1:8000
```

The athlete uploads a side-view and a head-on clip. Each clip is checked first
(`swimform.viewcheck`): a swapped or diagonal camera, a swimmer who is barely in shot, or
too few stroke cycles sends them back to the form with what to change. The report then
shows every check as a lane: needs work, may need work (estimated), couldn't check (with
what footage would fix it), or looks good, followed by drills, cues and land exercises for
the top findings and a "how we measured it" table.

Athletes sign up with an email and password; the first screen is the log-in page, then
"your reports", then upload. A report is visible only to the account that made it (anyone
else gets "not found"). Passwords are hashed with scrypt; sessions are httpOnly cookies whose
token is stored only as a hash; cross-site form posts are refused; repeated failed log-ins
are rate-limited.

| Setting | Default | |
|---|---|---|
| `SWIMFORM_DB` | `sqlite:///data/swimform.sqlite3` | any SQLAlchemy URL, e.g. Postgres in production |
| `SWIMFORM_DATA` | `data/` | where uploads and report JSON are kept |
| `SWIMFORM_SECURE_COOKIES` | `0` | set to `1` behind HTTPS |
| `SWIMFORM_POSE` | `synthetic` | `yolo` to run the real pose model |
| `SWIMFORM_BASE_URL` | from the request | public URL, used in reset emails and the Google redirect |
| `SWIMFORM_EMAIL` | `console` | `smtp` or `ses`; `console` prints emails (dev) |
| `SWIMFORM_EMAIL_FROM` | `no-reply@localhost` | sender, e.g. `SwimForm <no-reply@yourdomain.com>` |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | unset | turn on "Continue with Google" |

**Password reset.** "Forgot your password?" emails a link that works once, for an hour, and
signs the account out everywhere when used. The page gives the same answer whether or not an
email has an account, and sends at most three emails per address per hour. In development
the email is printed to the server log. For production, set `SWIMFORM_EMAIL=ses` (install
`.[ses]`; the sending domain must be verified in SES and the account out of the SES sandbox)
or `smtp` with `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`.

**Google sign-in.** In the Google Cloud console, create an OAuth client of type "Web
application" (after setting up the OAuth consent screen) and add
`<base url>/auth/google/callback` as an authorised redirect URI, e.g.
`http://localhost:8000/auth/google/callback` for local use. Set `GOOGLE_CLIENT_ID` and
`GOOGLE_CLIENT_SECRET`, and the button appears on the log-in and sign-up pages. A Google
sign-in with the same verified email as an existing account joins that account. Before
launch: replace the plain "Continue with Google" button with Google's official one from
their branding guidelines, and publish the consent screen.

The pose step is set by `SWIMFORM_POSE`. Until the fine-tuned model is wired in it defaults
to `synthetic`: a simulated swimmer derived from the uploaded files, and every report says
so in a "Demo data" banner. `SWIMFORM_POSE=yolo` with `SWIMFORM_WEIGHTS=path/to/best.pt`
runs `swimform.pose` instead. Not built yet: billing and plan limits, background jobs,
cloud storage, account settings (change email or password, delete account).

## Synthetic swimmers

`swimform.synthetic` generates pose tracks for a swimmer with any mix of the workbook's
faults, by `issue_id` and severity 0–1, at three footage qualities, from an underwater or a
deck-level camera. It is one 3D body projected into both views — body roll is a real
rotation and visibility fades with true depth — so the 3D fusion can be checked against
ground truth (`synthetic.truth3d`). It is how everything downstream of pose gets built and
tested before the model works.

```bash
python -m swimform.analyze --synthetic dropped_elbow=1 late_breath=1 --camera deck
python -m swimform.synthetic out/ --faults crossover_entry=0.8 --quality poor
pip install -e ".[dev]" && pytest
```

The tests hold three lines against the real workbook:

- a clean swimmer triggers **no** fault, across clean, noisy and poor footage, from either camera;
- each of the 14 faults triggers its own issue when the footage can show it;
- when it cannot (deck camera, underwater joint), the issue is reported not assessed.

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

**Direct vs proxy evidence.** Each `Issue_Rules` row has `evidence = direct` (measures the
fault itself) or `proxy` (an indirect clue, for footage that cannot show the fault, such as
where the hand breaks the surface when a deck camera cannot see the pull). Direct rules
decide whenever they can be measured, even over a disagreeing proxy. A finding resting on a
proxy is marked as such in the report, ranks lower, and uses the issue's `proxy_observed`
wording ("may be … an underwater side view would confirm it"). Proxy metrics are never
scored.

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
  schema.py            the 15-point keypoint schema (names, flip index, skeleton)
  posedata.py          pose JSON <-> PoseTrack: the contract any pose model must meet
  pose.py              video  -> landmarks, overlay, detection rates
  metrics.py           landmarks -> stroke cycles -> every workbook measurement, per view
  stroke3d.py          side + head-on tracks -> 3D average stroke -> 3D measurements
  analyze.py           side + head-on tracks -> measurements, faults, not-assessed reasons
  rules.py             workbook + measurements -> scores, faults, drills
  viewcheck.py         is this clip the right angle, with enough swimmer and strokes?
  synthetic.py         synthetic swimmers with injectable faults
app/                   web app: accounts, upload, clip checks, reports (FastAPI + Jinja)
  labeling/prep.py     pick frames worth labeling, pre-fill model guesses
  labeling/convert.py  Label Studio export -> YOLO-pose dataset
rules/swim_rules.xlsx  metric, fault and drill definitions
labeling/              Label Studio labeling config
docs/                  baseline findings, labeling guide
```

Clips, frames, weights and pipeline output are gitignored. Nothing in this repo is
large, and nothing in it is personal data.

## Findings from building the metrics

- **Deck-level coverage.** With visibility fading by depth, a deck camera can assess 12 of
  14 faults: 10 directly, plus short pull and over-gliding by proxy. Dropped elbow and
  straight-arm pull need underwater footage.
- **Proxy thresholds are calibrated on synthetic swimmers only.** Two known confounds: the
  hand-reappearance proxy for short pull reads over-gliders as somewhat short (threshold set
  to separate them), and spine angle rises with a lifted head as well as sinking hips. A
  "hand hidden time" proxy for over-gliding was tried and dropped: gliding hands sit
  shallow enough to stay visible, so it pointed the wrong way. Real coverage depends on how well the fine-tuned
  model places `occluded` underwater joints; re-measure once it runs on real clips.
- **`recovery_elbow_angle` is now measured in 3D.** A side camera flattens the outward swing
  of a high-elbow recovery, so its 2D angle read a textbook recovery as ~10° (score 11/100).
  On the fused 3D stroke it reads ~74° and scores 100. Its workbook view is now `both`.
- **Workbook edits for the 15-point schema** (no thresholds changed):
  `head_lift_angle` and `breath_head_lift` now use top-of-head and the spine;
  `breath_timing` moved from `side` to `head_on`, since head rotation is not visible from
  the side without face points; `recovery_elbow_angle` moved to `both` (3D); and `both`
  was added to the workbook's list of views.
- **`pose.py`, `labeling/prep.py` and `labeling/convert.py` are still COCO-17** in this
  branch. `PoseTrack` loads COCO files by name (face points dropped, spine points missing),
  so nothing breaks; the labeling tools should import `schema.KEYPOINTS` and `FLIP_IDX`.

- **`max` aggregates are the most fragile on real footage.** `breath_head_lift` and
  `kick_width` take the worst cycle, so one misread cycle can flag a fault. They pass on
  synthetic noise; consider `p75` once real clips are in.

## Open decisions

- **Camera position.** Several defined metrics (the catch, mid-pull, hip rotation,
  kick) are underwater and cannot be measured from a deck-level camera. Either the
  recording spec requires an underwater view or those metrics come out of V1.
- **Pose backbone.** `ultralytics` is AGPL-3.0. Fine for research, not for shipping —
  RTMPose (Apache-2.0) is the likely replacement.
- **Repeatability.** Longitudinal tracking only means something if the same stroke
  filmed twice scores roughly the same. Not yet measured, and it belongs in the
  success criteria.
