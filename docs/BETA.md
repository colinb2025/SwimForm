# Running the beta

What's built, the order to switch things on, and the weekly routine.

## Launch order

1. **Deploy** (docs/SETUP.md §1, §4, §9) and fill in `.env` from `.env.example`. Run
   `python -m app.check` until it prints "Ready for beta testers".
2. **Create your own account** with the email in `SWIMFORM_ADMINS`. Admin emails can always
   sign up, even in invite mode. Open **Beta console** from the account menu.
3. **Press "Send me a test email"** on the console. If it doesn't arrive, fix email first:
   invites, password resets and "your report is ready" all depend on it.
4. **Wave 0, collect footage** (`SWIMFORM_POSE=collect`, `SWIMFORM_SIGNUP=invite`): make an
   invite code per group at `/admin/invites` and send people `https://<site>/signup?invite=CODE`.
   They film, upload and tick "Help improve SwimForm". They're told the report is coming.
5. **Train the model on what they sent** (below).
6. **Wave 1, real reports** (`SWIMFORM_POSE=yolo`, `SWIMFORM_REVIEW=1`): press **Run all waiting
   analyses**. Read each report before releasing it; add a coach's note. Everyone gets an email.
7. **Open up** when the feedback page says swimmers agree with the findings: `SWIMFORM_REVIEW=0`,
   then Stripe (SETUP.md §7), then `SWIMFORM_SIGNUP=open`.

## Training the model from beta clips

On the machine with the GPU, with the repo and `pip install -e ".[pose,label,app]"`:

    python -m app.export_clips --out clips/                 # only clips whose owners said yes
    python -m swimform.train prepare --clips clips/ --round 1
    #   ...correct the pre-placed points in Label Studio, export JSON (docs/labeling.md)
    python -m swimform.train fit --round 1 --export export.json

`fit` builds the dataset, fine-tunes, grades the result on held-out clips, saves
`training/models/swimform-r1.pt`, and prints the two settings that switch the site to it. If it
says "not at the bar yet", run `prepare --round 2`: the new model pre-places the points, so each
round is less work. `export_clips` needs the same `SWIMFORM_DB` / storage settings as the site.

`clips/manifest.csv` maps every exported file to its account. If someone deletes their account
or withdraws consent, delete their files from `clips/` and their frames from `training/round*/images`
before the next `fit`: the privacy page promises this.

## Weekly routine

- `/admin/analyses?status=review`: release held reports.
- `/admin/analyses?status=failed`: each row shows why; **Re-run** after a fix.
- `/admin/feedback`: findings are listed least-agreed-with first. Low agreement on one finding
  means its threshold in `rules/swim_rules.xlsx` (still DRAFT) or its measurement needs work.
  Download the CSV to compare against your own read of the clip.
- `/admin/invites`: invite the next few from the waitlist.

## What the swimmer sees in each state

| State | Reports list | Report page |
|---|---|---|
| queued / running | Measuring… | progress page, refreshes itself |
| waiting (collect mode) | Clips received · report coming | "we're still finishing the analysis" |
| review | Being checked | "a coach looks over every report" |
| done | counts | the report, feedback buttons, your note if you left one |
| rejected | Clips need re-filming | what to change, per clip |
| failed | Didn't finish | apology; the reason is on the admin row |

## Not built yet

- A real pose model. Until `fit` passes, use `collect`. `synthetic` shows simulated reports
  with a "Demo data" banner and is for trying the site, not for testers.
- Tiers beyond Free and Pro; annual pricing; coupons beyond Stripe's own promo codes.
- A separate job worker (one app process only; see `app/jobs.py`).
- Google's official sign-in button artwork (a plain button for now).
- ultralytics is AGPL-3.0: get its commercial licence or change backbone before charging.
