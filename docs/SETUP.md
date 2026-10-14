# Setting up SwimForm

The connections the app needs, each one step by step. Nothing here is required to try the app
locally: with no settings at all it runs with demo pose data, prints emails to the terminal
and hides the Google button.

Settings go in environment variables. For Docker, copy `.env.example` to `.env` and fill it
in. Never commit `.env`.

## 1. Run it locally (5 minutes)

```bash
pip install -e ".[app]"
uvicorn app.main:app --reload
# open http://localhost:8000
```

To get the production-like stack with Postgres, run `cp .env.example .env`, then `docker compose up --build`.

## 2. Google sign-in (about 15 minutes)

You'll get two values out of this: `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.

1. **Create a project.** Go to <https://console.cloud.google.com/projectcreate> and name it
   "SwimForm". Then switch to it in the project picker at the top left. Billing isn't needed.
2. **Set up the consent screen.** This is the box people see when they click "Continue with Google".
   Search "OAuth consent screen" in the top bar, or go to <https://console.cloud.google.com/auth/branding>,
   then click **Get started**:
   - App information: app name **SwimForm** and your support email.
   - Audience: **External**. This choice can't be changed later.
   - Contact information: your email.
   - Agree to the policy, then click **Create**.
3. **Scopes.** On the **Data Access** tab, open **Add or remove scopes** and tick
   `openid`, `.../auth/userinfo.email` and `.../auth/userinfo.profile`. These are
   non-sensitive scopes, so the app won't need Google's verification review.
4. **Create the client.** On the **Clients** tab, click **Create client** and choose **Web application**.
   - Name: anything, e.g. "SwimForm web".
   - Authorized JavaScript origins: leave empty. The server does the exchange.
   - **Authorized redirect URIs**: add one for each place the app runs, character for character:
     - `http://localhost:8000/auth/google/callback` for your machine
     - `https://YOUR-DOMAIN/auth/google/callback` for production, once there is one
   - Click **Create**.
5. **Copy the ID and secret now.** The secret is shown only once. If you lose it, use
   **Add secret** on the client and then delete the old one.
6. **Test users.** New apps start in *testing* mode, and only the accounts you list can sign in.
   Everyone else sees `access_blocked`. On the **Audience** tab, add yourself and any testers.
   When you open it to everyone, publish the app from that same tab.
7. Set the values:
   ```
   GOOGLE_CLIENT_ID=1234-abc.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=GOCSPX-...
   SWIMFORM_BASE_URL=http://localhost:8000      # production: https://YOUR-DOMAIN
   ```
   Restart the app and the "Continue with Google" button appears.

**If it fails:**
- `redirect_uri_mismatch`: the URI in step 4 must match `SWIMFORM_BASE_URL` + `/auth/google/callback`
  exactly, including http vs https, the port and a trailing slash.
- `access_blocked`: the account isn't a test user (step 6).

**Before launch:** swap the plain "Continue with Google" button for Google's official branded
button (<https://developers.google.com/identity/branding-guidelines>).

## 3. Email (password resets, email-change notices)

Pick one:

| `SWIMFORM_EMAIL` | Use when | Also set |
|---|---|---|
| `console` (default) | Local development. Emails are printed in the terminal. | nothing |
| `smtp` | Any email provider (Postmark, Resend, SendGrid, Mailgun, Google Workspace…) | `SMTP_HOST`, `SMTP_PORT` (587, or 465), `SMTP_USER`, `SMTP_PASSWORD` |
| `ses` | Amazon SES | `AWS_REGION`, plus AWS credentials or an instance role |

Always set `SWIMFORM_EMAIL_FROM`, e.g. `SwimForm <no-reply@yourdomain.com>`. The sending
domain must be verified with your provider (they'll give you DNS records to add). Without
that, password-reset emails land in spam or bounce.

## 4. Database

SQLite (the default) is fine on one machine. For hosting, use Postgres:

```
SWIMFORM_DB=postgresql+psycopg://USER:PASSWORD@HOST:5432/swimform
```

Install the driver with `pip install -e ".[app,postgres]"`. The Docker image already has it.
Tables are created on first start. The full test suite also runs against Postgres:
`SWIMFORM_TEST_DB=postgresql+psycopg://... pytest`.

## 5. The pose model

```
SWIMFORM_POSE=yolo
SWIMFORM_WEIGHTS=/path/to/best.pt
```

Build the image with the model libraries: `docker build --build-arg POSE=1 -t swimform .`.
Before switching, run the scorecard against your held-out frames and check it passes
(see [labeling.md](labeling.md#is-the-model-good-enough)).
Note that ultralytics is AGPL-3.0, so it needs a commercial licence before this is sold as a product.

## 6. Hosting checklist

- [ ] Serve over HTTPS and set `SWIMFORM_SECURE_COOKIES=1` (the Docker image defaults to 1).
- [ ] Set `SWIMFORM_BASE_URL` to the public https address.
- [ ] Add the production redirect URI to the Google client.
- [ ] Postgres, with backups.
- [ ] A persistent volume for `/data`, where uploaded clips and reports live.
- [ ] Run one app process. Analyses run in background threads inside it (`SWIMFORM_WORKERS`,
      default 1). To scale past one process, move jobs to a separate worker first; see `app/jobs.py`.
- [ ] Health check: `GET /healthz`.

## All settings

| Variable | Default | What it does |
|---|---|---|
| `SWIMFORM_BASE_URL` | the request's own address | Public address, used in emailed links and the Google redirect |
| `SWIMFORM_SECURE_COOKIES` | `0` | `1` = session cookie only over HTTPS |
| `SWIMFORM_DB` | SQLite in the data folder | Any SQLAlchemy URL |
| `SWIMFORM_DATA` | `./data` | Uploaded clips, pose output, reports |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | unset | Google sign-in, which is hidden when unset |
| `SWIMFORM_EMAIL` | `console` | `console`, `smtp` or `ses` |
| `SWIMFORM_EMAIL_FROM` | `SwimForm <no-reply@localhost>` | Sender address |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | – | For `smtp` |
| `SWIMFORM_POSE` | `synthetic` | `synthetic` (demo) or `yolo` |
| `SWIMFORM_WEIGHTS` | – | Model weights for `yolo` |
| `SWIMFORM_JOBS` | `thread` | `thread` (background) or `inline` (in the request; tests) |
| `SWIMFORM_WORKERS` | `1` | Analyses that can run at once |
