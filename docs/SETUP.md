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

## 6. Beta settings

| Setting | What to put |
|---|---|
| `SWIMFORM_ADMINS` | Your email. That account sees the **Beta console** at `/admin` (also in the account menu). |
| `SWIMFORM_SIGNUP=invite` | Sign-up needs a code. Make codes at `/admin/invites`; everyone else lands on the waitlist, and one click emails a waiting person their invite. |
| `SWIMFORM_POSE=collect` | Until the model is trained: uploads are stored and the swimmer is told the report is coming. When you have weights, switch to `yolo` and press **Run all waiting analyses** in `/admin/analyses`; everyone is emailed their report. |
| `SWIMFORM_REVIEW=1` | Each finished report waits for you. Open it from `/admin/analyses`, add a note if you like, press **Release**. Turn off once you trust the model. |
| `SWIMFORM_CONTACT_EMAIL`, `SWIMFORM_COMPANY`, `SWIMFORM_GOVERNING_LAW` | Fill the Terms and Privacy pages. **They are drafts written to match what the code does, not legal advice: have a lawyer read them before you charge money.** |

`python -m app.check` (or the top of `/admin`) lists what is connected and what is still missing.

## 7. Payments (Stripe, about 20 minutes, optional)

Leave the Stripe values blank and everything is free. To charge:

1. Create a Stripe account. Stay in **test mode** until the flow works end to end.
2. Product catalog → **Add product** → "SwimForm Pro", recurring price. Copy the price ID (`price_...`) → `STRIPE_PRICE_ID`.
3. Developers → API keys → copy the secret key (`sk_test_...`) → `STRIPE_SECRET_KEY`.
4. Developers → Webhooks → **Add endpoint**: `https://<your site>/webhooks/stripe`, events
   `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`.
   Copy the signing secret (`whsec_...`) → `STRIPE_WEBHOOK_SECRET`.
5. Settings → Billing → Customer portal → turn it on (this is the "Manage or cancel" page).
6. Set `SWIMFORM_FREE_ANALYSES` (e.g. `2`) and `SWIMFORM_PRICE_LABEL` (e.g. `$12 a month`).
7. Test with card `4242 4242 4242 4242`: upgrade from `/billing`, check the account shows Pro, cancel from the portal.
8. Swap in the live keys (`sk_live_...`, a live price and a live webhook secret).

There is one paid plan (Pro = unlimited) as a placeholder until tiers are decided. You can give any
account Pro for free from `/admin/users`.

## 8. Clip storage (optional)

Default: clips and reports live in `SWIMFORM_DATA` on the server. That needs a persistent disk.
On hosts that wipe the disk on each deploy, use a bucket: `SWIMFORM_STORAGE=s3`, `S3_BUCKET`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, plus `S3_ENDPOINT_URL` for
Cloudflare R2 or Backblaze B2. Keep the bucket private; the app never hands out bucket links.

## 9. Hosting checklist

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
| `SWIMFORM_POSE` | `synthetic` | `synthetic` (demo), `collect` (store clips, report later) or `yolo` |
| `SWIMFORM_ADMINS` | – | Emails that can open `/admin` |
| `SWIMFORM_SIGNUP` | `open` | `open`, `invite` or `closed` |
| `SWIMFORM_REVIEW` | `0` | `1` = reports wait for an admin's release |
| `SWIMFORM_FIRST_RUN` | `1` | `0` = skip the filming walkthrough before a first upload |
| `SWIMFORM_FREE_ANALYSES` | no limit | Free analyses per account per month |
| `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `SWIMFORM_PRICE_LABEL` | – | Payments |
| `SWIMFORM_STORAGE`, `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_*` | `local` | Where clips and reports are kept |
| `SWIMFORM_COMPANY`, `SWIMFORM_CONTACT_EMAIL`, `SWIMFORM_GOVERNING_LAW` | – | Terms and Privacy pages |
| `SENTRY_DSN` | – | Send crashes to Sentry |
| `SWIMFORM_WEIGHTS` | – | Model weights for `yolo` |
| `SWIMFORM_JOBS` | `thread` | `thread` (background) or `inline` (in the request; tests) |
| `SWIMFORM_WORKERS` | `1` | Analyses that can run at once |
