"""
Accounts and analysis ownership.

SQLite by default (SWIMFORM_DB, any SQLAlchemy URL), so moving to Postgres is a
config change. Report contents stay as JSON files under DATA/analyses/<id>/; the
database records who owns each one, plus a small summary for the reports list.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    # Empty for accounts created through Google that never set a password.
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # Billing (app/billing.py). plan: "free" | "pro"; set by Stripe webhooks or by an admin.
    plan: Mapped[str] = mapped_column(String(16), default="free")
    stripe_customer: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    stripe_subscription: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # May we use this account's clips to train the pose model? Opt-in, changeable in settings.
    train_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    terms_accepted: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # When they went through the filming walkthrough shown before a first upload.
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invite_code: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Invite(Base):
    """A beta invite code. Usable max_uses times (0 = unlimited)."""
    __tablename__ = "invites"
    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    note: Mapped[str] = mapped_column(String(200), default="")       # who or what it's for
    max_uses: Mapped[int] = mapped_column(Integer, default=1)
    uses: Mapped[int] = mapped_column(Integer, default=0)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Waitlist(Base):
    __tablename__ = "waitlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    note: Mapped[str] = mapped_column(String(500), default="")       # what they told us about themselves
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    invited: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Feedback(Base):
    """What the swimmer said about a report. issue_id "" is the report as a whole
    (verdict = a 1-5 rating); otherwise verdict is right | wrong | unsure for that finding."""
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("analysis_id", "issue_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    issue_id: Mapped[str] = mapped_column(String(64), default="")
    verdict: Mapped[str] = mapped_column(String(16), default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class PasswordReset(Base):
    __tablename__ = "password_resets"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used: Mapped[bool] = mapped_column(default=False)


class OAuthState(Base):
    """One pending Google sign-in: ties the callback back to the browser that started it."""
    __tablename__ = "oauth_states"
    state: Mapped[str] = mapped_column(String(64), primary_key=True)
    nonce: Mapped[str] = mapped_column(String(64))
    verifier: Mapped[str] = mapped_column(String(128))
    next_url: Mapped[str] = mapped_column(String(512), default="/")
    invite: Mapped[str] = mapped_column(String(32), default="")
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class LoginSession(Base):
    __tablename__ = "sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # Lane statuses in report order, e.g. "flagged,estimated,clear,...", for the list view.
    lanes: Mapped[str] = mapped_column(Text, default="")
    flagged: Mapped[int] = mapped_column(Integer, default=0)
    estimated: Mapped[int] = mapped_column(Integer, default=0)
    unchecked: Mapped[int] = mapped_column(Integer, default=0)
    # queued -> running -> done | rejected (a clip failed its check) | failed (our fault)
    # Beta extras: waiting (clips collected, no model yet), review (finished, held for a coach's OK).
    status: Mapped[str] = mapped_column(String(16), default="done", index=True)
    # rejected: JSON {slot: [fix, ...]}; failed: a short internal note (never shown raw)
    problems: Mapped[str] = mapped_column(Text, default="")
    # A personal note from the reviewing coach, shown at the top of the report.
    coach_note: Mapped[str] = mapped_column(Text, default="")
    # The swimmer agreed, for these clips, that we may use them to train the pose model.
    train_ok: Mapped[bool] = mapped_column(Boolean, default=False)


_engine = None
_Session = None


def init(url=None):
    """(Re)connect; creates tables if missing. Called at app start and by tests."""
    global _engine, _Session
    data = Path(os.environ.get("SWIMFORM_DATA", ROOT / "data"))
    url = url or os.environ.get("SWIMFORM_DB") or f"sqlite:///{data / 'swimform.sqlite3'}"
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(_engine)
    _add_missing_columns(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def _add_missing_columns(engine):
    """Bring a database made by an earlier version up to date: any column the models
    have and the table lacks is added. Additive only; switch to Alembic before the
    schema changes in bigger ways."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"
                if not col.nullable:
                    d = col.default.arg if col.default is not None and not callable(col.default.arg) else ""
                    lit = ("TRUE" if d else "FALSE") if isinstance(d, bool) else str(d) if isinstance(d, int) \
                        else "'" + str(d).replace("'", "''") + "'"
                    if table.name == "analyses" and col.name == "status":
                        lit = "'done'"   # rows from before background jobs were all finished
                    ddl += f" NOT NULL DEFAULT {lit}"
                conn.execute(text(ddl))
                if col.index or col.unique:
                    conn.execute(text(f"CREATE {'UNIQUE ' if col.unique else ''}INDEX IF NOT EXISTS "
                                      f"ix_{table.name}_{col.name} ON {table.name} ({col.name})"))


def session() -> Session:
    if _Session is None:
        init()
    return _Session()


def user_by_email(s: Session, email: str):
    return s.scalar(select(User).where(User.email == email.strip().lower()))
