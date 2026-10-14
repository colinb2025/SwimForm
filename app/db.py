"""
Accounts and analysis ownership.

SQLite by default (SWIMFORM_DB, any SQLAlchemy URL), so moving to Postgres is a
config change. Report contents stay as JSON files under DATA/analyses/<id>/; the
database records who owns each one, plus a small summary for the reports list.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
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
    status: Mapped[str] = mapped_column(String(16), default="done", index=True)
    # rejected: JSON {slot: [fix, ...]}; failed: a short internal note (never shown raw)
    problems: Mapped[str] = mapped_column(Text, default="")


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
    """Bring a database made by an earlier version up to date (columns added since).
    Small and additive only; switch to Alembic before the schema changes in bigger ways."""
    from sqlalchemy import inspect, text
    have = {c["name"] for c in inspect(engine).get_columns("users")}
    have_a = {c["name"] for c in inspect(engine).get_columns("analyses")}
    with engine.begin() as conn:
        if "google_sub" not in have:
            conn.execute(text("ALTER TABLE users ADD COLUMN google_sub VARCHAR(255)"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)"))
        if "status" not in have_a:   # rows from before background jobs were all finished
            conn.execute(text("ALTER TABLE analyses ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT 'done'"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_analyses_status ON analyses (status)"))
        if "problems" not in have_a:
            conn.execute(text("ALTER TABLE analyses ADD COLUMN problems TEXT NOT NULL DEFAULT ''"))


def session() -> Session:
    if _Session is None:
        init()
    return _Session()


def user_by_email(s: Session, email: str):
    return s.scalar(select(User).where(User.email == email.strip().lower()))
