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
    password_hash: Mapped[str] = mapped_column(String(255))
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


_engine = None
_Session = None


def init(url=None):
    """(Re)connect; creates tables if missing. Called at app start and by tests."""
    global _engine, _Session
    url = url or os.environ.get("SWIMFORM_DB") or f"sqlite:///{ROOT / 'data' / 'swimform.sqlite3'}"
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def session() -> Session:
    if _Session is None:
        init()
    return _Session()


def user_by_email(s: Session, email: str):
    return s.scalar(select(User).where(User.email == email.strip().lower()))
