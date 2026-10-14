import os

import pytest

# Run analyses inside the request so tests see finished reports; test_jobs.py covers the
# background thread path explicitly.
os.environ.setdefault("SWIMFORM_JOBS", "inline")

# SWIMFORM_TEST_DB=postgresql+psycopg://... runs every app test against that database
# instead of a throwaway SQLite file (each test starts from empty tables).
TEST_DB = os.environ.get("SWIMFORM_TEST_DB")


def pytest_configure(config):
    config.addinivalue_line("markers", "sqlite_only: builds its own SQLite file; never redirected")


@pytest.fixture(autouse=True)
def _test_database(monkeypatch, request):
    if not TEST_DB or request.node.get_closest_marker("sqlite_only"):
        yield
        return
    try:
        from app import db
    except ImportError:
        yield
        return
    real = db.init

    def init(url=None):
        engine = real(TEST_DB)
        db.Base.metadata.drop_all(engine)
        return real(TEST_DB)

    monkeypatch.setattr(db, "init", init)
    yield
    if db._engine is not None:
        db._engine.dispose()
