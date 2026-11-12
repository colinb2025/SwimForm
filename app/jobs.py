"""
Background analysis jobs.

Pose estimation on two real clips takes far longer than a web request should wait, so
the upload request only saves the clips and queues a job; the report page shows a
"measuring" state until the job finishes.

SWIMFORM_JOBS
  thread  (default) a small pool of worker threads inside the app process.
  inline  run the job inside the request (tests, debugging).
SWIMFORM_WORKERS  pool size for "thread" (default 1: pose models are memory-hungry).
SWIMFORM_REVIEW   1 = a finished report is held (status "review") until an admin releases
                  it from /admin, optionally with a coach's note. For the early beta,
                  while the model is still being proven.

The pool lives in one app process. Run a single app process (uvicorn's default), or move
jobs to a separate worker service before scaling out: on startup, this process re-queues
anything left queued or running, which is only safe when it is the only process.
"""
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, update

from . import db, notify, pipeline, storage

log = logging.getLogger("swimform.jobs")
NOTIFY_AFTER_S = 20   # slower than this and the swimmer has likely left the page: email them

_pool = None
_pool_lock = threading.Lock()
# Set by the app at startup: builds the list-view summary from a finished record.
summarise = None


def mode():
    return os.environ.get("SWIMFORM_JOBS", "thread")


def _executor():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SWIMFORM_WORKERS", "1")),
                                       thread_name_prefix="analysis")
        return _pool


def submit(aid):
    if mode() == "inline":
        work(aid)
    else:
        _executor().submit(work, aid)


def _claim(aid):
    """queued -> running, atomically, so a job can't run twice."""
    with db.session() as s:
        n = s.execute(update(db.Analysis).where(db.Analysis.id == aid, db.Analysis.status == "queued")
                      .values(status="running")).rowcount
        s.commit()
        return n == 1


def _finish(aid, **values):
    with db.session() as s:
        s.execute(update(db.Analysis).where(db.Analysis.id == aid).values(**values))
        s.commit()


def review_on():
    """Hold finished reports for an admin's OK before the swimmer sees them?"""
    return os.environ.get("SWIMFORM_REVIEW", "0") == "1"


def _offload(aid):
    try:
        storage.offload(pipeline.folder(aid), aid)
    except Exception:   # noqa: BLE001 - the report is still on local disk; try again on the next run
        log.exception("could not copy analysis %s to storage", aid)


def work(aid, tell=None):
    """Run one queued analysis to a resting state.
    tell: email the owner when it's done? None = only if it took long enough that they
    have probably left the page."""
    if not _claim(aid):
        return
    started = time.monotonic()
    up = pipeline.upload_info(aid) or {}
    if up.get("provider") == "collect":
        _finish(aid, status="waiting")
        _offload(aid)
        return
    try:
        record = pipeline.process(aid)
    except pipeline.UploadProblem as e:
        _finish(aid, status="rejected", problems=json.dumps(e.problems))
        _offload(aid)
        if tell or (tell is None and time.monotonic() - started > NOTIFY_AFTER_S):
            notify.clips_rejected(aid)
        return
    except Exception as e:   # noqa: BLE001 - one bad clip must never take the worker down
        log.exception("analysis %s failed", aid)
        _finish(aid, status="failed", problems=f"{type(e).__name__}: {e}"[:500])
        _offload(aid)
        return
    summary = summarise(record) if summarise else {}
    if review_on():
        _finish(aid, status="review", **summary)
        _offload(aid)
        notify.review_needed(aid)
        return
    _finish(aid, status="done", **summary)
    _offload(aid)
    if tell or (tell is None and time.monotonic() - started > NOTIFY_AFTER_S):
        notify.report_ready(aid)


def release(aid, coach_note=""):
    """An admin approves a held report: the swimmer can now see it, and is told."""
    with db.session() as s:
        n = s.execute(update(db.Analysis).where(db.Analysis.id == aid, db.Analysis.status == "review")
                      .values(status="done", coach_note=coach_note.strip()[:4000])).rowcount
        s.commit()
    if n:
        notify.report_ready(aid)
    return n == 1


def rerun(aid):
    """Analyse stored clips again with the pose model configured now (after the model
    improved, or for clips collected before there was one). -> True if it was queued."""
    provider = os.environ.get("SWIMFORM_POSE", "synthetic")
    if provider == "collect" or pipeline.upload_info(aid) is None:
        return False
    pipeline.set_provider(aid, provider)
    with db.session() as s:
        n = s.execute(update(db.Analysis).where(db.Analysis.id == aid, db.Analysis.status.notin_(("queued", "running")))
                      .values(status="queued", problems="")).rowcount
        s.commit()
    if not n:
        return False
    if mode() == "inline":
        work(aid, tell=True)
    else:
        _executor().submit(work, aid, True)
    return True


def resume():
    """Re-queue work interrupted by a restart. Called once at app start."""
    with db.session() as s:
        stuck = list(s.scalars(select(db.Analysis.id).where(db.Analysis.status.in_(("queued", "running")))))
        if stuck:
            s.execute(update(db.Analysis).where(db.Analysis.id.in_(stuck)).values(status="queued"))
            s.commit()
    for aid in stuck:
        if pipeline.upload_info(aid) is not None:
            submit(aid)
        else:
            _finish(aid, status="failed", problems="uploads missing after restart")
    return stuck


def wait_all():
    """Block until every submitted job has finished (tests)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True)
