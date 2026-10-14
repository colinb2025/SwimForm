"""
Background analysis jobs.

Pose estimation on two real clips takes far longer than a web request should wait, so
the upload request only saves the clips and queues a job; the report page shows a
"measuring" state until the job finishes.

SWIMFORM_JOBS
  thread  (default) a small pool of worker threads inside the app process.
  inline  run the job inside the request (tests, debugging).
SWIMFORM_WORKERS  pool size for "thread" (default 1: pose models are memory-hungry).

The pool lives in one app process. Run a single app process (uvicorn's default), or move
jobs to a separate worker service before scaling out: on startup, this process re-queues
anything left queued or running, which is only safe when it is the only process.
"""
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, update

from . import db, pipeline

log = logging.getLogger("swimform.jobs")

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


def work(aid):
    if not _claim(aid):
        return
    try:
        record = pipeline.process(aid)
    except pipeline.UploadProblem as e:
        _finish(aid, status="rejected", problems=json.dumps(e.problems))
        return
    except Exception as e:   # noqa: BLE001 - one bad clip must never take the worker down
        log.exception("analysis %s failed", aid)
        _finish(aid, status="failed", problems=f"{type(e).__name__}: {e}"[:500])
        return
    summary = summarise(record) if summarise else {}
    _finish(aid, status="done", **summary)


def resume():
    """Re-queue work interrupted by a restart. Called once at app start."""
    with db.session() as s:
        stuck = list(s.scalars(select(db.Analysis.id).where(db.Analysis.status.in_(("queued", "running")))))
        if stuck:
            s.execute(update(db.Analysis).where(db.Analysis.id.in_(stuck)).values(status="queued"))
            s.commit()
    for aid in stuck:
        if (pipeline.folder(aid) / "upload.json").exists():
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
