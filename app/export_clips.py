"""
Copy beta clips whose owners agreed to model training into a folder for labeling.

  python -m app.export_clips --out clips/

Only uploads where the swimmer ticked "use these clips to improve the model" are
copied, and only while their account still exists. Files are named
u<user>_<view>_<analysis>.<ext>, so you can see whose is whose without a name. manifest.csv records where each file came from: when someone withdraws
consent or deletes their account, that is how you find and remove their frames.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from sqlalchemy import select

from . import db, pipeline, storage


def export(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with db.session() as s:
        rows = s.execute(select(db.Analysis, db.User.id).join(db.User, db.User.id == db.Analysis.user_id)
                         .where(db.Analysis.train_ok.is_(True)).order_by(db.Analysis.created)).all()
    manifest, copied = [], 0
    for a, uid in rows:
        up = pipeline.upload_info(a.id)
        if up is None:
            continue
        for slot, name in up["paths"].items():
            src = storage.fetch(pipeline.folder(a.id), a.id, name)
            if src is None:
                continue
            dest = out / f"u{uid}_{slot.replace('_', '')}_{a.id}{Path(name).suffix}"
            if not dest.exists():
                shutil.copy(src, dest)
                copied += 1
            manifest.append([dest.name, uid, a.id, slot, a.created.date().isoformat()])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "user_id", "analysis_id", "view", "uploaded"])
        w.writerows(manifest)
    return copied, len(manifest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    db.init()
    copied, total = export(a.out)
    print(f"{copied} new clips copied, {total} consented clips in {a.out} (see manifest.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
