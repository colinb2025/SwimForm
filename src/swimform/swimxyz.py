"""
SwimXYZ as a base training set: fetch the freestyle part of the dataset and look inside it.

SwimXYZ (Fiche et al., ACM MIG 2023) is a synthetic swimming dataset: rendered swimmers
with exact 2D and 3D joint positions on every frame, released under CC BY 4.0 (credit
required; commercial use allowed). Freestyle comes as three downloads from zenodo.org:

  labels   Freestyle_labels.zip   1.7 GB   joint annotations (COCO-17, COCO-25, 48-joint)
  side     Freestyle_part2.zip   36.0 GB   three side views: above water, water level, underwater
  front    Freestyle_part1.zip   40.3 GB   front and aerial views

  python -m swimform.swimxyz fetch labels            # start here: small
  python -m swimform.swimxyz inspect swimxyz/Freestyle_labels.zip
  python -m swimform.swimxyz fetch side front        # the big ones; safe to stop and resume

`inspect` prints how an archive is laid out and writes swimxyz_sample.zip (well under
1 MB: the file listing plus the first lines of a few annotation files). The converter
to our 15 keypoints is written against that sample, not guessed.

Credit to carry wherever the model or data is described:
  Fiche, G., Sevestre, V., Gonzalez-Barral, C., Leglaive, S., Séguier, R. "SwimXYZ: A
  large-scale dataset of synthetic swimming motions and videos." ACM MIG 2023. CC BY 4.0.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

FILES = {
    "labels": ("8399376", "Freestyle_labels.zip", None),
    "side": ("8402031", "Freestyle_part2.zip", "b2d38f70a144597aac06b02f7a8fc15b"),
    "front": ("8402009", "Freestyle_part1.zip", "e593eebc00f9d10390fc337766683ab6"),
}
TEXT_EXT = {".json", ".csv", ".txt", ".tsv", ".yaml", ".yml", ".xml", ".md"}
ARRAY_EXT = {".npy", ".npz"}
HEAD_BYTES = 4000        # how much of each sample file goes into the bundle
SAMPLES_PER_KIND = 3


def url(record, name):
    return f"https://zenodo.org/records/{record}/files/{name}?download=1"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(key, out, opener=urllib.request.urlopen, log=print):
    """Download one archive, resuming a partial file. -> path."""
    record, name, want = FILES[key]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest, part = out / name, out / (name + ".part")
    if dest.exists():
        log(f"{name}: already downloaded")
        return dest
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url(record, name), headers={"Range": f"bytes={have}-"} if have else {})
    with opener(req) as r:
        resumed = getattr(r, "status", 200) == 206
        total = int(r.headers.get("Content-Length") or 0) + (have if resumed else 0)
        if have and not resumed:      # the server sent the whole file again
            have = 0
        with open(part, "ab" if resumed else "wb") as f:
            done, shown = have, -1
            while block := r.read(1 << 20):
                f.write(block)
                done += len(block)
                pct = int(100 * done / total) if total else 0
                if pct != shown and pct % 5 == 0:
                    log(f"{name}: {pct}% ({done / 1e9:.1f} of {total / 1e9:.1f} GB)")
                    shown = pct
    if want and md5(part) != want:
        raise SystemExit(f"{name}: checksum mismatch. Delete {part} and run fetch again.")
    part.rename(dest)
    log(f"{name}: done -> {dest}")
    return dest


def _describe(name, data):
    """A short text description of one file's contents, without trusting it."""
    ext = Path(name).suffix.lower()
    if ext in ARRAY_EXT:
        import numpy as np
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as e:   # noqa: BLE001 - pickled or partial arrays: say so, don't load them
            return f"(array not readable without pickle: {type(e).__name__})"
        if hasattr(arr, "files"):
            return "\n".join(f"{k}: shape {arr[k].shape} dtype {arr[k].dtype}  first: {arr[k].ravel()[:8].tolist()}"
                             for k in arr.files)
        return f"shape {arr.shape} dtype {arr.dtype}\nfirst values: {arr.ravel()[:24].tolist()}"
    if ext == ".json":
        try:
            obj = json.loads(data)
        except ValueError:
            return data[:HEAD_BYTES].decode("utf-8", "replace")
        def shape(o, depth=0):
            if isinstance(o, dict):
                keys = list(o)[:12]
                return "{" + ", ".join(f"{k!r}: {shape(o[k], depth + 1)}" for k in keys) + (", ..." if len(o) > 12 else "") + "}" \
                    if depth < 3 else f"dict({len(o)})"
            if isinstance(o, list):
                return f"list({len(o)}) of {shape(o[0], depth + 1)}" if o else "list(0)"
            return repr(o)[:40]
        return "structure: " + shape(obj) + "\n\n" + data[:HEAD_BYTES].decode("utf-8", "replace")
    return data[:HEAD_BYTES].decode("utf-8", "replace")


def inspect(archive, bundle="swimxyz_sample.zip", log=print):
    """Summarise an archive's layout and write a small sample bundle. -> the listing text."""
    lines = []
    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        by_ext = Counter(Path(i.filename).suffix.lower() or "(none)" for i in infos)
        lines.append(f"{Path(archive).name}: {len(infos)} files, {sum(i.file_size for i in infos) / 1e9:.2f} GB unpacked")
        lines.append("by type: " + ", ".join(f"{e} x{n}" for e, n in by_ext.most_common()))
        dirs = Counter("/".join(Path(i.filename).parts[:-1][:4]) for i in infos)
        lines.append(f"\nfolders (first 4 levels), {len(dirs)} in all:")
        lines += [f"  {d or '.'}  ({n} files)" for d, n in sorted(dirs.items())[:60]]
        lines.append("\nfirst 40 files:")
        lines += [f"  {i.filename}  ({i.file_size} bytes)" for i in infos[:40]]

        samples, taken = {}, Counter()
        for i in infos:
            ext = Path(i.filename).suffix.lower()
            if ext not in TEXT_EXT | ARRAY_EXT or taken[ext] >= SAMPLES_PER_KIND:
                continue
            if ext in ARRAY_EXT and i.file_size > 50_000_000:
                continue
            taken[ext] += 1
            with z.open(i) as f:
                data = f.read() if ext in ARRAY_EXT or ext == ".json" and i.file_size < 20_000_000 else f.read(HEAD_BYTES)
            samples[i.filename] = _describe(i.filename, data)
    listing = "\n".join(lines)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("listing.txt", listing)
        for name, text in samples.items():
            out.writestr("samples/" + name.replace("/", "__") + ".txt", f"{name}\n\n{text}")
    log(listing)
    log(f"\nWrote {bundle} ({os.path.getsize(bundle) / 1e3:.0f} KB, {len(samples)} sample files). Send that file back.")
    return listing


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch and inspect the freestyle part of SwimXYZ.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download archives (resumable)")
    f.add_argument("what", nargs="+", choices=sorted(FILES))
    f.add_argument("--out", default="swimxyz")
    i = sub.add_parser("inspect", help="summarise an archive and write a small sample bundle")
    i.add_argument("archive")
    i.add_argument("--bundle", default="swimxyz_sample.zip")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        for key in a.what:
            fetch(key, a.out)
    else:
        inspect(a.archive, a.bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
