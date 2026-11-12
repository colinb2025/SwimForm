"""SwimXYZ fetch and inspect, against a stand-in server and a made-up archive."""
import io
import json
import zipfile

import numpy as np
import pytest

from swimform import swimxyz as X


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status, self.headers = status, {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_fetch_downloads_resumes_and_skips(tmp_path, monkeypatch):
    body = b"swim" * 5000
    monkeypatch.setitem(X.FILES, "labels", ("1", "Freestyle_labels.zip", X.hashlib.md5(body).hexdigest()))
    seen = []

    def server(req):
        rng = req.headers.get("Range")
        seen.append(rng)
        if rng:
            return FakeResponse(body[int(rng.split("=")[1].rstrip("-")):], status=206)
        return FakeResponse(body)

    (tmp_path / "Freestyle_labels.zip.part").write_bytes(body[:7000])        # an interrupted download
    dest = X.fetch("labels", tmp_path, opener=server, log=lambda *_: None)
    assert dest.read_bytes() == body and seen == ["bytes=7000-"]
    X.fetch("labels", tmp_path, opener=server, log=lambda *_: None)
    assert len(seen) == 1                                                      # already there: no request


def test_fetch_refuses_a_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setitem(X.FILES, "side", ("1", "Freestyle_part2.zip", "0" * 32))
    with pytest.raises(SystemExit):
        X.fetch("side", tmp_path, opener=lambda req: FakeResponse(b"junk"), log=lambda *_: None)
    assert not (tmp_path / "Freestyle_part2.zip").exists()


def test_inspect_summarises_and_bundles_small_samples(tmp_path):
    archive = tmp_path / "Freestyle_labels.zip"
    arr = io.BytesIO()
    np.save(arr, np.arange(17 * 3 * 4, dtype=np.float32).reshape(4, 17, 3))
    with zipfile.ZipFile(archive, "w") as z:
        for view in ("Front", "Side_above_water"):
            for n in range(5):
                z.writestr(f"Freestyle/{view}/seq{n}/COCO17/2D_cam.csv", "frame,nose_x,nose_y\n" + "0,1.0,2.0\n" * 5000)
                z.writestr(f"Freestyle/{view}/seq{n}/meta.json", json.dumps({"fps": 60, "joints": ["nose", "l_eye"]}))
        z.writestr("Freestyle/Front/seq0/joints3d.npy", arr.getvalue())
    bundle = tmp_path / "sample.zip"
    listing = X.inspect(archive, bundle, log=lambda *_: None)
    assert "21 files" in listing and ".csv x10" in listing and "Freestyle/Side_above_water/seq0/COCO17" in listing
    assert bundle.stat().st_size < 100_000
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        assert "listing.txt" in names and len(names) == 1 + 3 + 3 + 1          # three per kind, at most
        text = {n: z.read(n).decode() for n in names}
    csv_sample = next(v for k, v in text.items() if k.endswith(".csv.txt"))
    assert "frame,nose_x,nose_y" in csv_sample and len(csv_sample) < 5000      # only the head of big files
    assert "'fps': 60" in next(v for k, v in text.items() if k.endswith(".json.txt"))
    assert "shape (4, 17, 3)" in next(v for k, v in text.items() if k.endswith(".npy.txt"))
