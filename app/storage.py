"""
Where clips and reports live once an analysis has been processed.

SWIMFORM_STORAGE
  local  (default) the SWIMFORM_DATA folder. Fine on one server with a persistent disk.
  s3     any S3-compatible bucket: Amazon S3, Cloudflare R2, Backblaze B2. Use this on
         hosts whose disk is wiped on every deploy.
         S3_BUCKET            bucket name
         S3_ENDPOINT_URL      only for R2/B2/etc; leave unset for Amazon S3
         AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (R2: "auto")

Uploads always land on local disk first, because the pose model reads video files.
When the job is finished the folder is copied to the bucket and the local clips are
removed; the report stays cached locally and is fetched again if the disk was wiped.
"""
import logging
import os

log = logging.getLogger("swimform.storage")


def backend():
    return "s3" if os.environ.get("SWIMFORM_STORAGE", "local").strip().lower() == "s3" else "local"


def _client():
    """Separate so tests can stand in for the bucket."""
    import boto3   # optional dependency, only needed for this backend
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None)


def _bucket():
    return os.environ["S3_BUCKET"]


def _key(aid, name):
    return f"analyses/{aid}/{name}"


def offload(folder, aid):
    """Copy an analysis folder to the bucket, then drop the local clips."""
    if backend() != "s3" or not folder.exists():
        return
    client = _client()
    for f in sorted(folder.iterdir()):
        if f.is_file():
            client.upload_file(str(f), _bucket(), _key(aid, f.name))
    for f in folder.iterdir():
        if f.is_file() and f.suffix != ".json":
            f.unlink()


def fetch(folder, aid, name):
    """Make sure folder/name exists locally, downloading it if needed. -> path or None."""
    path = folder / name
    if path.exists():
        return path
    if backend() != "s3":
        return None
    folder.mkdir(parents=True, exist_ok=True)
    try:
        _client().download_file(_bucket(), _key(aid, name), str(path))
    except Exception:   # noqa: BLE001 - missing object, or the bucket is unreachable
        log.warning("could not fetch %s from storage", _key(aid, name))
        path.unlink(missing_ok=True)
        return None
    return path


def remove(aid):
    """Delete everything stored for one analysis from the bucket."""
    if backend() != "s3":
        return
    client = _client()
    listing = client.list_objects_v2(Bucket=_bucket(), Prefix=_key(aid, ""))
    keys = [{"Key": o["Key"]} for o in listing.get("Contents", [])]
    if keys:
        client.delete_objects(Bucket=_bucket(), Delete={"Objects": keys})
