"""Download a LibriSpeech subset, resumably, over a flaky connection.

    python -m s2s.data.download --config configs/codec.yaml
    python -m s2s.data.download --config configs/codec.yaml subset=train-clean-360

Unlike torchaudio's downloader (which restarts from zero on any interruption), this
streams the .tar.gz with HTTP Range requests: every interruption just resumes from
the bytes already on disk. It retries with backoff, falls back across OpenSLR mirrors,
verifies the archive, extracts to <data_root>/LibriSpeech/<subset>, and can delete the
archive afterwards. Re-running is always safe — it skips what's already done.
"""
from __future__ import annotations

import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

from ..utils import load_config

MIRRORS = [
    "https://www.openslr.org/resources/12",
    "https://us.openslr.org/resources/12",
    "https://openslr.elda.org/resources/12",
]
SUBSETS = ("dev-clean", "dev-other", "test-clean", "test-other",
           "train-clean-100", "train-clean-360", "train-other-500")


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _total_size(url: str) -> int | None:
    """Content-Length of the full file, or None if the server won't say."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_resumable(url: str, dest: str, retries: int = 10000,
                       chunk: int = 1 << 20) -> bool:
    """Stream url -> dest, resuming from the partial file on every retry.
    Returns True when the file is fully on disk."""
    total = _total_size(url)
    tmp = dest + ".part"
    last_err = None
    for attempt in range(retries):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and have >= total:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                # 200 = server ignored Range -> restart from scratch
                mode = "ab"
                if r.status == 200 and have:
                    have, mode = 0, "wb"
                if total is None:
                    cr = r.headers.get("Content-Range")
                    total_local = int(cr.split("/")[-1]) if cr else \
                        (int(r.headers.get("Content-Length", 0)) + have or None)
                else:
                    total_local = total
                with open(tmp, mode) as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        pct = f"{100*have/total_local:.1f}%" if total_local else "?"
                        sys.stdout.write(f"\r  {human(have)} / "
                                         f"{human(total_local) if total_local else '?'} ({pct})   ")
                        sys.stdout.flush()
            sys.stdout.write("\n")
            if total_local and have >= total_local:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            wait = min(30, 2 ** min(attempt, 5))
            sys.stdout.write(f"\n  interrupted ({e}); resuming from {human(have)} in {wait}s...\n")
            time.sleep(wait)
    else:
        raise RuntimeError(f"gave up after {retries} attempts (last error: {last_err})")
    os.replace(tmp, dest)
    return True


def verify_archive(path: str) -> bool:
    """A truncated/corrupt .tar.gz fails to open or list -> caller re-downloads."""
    try:
        with tarfile.open(path, "r:gz") as t:
            t.next()                      # read the first member's header
        return True
    except Exception:
        return False


def main():
    cfg = load_config("configs/codec.yaml")
    data_root = cfg.get("data_root", "data")
    subset = cfg.get("subset", "train-clean-100")
    assert subset in SUBSETS, f"unknown subset {subset!r}; one of {SUBSETS}"
    keep_archive = cfg.get("keep_archive", False)

    extracted = os.path.join(data_root, "LibriSpeech", subset)
    if os.path.isdir(extracted) and os.listdir(extracted):
        print(f"{subset} already extracted at {extracted} — nothing to do.")
        return

    os.makedirs(data_root, exist_ok=True)
    archive = os.path.join(data_root, f"{subset}.tar.gz")

    # download (resumably) and verify, trying each mirror; re-download if corrupt
    for pass_i in range(3):
        if not (os.path.exists(archive) and verify_archive(archive)):
            ok = False
            for base in MIRRORS:
                url = f"{base}/{subset}.tar.gz"
                print(f"downloading {subset} from {url}")
                try:
                    download_resumable(url, archive)
                    ok = True
                    break
                except Exception as e:
                    print(f"  mirror failed: {e}")
            if not ok:
                raise SystemExit("all mirrors failed; check your connection and re-run "
                                 "(it will resume from the partial file).")
        print("verifying archive...")
        if verify_archive(archive):
            break
        print("archive is corrupt — deleting and re-downloading.")
        os.remove(archive)
    else:
        raise SystemExit("archive stayed corrupt after 3 tries.")

    print(f"extracting to {os.path.join(data_root, 'LibriSpeech')} ...")
    with tarfile.open(archive, "r:gz") as t:
        t.extractall(data_root)           # tar contains LibriSpeech/<subset>/...
    print("extracted.")

    if not keep_archive:
        os.remove(archive)
        print("removed archive to save space (set keep_archive=true to keep it).")
    print(f"done — audio at {extracted}")


if __name__ == "__main__":
    main()