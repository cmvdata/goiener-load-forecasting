"""
00_download_goiener.py — Resilient download of the GoiEner Zenodo archives.

Residential connections drop mid-download for files this size; a naive
requests.get is unsafe. This script:

  - HEADs each remote file to learn its canonical size
  - Skips local files whose size already matches the canonical one exactly
  - Uses Range: bytes=N- to resume partial downloads
  - Retries up to 6 times with exponential backoff (5,10,20,40,80,160s)
  - Treats HTTP 416 (range not satisfiable) as success — file is complete
  - Streams in chunks with a tqdm progress bar

Sample mode is a no-op; the synthetic sample lives in data/sample/ and
needs no Zenodo download.

Files downloaded into data/raw/:
  metadata.csv      ~5.6 MB
  imp-pre.tzst      ~792 MB
  imp-in.tzst       ~530 MB
  imp-post.tzst     ~510 MB
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

import config as C


ZENODO_BASE = "https://zenodo.org/records/7362094/files"
FILES = ["metadata.csv", "imp-pre.tzst", "imp-in.tzst", "imp-post.tzst"]

CHUNK_BYTES = 1024 * 1024  # 1 MB
HEAD_TIMEOUT = 30
GET_TIMEOUT = 120
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 5


def head_size(url: str) -> int | None:
    """Return remote content length in bytes, or None if not advertised."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"  [warn] HEAD failed for {url}: {e}")
        return None

    cl = r.headers.get("Content-Length")
    if cl is None:
        return None
    try:
        return int(cl)
    except ValueError:
        return None


def stream_download(url: str, dest: Path, expected_size: int | None) -> None:
    """Download with resume + retry. Raises on terminal failure."""
    last_exc: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        local_size = dest.stat().st_size if dest.exists() else 0

        if expected_size is not None and local_size == expected_size:
            print(f"  [ok] already complete: {dest.name} ({local_size:,} bytes)")
            return

        if expected_size is not None and local_size > expected_size:
            print(f"  [warn] local file larger than remote ({local_size} > "
                  f"{expected_size}); discarding")
            dest.unlink()
            local_size = 0

        headers = {}
        if local_size > 0:
            headers["Range"] = f"bytes={local_size}-"

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=GET_TIMEOUT,
                allow_redirects=True,
            ) as r:
                if r.status_code == 416:
                    # Range not satisfiable → file is already complete on disk
                    print(f"  [ok] HTTP 416 — treating as complete: {dest.name}")
                    return

                r.raise_for_status()

                total = expected_size
                if total is None:
                    cl = r.headers.get("Content-Length")
                    total = (local_size + int(cl)) if cl is not None else None

                mode = "ab" if local_size > 0 else "wb"
                with open(dest, mode) as fh, tqdm(
                    desc=dest.name,
                    total=total,
                    initial=local_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bar.update(len(chunk))

            new_size = dest.stat().st_size
            if expected_size is None or new_size == expected_size:
                print(f"  [ok] downloaded {dest.name} ({new_size:,} bytes)")
                return

            print(f"  [warn] size mismatch after attempt {attempt}: "
                  f"{new_size} / {expected_size}; retrying")

        except Exception as e:
            last_exc = e
            print(f"  [retry {attempt}/{MAX_ATTEMPTS}] {dest.name}: {e}")

        backoff = BACKOFF_BASE_S * (2 ** (attempt - 1))
        if attempt < MAX_ATTEMPTS:
            print(f"  sleeping {backoff}s before next attempt")
            time.sleep(backoff)

    raise RuntimeError(
        f"Download of {dest.name} failed after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_exc}"
    )


def main():
    if C.DATA_MODE == "sample":
        print("[sample] No download needed; synthetic data lives in data/sample/.")
        return

    raw_dir = C.DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading GoiEner archives → {raw_dir}\n")

    for fname in FILES:
        url = f"{ZENODO_BASE}/{fname}"
        dest = raw_dir / fname
        size = head_size(url)
        if size is not None:
            print(f"\n{fname}: remote {size:,} bytes")
        else:
            print(f"\n{fname}: remote size unknown")
        stream_download(url, dest, size)

    print("\nFinal sizes:")
    for fname in FILES:
        p = raw_dir / fname
        if p.exists():
            mb = p.stat().st_size / 1024 / 1024
            print(f"  {fname}: {mb:.1f} MB")
    print("\n[ok] All archives downloaded.")


if __name__ == "__main__":
    C._print_summary()
    main()
