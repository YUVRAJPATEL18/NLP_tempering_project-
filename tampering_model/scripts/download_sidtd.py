#!/usr/bin/env python3
"""Acquire the SIDTD image subset used by IDGuard's tampering model.

What this does (idempotent, resumable):
  1. Downloads SIDTD ``templates.zip`` (1.27 GB) from figshare — the *only* file from the
     79 GB SIDTD corpus we need (videos/clips are deliberately skipped).
  2. Verifies the published MD5.
  3. Extracts ``templates/Images/**`` and ``templates/Annotations/**`` only.
  4. Downloads the official split CSVs (< 1 MB) for the record.
  5. Writes ``datasets/sidtd/manifest.csv``: per-image label, base-document group id,
     manipulation type and tamper source — everything the dataset pipeline needs.

Usage:
    .venv/bin/python scripts/download_sidtd.py [--skip-download]

Never fabricates URLs: the figshare file URL comes from the public figshare API record
for DOI 10.6084/m9.figshare.27136242.v1 (article id 27136242).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import shutil
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "sidtd"

FIGSHARE_API = "https://api.figshare.com/v2/articles/27136242"
TEMPLATES_NAME = "templates.zip"
TEMPLATES_MD5 = "a790947e773686a2f1b2ab35be0ae38d"  # published by figshare

# Official split archives (tiny, kept for reference only — see audit for leakage finding)
SPLIT_FILES = {
    "49510854": "hold_out_split.zip",
    "49510851": "kfold_split.zip",
    "49510857": "unbalanced_hold_out_split.zip",
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def _seed_parts_from_partial(dest: Path, parts: Path, bounds: list) -> None:
    """Salvage a sequential partial download (e.g. interrupted curl) into chunk files."""
    if not dest.exists():
        return
    have = dest.stat().st_size
    if have <= 0:
        return
    with dest.open("rb") as f:
        for idx, (a, b) in enumerate(bounds):
            seg = min(have, b + 1) - a
            if seg <= 0:
                break
            pa = parts / f"part{idx}"
            if pa.exists() and pa.stat().st_size >= seg:
                continue
            tmp = pa.with_suffix(".seed")
            with tmp.open("wb") as g:
                left = seg
                while left:
                    blk = f.read(min(1 << 20, left))
                    if not blk:
                        break
                    g.write(blk)
                    left -= len(blk)
            tmp.replace(pa)
    dest.unlink()
    log(f"salvaged {have} bytes of a previous partial download into chunks")


def _fetch_part(url: str, a: int, b: int, pa: Path, stop: threading.Event) -> None:
    attempts = 0
    while not stop.is_set():
        if pa.exists():
            cur = pa.stat().st_size
        else:
            cur = 0
        total = b - a + 1
        if cur > total:
            # over-run guard: truncate and start the range again
            with pa.open("r+b") as g:
                g.truncate(total)
            cur = total
        if cur == total:
            return
        attempts += 1
        if attempts > 40:
            raise RuntimeError(f"chunk {pa.name} failed after {attempts} attempts")
        start = a + cur
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{b}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, pa.open("ab") as g:
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    g.write(blk)
        except Exception:
            time.sleep(min(2 ** (attempts % 5), 8))
            continue
        if pa.stat().st_size == cur:
            # no progress on this attempt (stalled body) — retry with a fresh redirect
            time.sleep(2)


def download_chunked(url: str, dest: Path, size: int, workers: int = 6) -> None:
    parts = dest.parent / (dest.name + ".parts")
    parts.mkdir(exist_ok=True)
    bounds = [(i * size // workers, (i + 1) * size // workers - 1) for i in range(workers)]
    _seed_parts_from_partial(dest, parts, bounds)

    stop = threading.Event()
    errors: list = []

    def run(idx: int, a: int, b: int) -> None:
        try:
            _fetch_part(url, a, b, parts / f"part{idx}", stop)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            stop.set()

    t0 = time.time()
    threads = [threading.Thread(target=run, args=(i, a, b), daemon=True)
               for i, (a, b) in enumerate(bounds)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(10)
            got = sum(p.stat().st_size for p in parts.glob("part*"))
            dt = max(time.time() - t0, 1)
            eta = (size - got) / (got / dt) if got else 0
            log(f"{got / 1e6:.0f}/{size / 1e6:.0f} MB  {got / dt / 1e3:.0f} KB/s  eta {eta / 60:.1f} min")
    except KeyboardInterrupt:  # pragma: no cover
        stop.set()
        raise
    for t in threads:
        t.join()
    if errors:
        raise SystemExit(f"download failed: {errors[0]}")

    tmp = dest.with_suffix(".assembling")
    with tmp.open("wb") as out:
        for i in range(workers):
            with (parts / f"part{i}").open("rb") as src:
                shutil.copyfileobj(src, out, 1 << 20)
    tmp.replace(dest)
    shutil.rmtree(parts)
    log(f"assembled {dest} ({dest.stat().st_size} bytes)")


def fetch_templates_zip(skip_download: bool, workers: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / TEMPLATES_NAME
    if dest.exists() and md5(dest) == TEMPLATES_MD5:
        log(f"{TEMPLATES_NAME} already present and verified ({dest.stat().st_size} bytes)")
        return dest
    if skip_download:
        raise SystemExit(f"--skip-download given but {dest} is missing or corrupt")

    with urllib.request.urlopen(FIGSHARE_API, timeout=30) as r:
        record = json.load(r)
    matches = [f for f in record["files"] if f["name"] == TEMPLATES_NAME]
    if not matches:
        raise SystemExit("templates.zip not found in figshare record — dataset layout changed")
    meta = matches[0]
    if meta["supplied_md5"] != TEMPLATES_MD5:
        log(f"WARNING: published md5 changed to {meta['supplied_md5']}; using published value")
    if dest.exists() and dest.stat().st_size >= meta["size"]:
        log("existing file has unexpected content — removing and re-downloading")
        dest.unlink()
    elif dest.exists():
        log(f"resuming from partial download ({dest.stat().st_size} bytes)")
    log(f"downloading {TEMPLATES_NAME} ({meta['size']} bytes, {workers} connections) "
        f"from figshare article {record['id']}")

    download_chunked(meta["download_url"], dest, meta["size"], workers=workers)

    digest = md5(dest)
    if digest != meta["supplied_md5"]:
        dest.unlink()
        raise SystemExit(f"MD5 mismatch (got {digest}, want {meta['supplied_md5']}); file removed, re-run to retry")
    log("md5 verified")
    return dest


def extract(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if m.startswith(("templates/Images/", "templates/Annotations/"))
                   and not m.endswith("/")]
        log(f"extracting {len(members)} members")
        for m in members:
            z.extract(m, OUT)


def fetch_split_archives() -> None:
    splits_dir = OUT / "official_splits"
    splits_dir.mkdir(exist_ok=True)
    for fid, name in SPLIT_FILES.items():
        dest = splits_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://ndownloader.figshare.com/files/{fid}"
        log(f"fetching official split archive {name}")
        with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
            f.write(r.read())
    import zipfile as zf
    for name in SPLIT_FILES.values():
        with zf.ZipFile(splits_dir / name) as z:
            for m in z.namelist():
                if m.endswith(".csv") and not m.startswith("__MACOSX"):
                    z.extract(m, splits_dir)


def build_manifest() -> None:
    images_root = OUT / "templates" / "Images"
    annot_root = OUT / "templates" / "Annotations"
    rows = []

    fake_meta = {}
    for p in sorted((annot_root / "fakes").glob("*.json")):
        fake_meta[p.stem] = json.loads(p.read_text())

    def group_of(stem: str) -> str:
        return re.split(r"_fake_", stem)[0]

    for label_name, label in (("reals", 0), ("fakes", 1)):
        for img in sorted((images_root / label_name).glob("*.jpg")):
            stem = img.stem
            meta = fake_meta.get(stem, {})
            src = meta.get("src", "")
            if label == 1:
                src_path = f"templates/Images/reals/{src}"
            else:
                src_path = f"templates/Images/reals/{img.name}"
            rows.append({
                "image_path": str(img.relative_to(OUT)).replace("\\", "/"),
                "label": label,
                "label_name": label_name,
                "group_id": group_of(stem),
                "nationality": stem.split("_")[0],
                "manipulation_type": meta.get("ctype", ""),
                "tamper_field": meta.get("field", ""),
                "src_image": src_path if (label == 1 and src) else "",
            })

    # integrity: every faked source must exist among reals
    real_names = {r["image_path"] for r in rows if r["label"] == 0}
    bad = [r for r in rows if r["label"] == 1 and r["src_image"] not in real_names]

    out_csv = OUT / "manifest.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = len(rows) - n_real
    groups = len({r["group_id"] for r in rows})
    log(f"manifest: {len(rows)} images ({n_real} genuine / {n_fake} tampered), "
        f"{groups} base-document groups -> {out_csv}")
    if bad:
        log(f"WARNING: {len(bad)} fakes reference a missing source image "
            f"(first: {bad[0]['src_image']})")
    ctype = {}
    for r in rows:
        if r["label"] == 1:
            ctype[r["manipulation_type"]] = ctype.get(r["manipulation_type"], 0) + 1
    log(f"manipulation types: {ctype}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true",
                    help="use an already-downloaded templates.zip")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel connections for the chunked download")
    args = ap.parse_args()

    zip_path = fetch_templates_zip(args.skip_download, args.workers)
    extract(zip_path)
    fetch_split_archives()
    build_manifest()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
