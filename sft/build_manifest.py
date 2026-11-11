"""
build_manifest.py — Phase 2.1
Scans the protected set and writes sft/protected_manifest.json.

Protected set definition (repo-relative paths / subtrees):
  data/                           — all files recursively
  notebooks/                      — all files recursively
  scripts/                        — all files recursively
  requirements.txt                — root dependency manifest

Files >= 1 GiB and any checkpoint/weight file → hashed=false (size+mtime only).
Checkpoint/weight extensions: .safetensors .bin .pt .pth .ckpt .gguf .h5 .hdf5

Run from any working directory; the repo root is inferred from this file's location
(one level up from sft/).

Usage:
    python sft/build_manifest.py
    python sft/build_manifest.py --dry-run   # count files without writing
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
WEIGHT_EXTENSIONS = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".hdf5"}
)
GIB = 1024 ** 3
HASH_LIMIT_BYTES = GIB          # files larger than this → hashed=false
MAX_FILES = 5_000               # stop and report if exceeded

# Protected subtrees and individual files (repo-relative)
PROTECTED_SUBTREES = ["data", "notebooks", "scripts"]
PROTECTED_FILES    = ["requirements.txt"]

OUTPUT_PATH = Path(__file__).parent / "protected_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip(path: Path) -> bool:
    """Skip .git, venv, sft/ and __pycache__ trees."""
    parts = set(path.parts)
    skip_names = {".git", "venv", ".venv", "sft", "__pycache__", ".ipynb_checkpoints"}
    return bool(parts & skip_names)


def collect_protected(repo_root: Path) -> list[Path]:
    """Return all files in the protected set."""
    result: list[Path] = []

    for subtree_name in PROTECTED_SUBTREES:
        subtree = repo_root / subtree_name
        if not subtree.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(subtree):
            # prune unwanted dirs in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".ipynb_checkpoints"}
            ]
            for fname in sorted(filenames):
                fp = Path(dirpath) / fname
                result.append(fp)

    for rel in PROTECTED_FILES:
        fp = repo_root / rel
        if fp.exists():
            result.append(fp)

    return sorted(set(result))


def build_entry(fp: Path, repo_root: Path) -> dict:
    st = fp.stat()
    rel = fp.relative_to(repo_root).as_posix()
    is_weight = fp.suffix.lower() in WEIGHT_EXTENSIONS
    too_large  = st.st_size >= HASH_LIMIT_BYTES

    entry: dict = {
        "path":   rel,
        "size":   st.st_size,
        "mtime":  st.st_mtime,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "hashed": False,
        "sha256": None,
    }

    if is_weight:
        entry["skip_reason"] = "weight_file"
    elif too_large:
        entry["skip_reason"] = f"size_ge_1gib ({st.st_size // GIB:.1f} GiB)"
    else:
        entry["hashed"] = True
        entry["sha256"] = sha256_file(fp)
        entry.pop("skip_reason", None)

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protected-artifact manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without hashing or writing")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repo root : {repo_root}")
    print(f"Output    : {OUTPUT_PATH}")

    files = collect_protected(repo_root)
    print(f"Files found: {len(files)}")

    if len(files) > MAX_FILES:
        print(
            f"ERROR: {len(files)} files exceed the {MAX_FILES}-file safety limit. "
            "Stopping without writing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        for f in files:
            print(" ", f.relative_to(repo_root).as_posix())
        print(f"\nDry-run complete. {len(files)} files would be processed.")
        return

    entries: list[dict] = []
    unhashed = 0
    for i, fp in enumerate(files, 1):
        entry = build_entry(fp, repo_root)
        entries.append(entry)
        if not entry["hashed"]:
            unhashed += 1
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {entry['path']}")

    manifest = {
        "schema_version": "1",
        "created_utc":    datetime.now(tz=timezone.utc).isoformat(),
        "repo_root":      repo_root.as_posix(),
        "total_files":    len(entries),
        "hashed_count":   len(entries) - unhashed,
        "unhashed_count": unhashed,
        "entries":        entries,
    }

    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nManifest written -> {OUTPUT_PATH}")
    print(f"  total={len(entries)}  hashed={len(entries)-unhashed}  unhashed={unhashed}")


if __name__ == "__main__":
    main()
