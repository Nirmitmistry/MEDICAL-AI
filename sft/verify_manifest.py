"""
verify_manifest.py — Phase 2.2
stdlib-only verifier for the protected-artifact manifest.

Exits 0  if all protected files match the manifest exactly.
Exits 1  if any file is MISSING or its size/SHA-256 has CHANGED.
Prints a WARNING (but still exits 0 or 1 based on the above) for any
NEW files found inside protected directories that are not in the manifest.

Never writes to protected paths.  No third-party dependencies.

Usage:
    python sft/verify_manifest.py [--manifest <path>]
    python sft/verify_manifest.py --help
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# ── constants ────────────────────────────────────────────────────────────────
WEIGHT_EXTENSIONS = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".hdf5"}
)
GIB = 1024 ** 3
DEFAULT_MANIFEST = Path(__file__).parent / "protected_manifest.json"

# Protected subtrees — used to detect NEW files
PROTECTED_SUBTREES = ["data", "notebooks", "scripts"]
PROTECTED_FILES    = ["requirements.txt"]


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_current_files(repo_root: Path) -> set[str]:
    """Return repo-relative POSIX paths of all files in the protected set."""
    result: set[str] = set()
    for subtree_name in PROTECTED_SUBTREES:
        subtree = repo_root / subtree_name
        if not subtree.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(subtree):
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".ipynb_checkpoints"}
            ]
            for fname in filenames:
                fp = Path(dirpath) / fname
                result.add(fp.relative_to(repo_root).as_posix())
    for rel in PROTECTED_FILES:
        if (repo_root / rel).exists():
            result.add(rel)
    return result


def verify_entry(entry: dict, repo_root: Path) -> tuple[str | None, str]:
    """
    Return (status, detail).
    status: None = ok, "MISSING" = file gone, "CHANGED" = hash/size mismatch.
    """
    rel  = entry["path"]
    fp   = repo_root / rel

    if not fp.exists():
        return "MISSING", f"File not found: {rel}"

    st = fp.stat()

    if not entry["hashed"]:
        # size-only check (large files / weights)
        if st.st_size != entry["size"]:
            return "CHANGED", (
                f"Size mismatch: {rel}  "
                f"expected={entry['size']}  actual={st.st_size}"
            )
        return None, "ok (size-only)"

    # full SHA-256 check
    if st.st_size != entry["size"]:
        return "CHANGED", (
            f"Size mismatch: {rel}  "
            f"expected={entry['size']}  actual={st.st_size}"
        )

    actual_hash = sha256_file(fp)
    if actual_hash != entry["sha256"]:
        return "CHANGED", (
            f"Hash mismatch: {rel}  "
            f"expected={entry['sha256'][:16]}…  actual={actual_hash[:16]}…"
        )

    return None, "ok"


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify protected-artifact manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to the manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parent.parent
    entries   = manifest["entries"]

    print(f"Manifest  : {manifest_path}")
    print(f"Created   : {manifest.get('created_utc', 'unknown')}")
    print(f"Repo root : {repo_root}")
    print(f"Entries   : {manifest['total_files']}")
    print()

    # ── 1. check every manifest entry ────────────────────────────────────────
    errors: list[str] = []
    ok_count = 0
    for entry in entries:
        status, detail = verify_entry(entry, repo_root)
        if status is None:
            ok_count += 1
        else:
            errors.append(f"[{status}] {detail}")

    # ── 2. detect NEW files in protected dirs ─────────────────────────────────
    manifest_paths = {e["path"] for e in entries}
    current_paths  = collect_current_files(repo_root)
    new_paths      = current_paths - manifest_paths

    warnings: list[str] = []
    for p in sorted(new_paths):
        warnings.append(f"[WARNING] New unmanifested file: {p}")

    # ── report ────────────────────────────────────────────────────────────────
    if warnings:
        for w in warnings:
            print(w)
        print()

    if errors:
        for e in errors:
            print(e)
        print(
            f"\nVERIFICATION FAILED: {len(errors)} error(s), "
            f"{ok_count} ok, {len(warnings)} warning(s)."
        )
        sys.exit(1)

    print(
        f"VERIFICATION PASSED: {ok_count} file(s) ok, "
        f"{len(warnings)} warning(s)."
    )


if __name__ == "__main__":
    main()
