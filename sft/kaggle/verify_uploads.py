"""
sft/kaggle/verify_uploads.py — A6: Verify uploaded Kaggle dataset files against SHA-256.

Stdlib-only. Run before training to confirm all uploaded files are intact.

Usage:
    python sft/kaggle/verify_uploads.py --manifest sft/kaggle/upload_manifest.json
    python sft/kaggle/verify_uploads.py --manifest sft/kaggle/upload_manifest.json --root /kaggle/input

Exit codes: 0 all files ok, 1 at least one mismatch or missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: Path, root: Path) -> tuple[bool, list[str]]:
    """
    Returns (all_ok, list_of_error_messages).
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("files", [])
    errors: list[str] = []
    ok_count = 0

    for entry in entries:
        rel = entry["path"]
        expected_sha = entry.get("sha256", "")
        expected_size = entry.get("size_bytes", None)

        fp = root / rel
        if not fp.exists():
            errors.append(f"[MISSING] {rel}")
            continue

        if expected_size is not None:
            actual_size = fp.stat().st_size
            if actual_size != expected_size:
                errors.append(
                    f"[SIZE_MISMATCH] {rel}: got {actual_size}, expected {expected_size}"
                )
                continue

        if expected_sha:
            actual_sha = sha256_file(fp)
            if actual_sha != expected_sha:
                errors.append(
                    f"[SHA_MISMATCH] {rel}: got {actual_sha[:16]}..., "
                    f"expected {expected_sha[:16]}..."
                )
                continue

        ok_count += 1
        print(f"  OK  {rel}")

    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Kaggle upload file integrity against SHA-256 manifest.",
    )
    parser.add_argument(
        "--manifest",
        default="sft/kaggle/upload_manifest.json",
        help="Path to upload_manifest.json (default: sft/kaggle/upload_manifest.json)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to resolve relative paths against (default: .)",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    root = Path(args.root).resolve()

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    print(f"Verifying uploads from {manifest_path} (root: {root})...")
    ok, errors = verify_manifest(manifest_path, root)

    if ok:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
        n = len(m.get("files", []))
        print(f"VERIFICATION PASSED: {n} file(s) ok.")
        return 0
    else:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"VERIFICATION FAILED: {len(errors)} error(s).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
