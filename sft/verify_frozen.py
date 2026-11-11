"""
sft/verify_frozen.py — verifies every sft/frozen_manifest*.json it finds (C5)

stdlib-only. Mirrors the logic of verify_manifest.py but scans for all
frozen_manifest*.json files under the sft/ directory.

Usage
-----
python sft/verify_frozen.py            # run from repo root
python sft/verify_frozen.py --help

Exit codes
----------
0  all manifests verify (or none found — prints a warning)
1  at least one entry in at least one manifest failed
2  a manifest file itself is malformed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_entry(entry: dict, repo_root: Path) -> tuple[str | None, str]:
    """
    Returns (error_code, detail) where error_code is None on success.
    error_codes: 'MISSING', 'CHANGED', 'SIZE_MISMATCH'
    """
    rel  = entry['path']
    fp   = repo_root / rel

    if not fp.exists():
        return 'MISSING', f'{rel}: file not found'

    expected_size = entry.get('size')
    if expected_size is not None and fp.stat().st_size != expected_size:
        return 'SIZE_MISMATCH', (
            f'{rel}: size {fp.stat().st_size} != expected {expected_size}'
        )

    expected_sha = entry.get('sha256')
    if expected_sha:
        actual_sha = sha256_file(fp)
        if actual_sha != expected_sha:
            return 'CHANGED', (
                f'{rel}: sha256 {actual_sha[:16]}... != expected {expected_sha[:16]}...'
            )

    return None, rel


def verify_manifest(manifest_path: Path, repo_root: Path) -> tuple[bool, list[str]]:
    """
    Verify one frozen manifest file.
    Returns (all_ok, list_of_error_messages).
    """
    try:
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f'[MALFORMED] {manifest_path}: {e}']

    entries = manifest.get('entries', [])
    errors: list[str] = []
    ok_count = 0

    for entry in entries:
        status, detail = verify_entry(entry, repo_root)
        if status is None:
            ok_count += 1
        else:
            errors.append(f'[{status}] {detail}')

    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Verify all sft/frozen_manifest*.json files.',
    )
    parser.add_argument(
        '--repo-root',
        default='.',
        help='Repo root directory (default: current directory)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-file OK messages; only print failures.',
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    sft_dir   = repo_root / 'sft'

    manifests = sorted(sft_dir.glob('frozen_manifest*.json'))
    if not manifests:
        print(
            'WARNING: no frozen_manifest*.json files found under sft/. '
            'Nothing to verify.',
            file=sys.stderr,
        )
        return 0

    overall_ok = True
    for mp in manifests:
        ok, errors = verify_manifest(mp, repo_root)
        if ok:
            if not args.quiet:
                entries_count = 0
                try:
                    with open(mp, encoding='utf-8') as f:
                        data = json.load(f)
                    entries_count = len(data.get('entries', []))
                except Exception:
                    pass
                print(f'PASS  {mp.name}  ({entries_count} entries)')
        else:
            overall_ok = False
            print(f'FAIL  {mp.name}')
            for err in errors:
                print(f'  {err}')

    if overall_ok:
        print(f'FROZEN VERIFICATION PASSED: {len(manifests)} manifest(s) ok.')
    else:
        print(f'FROZEN VERIFICATION FAILED.', file=sys.stderr)

    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
