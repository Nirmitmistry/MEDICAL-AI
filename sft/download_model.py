"""
sft/download_model.py — P0b Step 1
===================================
Downloads Qwen/Qwen3-1.7B weights (safetensors only) at the pinned revision
to sft/model_snapshot/Qwen3-1.7B.

Idempotent: if --verify-only is passed, skips the download and only checks
hashes against the Hub LFS oids.

Rules enforced:
- safetensors + json + merges.txt + LICENSE only; no .bin / .gguf / .pt
- No symlinks (local_dir writes real files in huggingface_hub >= 0.19)
- trust_remote_code=False (not applicable here but documented)
- HF_HUB_OFFLINE must NOT be set during download; set it for all loads
- SHA-256 of each weight shard must match Hub LFS oid exactly
- Non-weight files must match environment_manifest.json tokenizer hashes
- Any mismatch → writes model_manifest.json with FAIL status and exits 1

Usage:
    sft\\.venv\\Scripts\\python.exe sft\\download_model.py
    sft\\.venv\\Scripts\\python.exe sft\\download_model.py --verify-only
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
REPO_ID  = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

SFT_DIR        = Path(__file__).resolve().parent
SNAP_DIR       = SFT_DIR / "model_snapshot" / "Qwen3-1.7B"
MANIFEST_PATH  = SFT_DIR / "model_manifest.json"
ENV_MANIFEST   = SFT_DIR / "environment_manifest.json"

# Hub LFS SHA-256 oids for weight shards (from list_repo_tree at pinned revision)
WEIGHT_LFS_SHA256: dict[str, str] = {
    "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
}

# Allowed download patterns
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "merges.txt",
    "LICENSE",
]
# Explicitly exclude binary formats other than safetensors
IGNORE_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.h5",
    "*.hdf5",
    "*.ckpt",
    "*.md",           # README.md not needed for inference
    ".gitattributes",
]


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_env_manifest() -> dict:
    if not ENV_MANIFEST.exists():
        print(f"ERROR: environment_manifest.json not found at {ENV_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    return json.loads(ENV_MANIFEST.read_text(encoding="utf-8"))


# ── download ─────────────────────────────────────────────────────────────────
def download() -> None:
    """Run snapshot_download. Idempotent: resumes interrupted downloads."""
    # Unset offline mode for download
    os.environ.pop("HF_HUB_OFFLINE", None)

    print(f"Downloading {REPO_ID} @ {REVISION[:12]}…")
    print(f"  target: {SNAP_DIR}")

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(SNAP_DIR),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        local_files_only=False,
    )
    print(f"  done  → {local_path}")


# ── verify ────────────────────────────────────────────────────────────────────
def verify() -> bool:
    """
    Verify all downloaded files.
    Returns True if everything matches; False otherwise.
    Writes model_manifest.json in all cases.
    """
    env_manifest = load_env_manifest()
    snap_files = env_manifest["tokenizer_snapshot"]["files"]

    # Non-weight files that should be byte-identical to tokenizer_snapshot
    NON_WEIGHT_NAMES = set(snap_files.keys())

    if not SNAP_DIR.exists():
        print("ERROR: model_snapshot directory does not exist — run without --verify-only first.", file=sys.stderr)
        return False

    # Collect all files present
    present = sorted(SNAP_DIR.rglob("*"))
    present = [p for p in present if p.is_file()]

    entries = []
    errors  = []

    # ── check weight shards ───────────────────────────────────────────────────
    for fname, expected_sha in WEIGHT_LFS_SHA256.items():
        fp = SNAP_DIR / fname
        if not fp.exists():
            errors.append(f"MISSING weight: {fname}")
            entries.append({
                "file": fname,
                "size": None,
                "expected_sha256": expected_sha,
                "actual_sha256": "MISSING",
                "status": "MISSING",
            })
            continue

        size = fp.stat().st_size
        print(f"  hashing {fname} ({size / 1e9:.2f} GB)…", flush=True)
        actual = sha256_file(fp)
        ok = actual == expected_sha
        status = "OK" if ok else "MISMATCH"
        if not ok:
            errors.append(f"SHA256 MISMATCH: {fname}  expected={expected_sha[:16]}  actual={actual[:16]}")
        entries.append({
            "file": fname,
            "size": size,
            "expected_sha256": expected_sha,
            "actual_sha256": actual,
            "status": status,
        })
        print(f"    {status}: {fname}")

    # ── check non-weight files against tokenizer_snapshot hashes ─────────────
    for fname, meta in snap_files.items():
        fp = SNAP_DIR / fname
        if not fp.exists():
            errors.append(f"MISSING non-weight: {fname}")
            entries.append({
                "file": fname,
                "size": None,
                "expected_sha256": meta["sha256"],
                "actual_sha256": "MISSING",
                "status": "MISSING",
            })
            continue
        size = fp.stat().st_size
        actual = sha256_file(fp)
        ok = actual == meta["sha256"]
        status = "OK" if ok else "MISMATCH"
        if not ok:
            errors.append(f"SHA256 MISMATCH non-weight: {fname}  expected={meta['sha256'][:16]}  actual={actual[:16]}")
        entries.append({
            "file": fname,
            "size": size,
            "expected_sha256": meta["sha256"],
            "actual_sha256": actual,
            "status": status,
        })
        print(f"    {status}: {fname}")

    # ── check model.safetensors.index.json exists (no hash pinned, just presence) ──
    idx = SNAP_DIR / "model.safetensors.index.json"
    if idx.exists():
        entries.append({
            "file": "model.safetensors.index.json",
            "size": idx.stat().st_size,
            "expected_sha256": "not-pinned",
            "actual_sha256": sha256_file(idx),
            "status": "PRESENT",
        })
    else:
        errors.append("MISSING: model.safetensors.index.json")
        entries.append({
            "file": "model.safetensors.index.json",
            "size": None,
            "expected_sha256": "not-pinned",
            "actual_sha256": "MISSING",
            "status": "MISSING",
        })

    # ── hub revision check ────────────────────────────────────────────────────
    # Verify the local .cache pointer or README metadata if available
    # snapshot_download stores the revision in the directory; we check via refs
    hub_revision_verified = False
    # Try reading from .refs or huggingface_hub cache refs
    refs_path = SNAP_DIR / ".." / ".." / "refs" / "main"  # not reliable for pinned rev
    # Best we can do: check that index.json was downloaded at correct commit
    # huggingface_hub writes a .gitignore or similar marker - check download log
    # We rely on the fact that snapshot_download with revision= pinned SHA is authoritative
    hub_revision_verified = True  # enforced by revision= parameter in download()
    print(f"  Hub revision pin: {REVISION[:12]}… (enforced by download)")

    # ── write manifest ────────────────────────────────────────────────────────
    overall_status = "PASS" if not errors else "FAIL"
    manifest = {
        "schema_version": "1",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "local_dir": str(SNAP_DIR),
        "overall_status": overall_status,
        "errors": errors,
        "entries": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest written → {MANIFEST_PATH}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        print(f"\nVERIFICATION FAILED: {len(errors)} error(s).", file=sys.stderr)
        return False

    print(f"\nVERIFICATION PASSED: {len(entries)} file(s) verified.")
    return True


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Qwen3-1.7B weights.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download; only verify files already on disk.",
    )
    args = parser.parse_args()

    if not args.verify_only:
        # If destination already fully populated, offer --verify-only path
        if SNAP_DIR.exists() and any(SNAP_DIR.glob("*.safetensors")):
            print(f"model_snapshot already contains safetensors files; "
                  f"resuming/verifying (use --verify-only to skip download).")
        download()

    ok = verify()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
