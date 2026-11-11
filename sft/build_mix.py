"""
sft/build_mix.py — A4: Build tokenized training mix and evaluation sets.

Architecture refs: ARCHITECTURE.md E3, E4, G(P3); IMPLEMENTATION_PLAN Stage 3b.

Usage (from repo root):
    sft/.venv/Scripts/python.exe sft/build_mix.py --help

What it produces (all under sft/work/, git-ignored):
    pilot_mix.jsonl              — E2 pilot mix (4,800 dialogues, 150 steps × 32)
    overfit_set.jsonl            — 16-dialogue overfit set (8 MedDialog + 8 HealthChat)
    val_meddialog_tok.jsonl      — tokenized val set, 200 MedDialog
    val_healthchat_tok.jsonl     — tokenized val set, 200 HealthChat
    final_mix.jsonl              — full final mix (only if --n-dialogues given)
    build_mix_report.json        — realised token shares, truncation/drop counts

Tokenized JSONL format (each line):
    {"dialogue_id": str, "source": str, "input_ids": [int, ...],
     "labels": [int, ...], "n_tokens": int, "truncated": bool}

Mix ratios by tokens (Architecture E4):
    MedDialog   50%
    HealthChat  45%
    Behavior    5%  (each behavior item upsampled × 3)

Seed: 42 (Architecture E4).
max_len: 2,048 (Architecture E3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIX_SEED = 42
MAX_LEN = 2_048
HARD_CAP = 3_072
PILOT_DIALOGUES = 4_800   # 150 steps × 32 effective batch
OVERFIT_MD = 8
OVERFIT_HC = 8
VAL_SUBSET_MD = 200
VAL_SUBSET_HC = 200
# Token-share targets
SHARE_MD = 0.50
SHARE_HC = 0.45
SHARE_BEH = 0.05
BEH_UPSAMPLE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _is_healthchat(source: str) -> bool:
    return "HealthChat" in source


def _is_meddialog(source: str) -> bool:
    return source == "MedDialog"


def _is_behavior(source: str) -> bool:
    return source == "Behavior"


# ---------------------------------------------------------------------------
# Tokenize a JSONL stream and return lists by source
# ---------------------------------------------------------------------------
def tokenize_jsonl(
    tok,
    jsonl_path: Path,
    max_len: int = MAX_LEN,
    hard_cap: int = HARD_CAP,
) -> tuple[list[dict], dict]:
    """
    Stream tokenize_jsonl_path, return:
        items   : list of dicts with keys: dialogue_id, source, input_ids, labels, n_tokens, truncated
        stats   : per-source TokenizationStats as dicts
    """
    from tokenize_mask import iter_tokenized, TokenizationStats

    stats_by_source: dict[str, TokenizationStats] = {}
    items = []

    for item in iter_tokenized(tok, jsonl_path, max_len, hard_cap, stats_by_source):
        items.append({
            "dialogue_id": item.dialogue_id,
            "source": item.source,
            "input_ids": item.input_ids,
            "labels": item.labels,
            "n_tokens": item.n_tokens,
            "truncated": item.truncated,
        })

    stats_dicts = {src: st.to_dict() for src, st in stats_by_source.items()}
    return items, stats_dicts


# ---------------------------------------------------------------------------
# Mix builder
# ---------------------------------------------------------------------------
def build_pilot_mix(
    tok,
    train_jsonl: Path,
    behavior_jsonl: Path | None,
    n_dialogues: int,
    seed: int,
    max_len: int,
    hard_cap: int,
) -> tuple[list[dict], dict]:
    """
    Build a mix of n_dialogues from train + behavior following token-share targets.

    Returns (mix_items, report_dict).
    mix_items: each has dialogue_id, source, input_ids, labels, n_tokens, truncated.
    """
    rng = random.Random(seed)

    # Tokenize train dialogues (stream, don't hold duplicates)
    print(f"  Tokenizing train dialogues from {train_jsonl.name}...")
    all_items, train_stats = tokenize_jsonl(tok, train_jsonl, max_len, hard_cap)

    # Separate by source
    md_items = [x for x in all_items if _is_meddialog(x["source"])]
    hc_items = [x for x in all_items if _is_healthchat(x["source"])]
    beh_items: list[dict] = []

    # Tokenize behavior (if provided)
    if behavior_jsonl and behavior_jsonl.exists():
        print(f"  Tokenizing behavior dialogues from {behavior_jsonl.name}...")
        beh_raw, beh_stats = tokenize_jsonl(tok, behavior_jsonl, max_len, hard_cap)
        beh_items = [x for x in beh_raw if _is_behavior(x["source"])]
        # Upsample behavior × BEH_UPSAMPLE
        beh_items = beh_items * BEH_UPSAMPLE
        train_stats.update(beh_stats)
    else:
        print("  WARNING: behavior_dialogues.jsonl not found — behavior share will be 0")
        beh_items = []

    # Free all_items
    del all_items

    # Shuffle each pool
    rng.shuffle(md_items)
    rng.shuffle(hc_items)
    rng.shuffle(beh_items)

    # Determine how many tokens we expect total and allocate
    # Strategy: iterate rounds, drawing from each pool to maintain share
    # We use a token-budget approach.

    # Estimate total tokens from pool sizes
    md_tokens_total = sum(x["n_tokens"] for x in md_items)
    hc_tokens_total = sum(x["n_tokens"] for x in hc_items)
    beh_tokens_total = sum(x["n_tokens"] for x in beh_items)

    print(f"  Pools: MD={len(md_items)} ({md_tokens_total:,} tok), "
          f"HC={len(hc_items)} ({hc_tokens_total:,} tok), "
          f"BEH={len(beh_items)} ({beh_tokens_total:,} tok)")

    # Build mix up to n_dialogues, maintaining token shares
    mix: list[dict] = []
    md_ptr = hc_ptr = beh_ptr = 0
    md_tok = hc_tok = beh_tok = 0

    for _ in range(n_dialogues):
        total_tok = md_tok + hc_tok + beh_tok
        if total_tok == 0:
            # Start with equal probability weighted by pool presence
            choice = rng.choices(
                ["md", "hc", "beh"],
                weights=[1 if md_items else 0, 1 if hc_items else 0, 1 if beh_items else 0],
            )[0]
        else:
            # Choose source that is most under-represented by token share
            current_md = md_tok / total_tok if total_tok else 0
            current_hc = hc_tok / total_tok if total_tok else 0
            current_beh = beh_tok / total_tok if total_tok else 0

            gaps = [
                ("md",  SHARE_MD  - current_md,  md_ptr < len(md_items)),
                ("hc",  SHARE_HC  - current_hc,  hc_ptr < len(hc_items)),
                ("beh", SHARE_BEH - current_beh, beh_ptr < len(beh_items)),
            ]
            gaps = [(name, gap, avail) for name, gap, avail in gaps if avail]
            if not gaps:
                break  # all pools exhausted
            choice = max(gaps, key=lambda x: x[1])[0]

        if choice == "md":
            if md_ptr >= len(md_items):
                choice = "hc" if hc_ptr < len(hc_items) else "beh"
                if choice == "beh" and beh_ptr >= len(beh_items):
                    break
            if choice == "md":
                item = md_items[md_ptr]
                md_ptr += 1
                md_tok += item["n_tokens"]
                mix.append(item)
                continue

        if choice == "hc":
            if hc_ptr >= len(hc_items):
                choice = "beh" if beh_ptr < len(beh_items) else None
                if choice is None:
                    break
            if choice == "hc":
                item = hc_items[hc_ptr]
                hc_ptr += 1
                hc_tok += item["n_tokens"]
                mix.append(item)
                continue

        if choice == "beh":
            if beh_ptr >= len(beh_items):
                break
            item = beh_items[beh_ptr]
            beh_ptr += 1
            beh_tok += item["n_tokens"]
            mix.append(item)

    # Final shuffle
    rng.shuffle(mix)

    # Count max repeats per dialogue_id (from upsampling)
    repeat_counts: dict[str, int] = defaultdict(int)
    for item in mix:
        repeat_counts[item["dialogue_id"]] += 1
    max_repeat = max(repeat_counts.values(), default=0)

    total_mix_tok = md_tok + hc_tok + beh_tok
    report = {
        "n_dialogues_requested": n_dialogues,
        "n_dialogues_produced": len(mix),
        "total_tokens": total_mix_tok,
        "token_shares": {
            "MedDialog": round(md_tok / max(1, total_mix_tok), 4),
            "HealthChat": round(hc_tok / max(1, total_mix_tok), 4),
            "Behavior": round(beh_tok / max(1, total_mix_tok), 4),
        },
        "max_repeats_per_dialogue": max_repeat,
        "tokenization_stats": train_stats,
    }

    return mix, report


def build_overfit_set(
    tok,
    train_jsonl: Path,
    seed: int,
    max_len: int,
    hard_cap: int,
) -> list[dict]:
    """
    Build the 16-dialogue overfit set: 8 MedDialog + 8 HealthChat.
    Uses the train JSONL; picks by deterministic hash ordering.
    """
    rng = random.Random(seed + 1)

    md_pool: list[dict] = []
    hc_pool: list[dict] = []

    from tokenize_mask import iter_tokenized
    for item in iter_tokenized(tok, train_jsonl, max_len, hard_cap):
        if _is_meddialog(item.source) and len(md_pool) < 500:
            md_pool.append({
                "dialogue_id": item.dialogue_id,
                "source": item.source,
                "input_ids": item.input_ids,
                "labels": item.labels,
                "n_tokens": item.n_tokens,
                "truncated": item.truncated,
            })
        elif _is_healthchat(item.source) and len(hc_pool) < 500:
            hc_pool.append({
                "dialogue_id": item.dialogue_id,
                "source": item.source,
                "input_ids": item.input_ids,
                "labels": item.labels,
                "n_tokens": item.n_tokens,
                "truncated": item.truncated,
            })
        if len(md_pool) >= 500 and len(hc_pool) >= 500:
            break

    # Sort by SHA of dialogue_id for determinism, then pick first 8
    md_pool.sort(key=lambda x: sha256_str(x["dialogue_id"]))
    hc_pool.sort(key=lambda x: sha256_str(x["dialogue_id"]))

    return md_pool[:OVERFIT_MD] + hc_pool[:OVERFIT_HC]


def build_val_subsets(
    tok,
    val_jsonl: Path,
    max_len: int,
    hard_cap: int,
) -> tuple[list[dict], list[dict], dict]:
    """
    Build tokenized val subsets: 200 MedDialog + 200 HealthChat.
    Returns (md_val, hc_val, stats).
    """
    from tokenize_mask import iter_tokenized, TokenizationStats

    md_val: list[dict] = []
    hc_val: list[dict] = []
    stats_by_source: dict[str, TokenizationStats] = {}

    for item in iter_tokenized(tok, val_jsonl, max_len, hard_cap, stats_by_source):
        d = {
            "dialogue_id": item.dialogue_id,
            "source": item.source,
            "input_ids": item.input_ids,
            "labels": item.labels,
            "n_tokens": item.n_tokens,
            "truncated": item.truncated,
        }
        if _is_meddialog(item.source) and len(md_val) < VAL_SUBSET_MD:
            md_val.append(d)
        elif _is_healthchat(item.source) and len(hc_val) < VAL_SUBSET_HC:
            hc_val.append(d)
        if len(md_val) >= VAL_SUBSET_MD and len(hc_val) >= VAL_SUBSET_HC:
            break

    stats_dicts = {src: st.to_dict() for src, st in stats_by_source.items()}
    return md_val, hc_val, stats_dicts


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------
def write_jsonl(items: list[dict], path: Path) -> str:
    """Write tokenized items to JSONL, return SHA-256."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build tokenized training mixes (A4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pkg", default="sft/tokenizer_pkg",
                        help="tokenizer_pkg directory")
    parser.add_argument("--train-jsonl", default="sft/work/dialogues_train.jsonl",
                        help="Train split JSONL")
    parser.add_argument("--val-jsonl", default="sft/work/dialogues_val.jsonl",
                        help="Val split JSONL")
    parser.add_argument("--behavior-jsonl", default="sft/work/behavior_dialogues.jsonl",
                        help="Behavior set JSONL (optional)")
    parser.add_argument("--out-dir", default="sft/work",
                        help="Output directory for tokenized files")
    parser.add_argument("--n-dialogues", type=int, default=None,
                        help="Build final mix with N dialogues (optional; not used for pilots)")
    parser.add_argument("--pilot", action="store_true", default=True,
                        help="Build pilot mix and overfit set (default: True)")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--seed", type=int, default=MIX_SEED)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    pkg_dir = repo_root / args.pkg
    train_jsonl = repo_root / args.train_jsonl
    val_jsonl = repo_root / args.val_jsonl
    behavior_jsonl_path = repo_root / args.behavior_jsonl
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_jsonl.exists():
        print(f"ERROR: train JSONL not found: {train_jsonl}", file=sys.stderr)
        return 1

    # Load tokenizer
    print(f"Loading tokenizer from {pkg_dir}...")
    from tokenize_mask import load_tokenizer
    tok = load_tokenizer(pkg_dir)

    all_reports = {}

    # --- Pilot mix ---
    if args.pilot:
        print(f"\nBuilding pilot mix ({PILOT_DIALOGUES} dialogues)...")
        pilot_items, pilot_report = build_pilot_mix(
            tok, train_jsonl,
            behavior_jsonl_path if behavior_jsonl_path.exists() else None,
            n_dialogues=PILOT_DIALOGUES,
            seed=args.seed,
            max_len=args.max_len,
            hard_cap=HARD_CAP,
        )
        pilot_path = out_dir / "pilot_mix.jsonl"
        pilot_sha = write_jsonl(pilot_items, pilot_path)
        pilot_report["path"] = str(pilot_path)
        pilot_report["sha256"] = pilot_sha
        all_reports["pilot_mix"] = pilot_report
        print(f"  Pilot mix: {len(pilot_items)} dialogues, sha256={pilot_sha[:16]}...")

        # --- Overfit set ---
        print("\nBuilding overfit set (16 dialogues)...")
        overfit_items = build_overfit_set(
            tok, train_jsonl, args.seed, args.max_len, HARD_CAP
        )
        overfit_path = out_dir / "overfit_set.jsonl"
        overfit_sha = write_jsonl(overfit_items, overfit_path)
        all_reports["overfit_set"] = {
            "n": len(overfit_items),
            "path": str(overfit_path),
            "sha256": overfit_sha,
            "md_count": sum(1 for x in overfit_items if _is_meddialog(x["source"])),
            "hc_count": sum(1 for x in overfit_items if _is_healthchat(x["source"])),
        }
        print(f"  Overfit set: {len(overfit_items)} dialogues, sha256={overfit_sha[:16]}...")

    # --- Val subsets ---
    if val_jsonl.exists():
        print("\nBuilding tokenized val subsets...")
        md_val, hc_val, val_stats = build_val_subsets(
            tok, val_jsonl, args.max_len, HARD_CAP
        )
        md_val_path = out_dir / "val_meddialog_tok.jsonl"
        hc_val_path = out_dir / "val_healthchat_tok.jsonl"
        md_sha = write_jsonl(md_val, md_val_path)
        hc_sha = write_jsonl(hc_val, hc_val_path)
        all_reports["val_subsets"] = {
            "meddialog": {"n": len(md_val), "sha256": md_sha},
            "healthchat": {"n": len(hc_val), "sha256": hc_sha},
            "stats": val_stats,
        }
        print(f"  Val MD: {len(md_val)}, HC: {len(hc_val)}")

    # --- Final mix (if --n-dialogues given) ---
    if args.n_dialogues:
        print(f"\nBuilding final mix ({args.n_dialogues} dialogues)...")
        final_items, final_report = build_pilot_mix(
            tok, train_jsonl,
            behavior_jsonl_path if behavior_jsonl_path.exists() else None,
            n_dialogues=args.n_dialogues,
            seed=args.seed,
            max_len=args.max_len,
            hard_cap=HARD_CAP,
        )
        final_path = out_dir / "final_mix.jsonl"
        final_sha = write_jsonl(final_items, final_path)
        final_report["path"] = str(final_path)
        final_report["sha256"] = final_sha
        all_reports["final_mix"] = final_report
        print(f"  Final mix: {len(final_items)} dialogues, sha256={final_sha[:16]}...")

    # Write report
    report_path = out_dir / "build_mix_report.json"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nReport written to {report_path}")
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
