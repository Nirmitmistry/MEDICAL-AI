"""
sft/build_tokenizer_pkg.py — A1: Build sft/tokenizer_pkg/ from tokenizer_snapshot/
                              with our custom ChatML template (Architecture D4).

Usage (from repo root):
    sft/.venv/Scripts/python.exe sft/build_tokenizer_pkg.py [--out sft/tokenizer_pkg]

Exit codes: 0 success, 1 failure.

What it does:
  1. Copies every file from tokenizer_snapshot/ to tokenizer_pkg/.
  2. Overwrites tokenizer_config.json with our custom chat_template (no <think>,
     generation prompt "<|im_start|>assistant\\n", default system prompt injected
     when messages[0] is not system).
  3. Overwrites generation_config.json with pinned eos=[151645,151643], pad=151643.
  4. Writes tokenizer_pkg_manifest.json with SHA-256 of every file and the
     template SHA-256 separately.

The custom template is architecture D4 exactly:
    <|im_start|>system\\n{SYSTEM_PROMPT_V1}<|im_end|>\\n
    <|im_start|>user\\n{content}<|im_end|>\\n
    <|im_start|>assistant\\n{content}<|im_end|>\\n
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Our custom Jinja2 chat template (Architecture D4)
# Rules:
#  - No <think> blocks anywhere
#  - If messages[0].role != "system", inject the default SYSTEM_PROMPT_V1
#  - generation prompt: "<|im_start|>assistant\n"
#  - EOS is <|im_end|>; pad is <|endoftext|> (different from EOS)
# ---------------------------------------------------------------------------
CUSTOM_CHAT_TEMPLATE = (
    "{%- set ns = namespace(has_system=false) -%}"
    "{%- for message in messages -%}"
    "  {%- if message.role == 'system' -%}"
    "    {%- set ns.has_system = true -%}"
    "  {%- endif -%}"
    "{%- endfor -%}"
    "{%- if not ns.has_system -%}"
    "{{- '<|im_start|>system\\n"
    "You are a general health-information assistant. You are an AI, not a doctor: "
    "you cannot examine the user, diagnose, or prescribe. Use everything the user "
    "has told you in this conversation. If key details are missing, ask one or two "
    "focused questions. State uncertainty plainly. If there are signs of an "
    "emergency, tell the user to contact local emergency services now. Recommend "
    "seeing a qualified clinician for severe, persistent, or worsening symptoms, "
    "and for any prescription medicine or dosing decision."
    "<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "  {%- if message.role == 'system' -%}"
    "    {{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' -}}"
    "  {%- elif message.role == 'user' -%}"
    "    {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' -}}"
    "  {%- elif message.role == 'assistant' -%}"
    "    {{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' -}}"
    "  {%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)

# Default system prompt text (must match sft/prompts/system_prompt_v1.txt exactly)
DEFAULT_SYSTEM_PROMPT = (
    "You are a general health-information assistant. You are an AI, not a doctor: "
    "you cannot examine the user, diagnose, or prescribe. Use everything the user "
    "has told you in this conversation. If key details are missing, ask one or two "
    "focused questions. State uncertainty plainly. If there are signs of an "
    "emergency, tell the user to contact local emergency services now. Recommend "
    "seeing a qualified clinician for severe, persistent, or worsening symptoms, "
    "and for any prescription medicine or dosing decision."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tokenizer_pkg(
    snapshot_dir: Path,
    out_dir: Path,
    system_prompt_path: Path | None = None,
) -> dict:
    """
    Copy snapshot → out_dir, patch template + generation_config.
    Returns the manifest dict (entries sorted by path).
    """
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify system prompt matches the embedded text
    if system_prompt_path and system_prompt_path.exists():
        sp_text = system_prompt_path.read_text(encoding="utf-8").strip()
        if sp_text != DEFAULT_SYSTEM_PROMPT:
            raise ValueError(
                f"system_prompt_v1.txt does not match embedded DEFAULT_SYSTEM_PROMPT.\n"
                f"  File:     {sp_text!r}\n"
                f"  Embedded: {DEFAULT_SYSTEM_PROMPT!r}"
            )

    # Copy all files from snapshot (overwrite existing)
    skip_names = {".cache"}  # skip cache subdirectory
    for item in snapshot_dir.iterdir():
        if item.name in skip_names or item.is_dir():
            continue
        dst = out_dir / item.name
        shutil.copy2(str(item), str(dst))

    # --- Patch tokenizer_config.json ---
    tc_path = out_dir / "tokenizer_config.json"
    with open(tc_path, "r", encoding="utf-8") as f:
        tc = json.load(f)

    tc["chat_template"] = CUSTOM_CHAT_TEMPLATE
    # Ensure pad_token is <|endoftext|> (151643) != <|im_end|> (151645)
    tc["pad_token"] = "<|endoftext|>"
    # Ensure eos_token is <|im_end|>
    tc["eos_token"] = "<|im_end|>"

    with open(tc_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # --- Patch generation_config.json ---
    gc_path = out_dir / "generation_config.json"
    with open(gc_path, "r", encoding="utf-8") as f:
        gc = json.load(f)

    # Architecture D4: eos=[<|im_end|>=151645, <|endoftext|>=151643]; pad=151643 != 151645
    gc["eos_token_id"] = [151645, 151643]
    gc["pad_token_id"] = 151643
    # Remove decode parameters that belong to inference config, not generation_config
    gc.pop("temperature", None)
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    gc.pop("do_sample", None)

    with open(gc_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(gc, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # --- Build manifest ---
    template_bytes = CUSTOM_CHAT_TEMPLATE.encode("utf-8")
    template_sha = sha256_bytes(template_bytes)
    system_prompt_sha = sha256_bytes(DEFAULT_SYSTEM_PROMPT.encode("utf-8"))

    entries = []
    for fp in sorted(out_dir.iterdir()):
        if fp.is_dir():
            continue
        entries.append({
            "path": fp.name,
            "sha256": sha256_file(fp),
            "size": fp.stat().st_size,
        })

    manifest = {
        "schema": "tokenizer_pkg_manifest_v1",
        "snapshot_dir": str(snapshot_dir),
        "out_dir": str(out_dir),
        "template_sha256": template_sha,
        "system_prompt_sha256": system_prompt_sha,
        "files": entries,
    }

    manifest_path = out_dir / "tokenizer_pkg_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build sft/tokenizer_pkg/ from tokenizer_snapshot/ with custom ChatML template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--snapshot",
        default="sft/tokenizer_snapshot",
        help="Path to tokenizer_snapshot/ (default: sft/tokenizer_snapshot)",
    )
    parser.add_argument(
        "--out",
        default="sft/tokenizer_pkg",
        help="Output directory for tokenizer_pkg/ (default: sft/tokenizer_pkg)",
    )
    parser.add_argument(
        "--system-prompt",
        default="sft/prompts/system_prompt_v1.txt",
        help="Path to system_prompt_v1.txt for hash verification",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    snapshot_dir = repo_root / args.snapshot
    out_dir = repo_root / args.out
    sp_path = repo_root / args.system_prompt

    try:
        manifest = build_tokenizer_pkg(snapshot_dir, out_dir, sp_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"tokenizer_pkg built at: {out_dir}")
    print(f"Template SHA-256:       {manifest['template_sha256']}")
    print(f"System prompt SHA-256:  {manifest['system_prompt_sha256']}")
    print(f"Files:")
    for entry in manifest["files"]:
        print(f"  {entry['path']:40s}  {entry['sha256'][:16]}...  {entry['size']:,} B")
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
