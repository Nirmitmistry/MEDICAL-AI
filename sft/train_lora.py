"""
sft/train_lora.py — A5: LoRA SFT training script (Kaggle GPU only).

Architecture refs: ARCHITECTURE.md E1, E4, E8; IMPLEMENTATION_PLAN Stage 3c/3d.

Usage (from /kaggle/working or repo root):
    python sft/train_lora.py --help

This script:
  - Asserts CUDA is available and limits to exactly 1 GPU (CUDA_VISIBLE_DEVICES=0).
  - Selects bf16 (compute capability >= 8.0) or fp16 (T4/P100).
  - Applies LoRA (r=16, alpha=32, dropout=0.05) to all linear layers
    except lm_head and embedding layers.
  - Trains with HF Trainer, effective batch=32, cosine LR, 3% warmup.
  - Evaluates every 100 steps on two tokenized val sets (MedDialog + HealthChat).
  - Saves every 100 steps; keeps last 3 + best by macro val loss.
  - Supports resume from a read-only attached Kaggle input directory.
  - Writes run_manifest.json.

Torch-free CPU tests: import this module only inside functions;
the top-level code is import-safe (no torch at module level).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (Architecture E4)
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_PATTERN = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# All linear layers EXCEPT lm_head and embed_tokens
LORA_EXCLUDE_NAMES = {"lm_head", "embed_tokens"}

EFFECTIVE_BATCH = 32          # dialogues
CLIP_NORM = 1.0
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LR_SCHEDULE = "cosine"
SEED = 42
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3          # keep last 3 + best
MAX_LEN = 2_048
HARD_CAP = 3_072

MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


# ---------------------------------------------------------------------------
# Helpers (torch-free)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GPU / environment check
# ---------------------------------------------------------------------------
def check_gpu_environment() -> dict:
    """
    Assert exactly 1 CUDA GPU is visible. Return GPU info dict.
    Must be called AFTER setting CUDA_VISIBLE_DEVICES=0.
    """
    import torch  # lazy import
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError(
            "No CUDA GPU found. This script must run on a Kaggle GPU session. "
            "Local CPU training is not supported (Architecture E8)."
        )
    if n != 1:
        raise RuntimeError(
            f"Found {n} CUDA GPUs. CUDA_VISIBLE_DEVICES=0 must be set before "
            "importing torch to prevent DataParallel (Architecture E4)."
        )
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    vram_gb = props.total_memory / (1024 ** 3)
    bf16_ok = cc >= 80  # Ampere and newer
    return {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(vram_gb, 2),
        "bf16_supported": bf16_ok,
        "precision": "bf16" if bf16_ok else "fp16",
    }


# ---------------------------------------------------------------------------
# Data collator (dynamic padding, length-grouped)
# ---------------------------------------------------------------------------
def make_collator(pad_token_id: int):
    """
    Returns a collate_fn that:
      - Pads input_ids to the longest sequence in the batch (right-pad with pad_token_id)
      - Pads labels to the same length with MASKED (-100)
      - Returns attention_mask (1 for real tokens, 0 for pad)
      - Passes num_items_in_batch for token-mean loss (Architecture E4 / T7)
    """
    import torch  # lazy import

    MASKED = -100

    def collate_fn(batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            ids = item["input_ids"]
            lbl = item["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            labels.append(lbl + [MASKED] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "num_items_in_batch": len(batch),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Token-mean loss (honours num_items_in_batch)
# ---------------------------------------------------------------------------
class TokenMeanLossTrainer:
    """
    Mixin to override compute_loss with token-mean loss (Architecture E4).
    Use as: class LoRATrainer(TokenMeanLossTrainer, Trainer): ...
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch  # lazy import
        # Pop num_items_in_batch (not consumed by the model)
        inputs = dict(inputs)
        num_items = inputs.pop("num_items_in_batch", None)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)
        labels = inputs["labels"]  # (B, T)

        # Shift: predict token at position t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        per_token_loss = loss_fct(flat_logits, flat_labels)

        # Token-mean (not sequence-mean): sum over all supervised tokens in the batch
        n_supervised = (flat_labels != -100).sum().float()
        loss = per_token_loss.sum() / n_supervised.clamp(min=1.0)

        if return_outputs:
            return loss, outputs
        return loss


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class TokenizedDataset:
    """Simple map-style dataset wrapping a list of tokenized items."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "n_tokens": item["n_tokens"],
        }


# ---------------------------------------------------------------------------
# Length-grouped sampler (Architecture E4)
# ---------------------------------------------------------------------------
def make_length_grouped_sampler(dataset: TokenizedDataset, batch_size: int, seed: int = SEED):
    """
    Sort by length descending, then shuffle within buckets of batch_size.
    Returns a list of indices.
    """
    import random
    rng = random.Random(seed)
    indices = sorted(range(len(dataset)), key=lambda i: dataset[i]["n_tokens"], reverse=True)
    # Shuffle within buckets
    for start in range(0, len(indices), batch_size):
        bucket = indices[start: start + batch_size]
        rng.shuffle(bucket)
        indices[start: start + batch_size] = bucket
    return indices


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    model_dir: Path,
    tokenizer_pkg_dir: Path,
    train_jsonl: Path,
    val_md_jsonl: Path,
    val_hc_jsonl: Path,
    out_dir: Path,
    resume_dir: Optional[Path],
    learning_rate: float,
    micro_batch: int,
    n_steps: int,
    seed: int = SEED,
    precision: Optional[str] = None,
) -> dict:
    """
    Full training run. Returns run_manifest dict.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        set_seed,
    )
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415

    set_seed(seed)

    # --- GPU check ---
    gpu_info = check_gpu_environment()
    if precision is None:
        precision = gpu_info["precision"]
    print(f"GPU: {gpu_info['device_name']} | VRAM: {gpu_info['vram_gb']:.1f} GB | "
          f"precision: {precision}")

    # --- Load tokenizer ---
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(str(tokenizer_pkg_dir), local_files_only=True)

    # --- Load model (base weights in bf16/fp16) ---
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    print(f"Loading base model from {model_dir} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map={"": 0},          # single GPU
        attn_implementation="sdpa",  # PyTorch SDPA (Architecture E4)
        local_files_only=True,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    # --- LoRA config (Architecture E4) ---
    # Find all linear layer names except lm_head and embed_tokens
    target_modules = [
        name.split(".")[-1]
        for name, module in model.named_modules()
        if (
            isinstance(module, torch.nn.Linear)
            and not any(ex in name for ex in LORA_EXCLUDE_NAMES)
        )
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    target_modules_dedup = []
    for m in target_modules:
        if m not in seen:
            seen.add(m)
            target_modules_dedup.append(m)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules_dedup,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # LoRA weights stay in fp32 (Architecture E4)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Load tokenized data ---
    def load_tok_jsonl(path: Path) -> list[dict]:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    print("Loading tokenized datasets...")
    train_items = load_tok_jsonl(train_jsonl)
    val_md_items = load_tok_jsonl(val_md_jsonl)
    val_hc_items = load_tok_jsonl(val_hc_jsonl)

    train_dataset = TokenizedDataset(train_items)
    val_md_dataset = TokenizedDataset(val_md_items)
    val_hc_dataset = TokenizedDataset(val_hc_items)

    # Gradient accumulation = effective_batch / micro_batch
    grad_accum = max(1, EFFECTIVE_BATCH // micro_batch)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        max_steps=n_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type=LR_SCHEDULE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=CLIP_NORM,
        # Precision
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        # Eval / save
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT + 1,  # +1 because best is also saved
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # will be overridden by custom eval
        # Logging
        logging_steps=10,
        logging_dir=str(out_dir / "logs"),
        # Reproducibility
        seed=seed,
        data_seed=seed,
        # Offline
        hub_model_id=None,
        push_to_hub=False,
        # Save full state for resumability (Architecture E4)
        save_safetensors=True,
        # Length-grouped is handled manually via custom sampler
        group_by_length=False,
    )

    # --- Collator ---
    collator = make_collator(tok.pad_token_id)

    # --- Custom Trainer with token-mean loss ---
    class LoRATrainer(TokenMeanLossTrainer, Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            # Override to evaluate on both val sets and return macro loss
            md_metrics = super().evaluate(
                eval_dataset=val_md_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix="eval_md",
            )
            hc_metrics = super().evaluate(
                eval_dataset=val_hc_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix="eval_hc",
            )
            md_loss = md_metrics.get("eval_md_loss", float("inf"))
            hc_loss = hc_metrics.get("eval_hc_loss", float("inf"))
            macro = (md_loss + hc_loss) / 2.0
            combined = {**md_metrics, **hc_metrics, "eval_loss": macro, "eval_macro": macro}
            return combined

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_md_dataset,  # required by Trainer; overridden in evaluate()
        data_collator=collator,
    )

    # --- Resume ---
    if resume_dir and resume_dir.exists():
        # Copy checkpoint to working dir (read-only input → working copy)
        import shutil
        resume_work = out_dir / "resume_checkpoint"
        if not resume_work.exists():
            print(f"Copying resume checkpoint from {resume_dir} → {resume_work}...")
            shutil.copytree(str(resume_dir), str(resume_work))
        resume_from = str(resume_work)
        print(f"Resuming from: {resume_from}")
    else:
        resume_from = None

    # --- Train ---
    print(f"Starting training: lr={learning_rate}, micro_batch={micro_batch}, "
          f"grad_accum={grad_accum}, max_steps={n_steps}")
    trainer.train(resume_from_checkpoint=resume_from)

    # --- Build run_manifest ---
    pkg_manifest_path = tokenizer_pkg_dir / "tokenizer_pkg_manifest.json"
    pkg_manifest = {}
    if pkg_manifest_path.exists():
        with open(pkg_manifest_path, encoding="utf-8") as f:
            pkg_manifest = json.load(f)

    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    pkg_versions = {}
    for pkg_name in ["torch", "transformers", "peft", "accelerate"]:
        try:
            pkg_versions[pkg_name] = _pkg_version(pkg_name)
        except (PackageNotFoundError, Exception):
            pkg_versions[pkg_name] = "unknown"

    run_manifest = {
        "schema": "run_manifest_v1",
        "seed": seed,
        "model_revision": MODEL_REVISION,
        "precision": precision,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": target_modules_dedup,
        },
        "training": {
            "learning_rate": learning_rate,
            "micro_batch": micro_batch,
            "gradient_accumulation": grad_accum,
            "effective_batch": EFFECTIVE_BATCH,
            "max_steps": n_steps,
            "max_len": MAX_LEN,
        },
        "data": {
            "train_jsonl": str(train_jsonl),
            "train_sha256": sha256_file(train_jsonl),
            "val_md_jsonl": str(val_md_jsonl),
            "val_md_sha256": sha256_file(val_md_jsonl),
            "val_hc_jsonl": str(val_hc_jsonl),
            "val_hc_sha256": sha256_file(val_hc_jsonl),
        },
        "tokenizer_pkg": {
            "template_sha256": pkg_manifest.get("template_sha256", ""),
            "system_prompt_sha256": pkg_manifest.get("system_prompt_sha256", ""),
        },
        "gpu": gpu_info,
        "package_versions": pkg_versions,
        "trainable_params": trainable,
        "total_params": total,
    }

    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"run_manifest.json written to {manifest_path}")

    return run_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    # Set CUDA_VISIBLE_DEVICES BEFORE any torch import (Architecture E4)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    parser = argparse.ArgumentParser(
        description="LoRA SFT training script (Kaggle GPU only). "
                    "Architecture: ARCHITECTURE.md E1, E4, E8.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", default="/kaggle/input/qwen3-1-7b/model_snapshot",
                        help="Local path to Qwen3-1.7B model weights")
    parser.add_argument("--pkg", default="sft/tokenizer_pkg",
                        help="tokenizer_pkg directory")
    parser.add_argument("--train-tok", required=True,
                        help="Tokenized train JSONL (pilot_mix or final_mix)")
    parser.add_argument("--val-md-tok", required=True,
                        help="Tokenized val MedDialog JSONL")
    parser.add_argument("--val-hc-tok", required=True,
                        help="Tokenized val HealthChat JSONL")
    parser.add_argument("--out", default="/kaggle/working/lora_run",
                        help="Output directory for checkpoints and manifest")
    parser.add_argument("--resume", default=None,
                        help="Path to previous checkpoint directory (read-only Kaggle input)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4; E2 pilots: 5e-5, 1e-4, 2e-4)")
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="Per-device micro-batch size (from E0; default: 2)")
    parser.add_argument("--n-steps", type=int, default=150,
                        help="Max training steps (pilot: 150; final: from budget)")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default=None,
                        help="Override precision (default: auto from compute capability)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()

    try:
        result = train(
            model_dir=Path(args.model_dir),
            tokenizer_pkg_dir=repo_root / args.pkg,
            train_jsonl=Path(args.train_tok),
            val_md_jsonl=Path(args.val_md_tok),
            val_hc_jsonl=Path(args.val_hc_tok),
            out_dir=Path(args.out),
            resume_dir=Path(args.resume) if args.resume else None,
            learning_rate=args.lr,
            micro_batch=args.micro_batch,
            n_steps=args.n_steps,
            seed=args.seed,
            precision=args.precision,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nTraining complete. Run manifest: {result.get('schema')}")
    return 0


if __name__ == "__main__":
    # Set CUDA_VISIBLE_DEVICES before any torch import
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.exit(main())
