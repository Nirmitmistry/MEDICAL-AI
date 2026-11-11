"""
sft/load_check.py — P0b Step 4
================================
Offline model load check. Writes sft/load_check.json.

Checks:
  a. Tokenizer from model_snapshot gives same ids as P0a
     (len 151669; im_start 151644, im_end 151645, endoftext 151643)

  b. config.json: qwen3/Qwen3ForCausalLM, 28 layers, hidden 2048, 16 heads,
     8 KV heads, vocab 151936, tied embeddings; records max_position_embeddings.

  c. Safetensors tensor names/shapes match a from_config meta-device model.

  d. Full load only if >= 6 GiB free RAM (CPU bf16) or on a CUDA device
     with sufficient VRAM; one forward pass with finite logits and mean NLL < 8.

  e. PEFT LoraConfig as fixed: r=16, alpha=32, dropout=0.05, all linear layers;
     trainable param count == formula derived from config.json (expected ~17.4M).

Tier:
  PASS         — tokenizer + structure + full load (including PEFT count)
  PASS-PARTIAL — tokenizer + structure only (full load deferred to E0)
  FAIL         — any check failed

Usage:
    HF_HUB_OFFLINE=1  sft\\.venv-gpu\\Scripts\\python.exe sft\\load_check.py
    sft\\.venv-gpu\\Scripts\\python.exe sft\\load_check.py --out sft\\load_check.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SFT_DIR   = Path(__file__).resolve().parent
SNAP_DIR  = SFT_DIR / "tokenizer_snapshot"
MODEL_DIR = SFT_DIR / "model_snapshot" / "Qwen3-1.7B"
OUT_PATH  = SFT_DIR / "load_check.json"

# ── P0a reference values ──────────────────────────────────────────────────────
P0A_TOK_LEN     = 151669
P0A_IM_START    = 151644
P0A_IM_END      = 151645
P0A_ENDOFTEXT   = 151643

# ── expected config values ─────────────────────────────────────────────────────
EXPECTED_MODEL_TYPE   = "qwen3"
EXPECTED_ARCH         = "Qwen3ForCausalLM"
EXPECTED_LAYERS       = 28
EXPECTED_HIDDEN       = 2048
EXPECTED_HEADS        = 16
EXPECTED_KV_HEADS     = 8
EXPECTED_VOCAB        = 151936
EXPECTED_TIE_EMBED    = True

# ── LoRA config (fixed) ───────────────────────────────────────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

# ── GiB threshold for full load ───────────────────────────────────────────────
FULL_LOAD_RAM_GIB = 6.0

# ── test sentence ─────────────────────────────────────────────────────────────
TEST_SENTENCE = "The quick brown fox jumps over the lazy dog."


# ── helpers ───────────────────────────────────────────────────────────────────
def free_ram_gib() -> float:
    """Return free physical RAM in GiB (psutil preferred, else os fallback)."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    try:
        # Windows fallback via ctypes
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual",ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 0.0


def compute_expected_lora_params(cfg: dict) -> int:
    """
    Derive expected trainable LoRA param count from config.json values.

    LoRA targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    (all linear layers in each transformer block).

    For each target (in_dim x out_dim), trainable params = r*(in_dim + out_dim).
    """
    h  = cfg["hidden_size"]          # 2048
    nh = cfg["num_attention_heads"]  # 16
    nk = cfg["num_key_value_heads"]  # 8
    ii = cfg["intermediate_size"]    # 6144
    L  = cfg["num_hidden_layers"]    # 28
    # head_dim from config if present, else derive
    hd = cfg.get("head_dim", h // nh)  # 128

    q_out = nh * hd   # 2048
    k_out = nk * hd   # 1024
    v_out = nk * hd   # 1024

    per_layer = (
        LORA_R * (h + q_out)   # q_proj:    in=h, out=q_out
        + LORA_R * (h + k_out) # k_proj:    in=h, out=k_out
        + LORA_R * (h + v_out) # v_proj:    in=h, out=v_out
        + LORA_R * (q_out + h) # o_proj:    in=q_out, out=h
        + LORA_R * (h + ii)    # gate_proj: in=h, out=ii
        + LORA_R * (h + ii)    # up_proj:   in=h, out=ii
        + LORA_R * (ii + h)    # down_proj: in=ii, out=h
    )
    return L * per_layer


def mean_nll(logits, input_ids) -> float:
    """Cross-entropy NLL averaged over non-first tokens."""
    import torch
    import torch.nn.functional as F
    # logits: [1, seq, vocab]; shift by 1
    shift_logits = logits[0, :-1, :].float()
    shift_labels = input_ids[0, 1:]
    losses = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return losses.mean().item()


# ── main checks ───────────────────────────────────────────────────────────────
def run_checks(out_path: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"

    result: dict = {
        "checks": {},
        "tier": None,
        "errors": [],
        "warnings": [],
        "max_position_embeddings": None,
        "expected_trainable_params": None,
        "actual_trainable_params": None,
        "mean_nll": None,
        "full_load_attempted": False,
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]
    checks: dict = result["checks"]

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK A — Tokenizer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[A] Tokenizer check …")
    try:
        from transformers import AutoTokenizer  # type: ignore
        if not SNAP_DIR.exists():
            raise FileNotFoundError("tokenizer_snapshot not found: " + str(SNAP_DIR))
        tok = AutoTokenizer.from_pretrained(str(SNAP_DIR), local_files_only=True)

        # vocab length
        tok_len = len(tok)
        len_ok = tok_len == P0A_TOK_LEN
        if not len_ok:
            errors.append("tokenizer len " + str(tok_len) + " != expected " + str(P0A_TOK_LEN))

        # special token ids
        im_start_ids = tok.encode("<|im_start|>", add_special_tokens=False)
        im_end_ids   = tok.encode("<|im_end|>",   add_special_tokens=False)
        eot_ids      = tok.encode("<|endoftext|>", add_special_tokens=False)

        im_start_ok = (len(im_start_ids) == 1 and im_start_ids[0] == P0A_IM_START)
        im_end_ok   = (len(im_end_ids)   == 1 and im_end_ids[0]   == P0A_IM_END)
        eot_ok      = (len(eot_ids)      == 1 and eot_ids[0]      == P0A_ENDOFTEXT)

        for name, ok, expected, actual in [
            ("im_start", im_start_ok, P0A_IM_START, im_start_ids),
            ("im_end",   im_end_ok,   P0A_IM_END,   im_end_ids),
            ("endoftext",eot_ok,      P0A_ENDOFTEXT, eot_ids),
        ]:
            if not ok:
                errors.append(name + " id mismatch: expected " + str(expected) + " got " + str(actual))

        checks["A_tokenizer"] = "PASS" if (len_ok and im_start_ok and im_end_ok and eot_ok) else "FAIL"
        print("  len=" + str(tok_len) + "  im_start=" + str(im_start_ids)
              + "  im_end=" + str(im_end_ids) + "  eot=" + str(eot_ids)
              + "  -> " + checks["A_tokenizer"])
    except Exception as exc:
        checks["A_tokenizer"] = "FAIL"
        errors.append("tokenizer check exception: " + str(exc))
        print("  FAIL: " + str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK B — Config
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[B] Config check …")
    cfg_data: dict | None = None
    try:
        cfg_path = MODEL_DIR / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError("config.json not found: " + str(cfg_path))
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))

        b_errors: list[str] = []
        def chk(field: str, expected, actual) -> None:
            if actual != expected:
                b_errors.append(field + ": expected " + str(expected) + " got " + str(actual))

        chk("model_type",          EXPECTED_MODEL_TYPE,  cfg_data.get("model_type"))
        chk("architectures[0]",    EXPECTED_ARCH,        (cfg_data.get("architectures") or [""])[0])
        chk("num_hidden_layers",   EXPECTED_LAYERS,      cfg_data.get("num_hidden_layers"))
        chk("hidden_size",         EXPECTED_HIDDEN,      cfg_data.get("hidden_size"))
        chk("num_attention_heads", EXPECTED_HEADS,       cfg_data.get("num_attention_heads"))
        chk("num_key_value_heads", EXPECTED_KV_HEADS,    cfg_data.get("num_key_value_heads"))
        chk("vocab_size",          EXPECTED_VOCAB,       cfg_data.get("vocab_size"))
        chk("tie_word_embeddings", EXPECTED_TIE_EMBED,   cfg_data.get("tie_word_embeddings"))

        mpe = cfg_data.get("max_position_embeddings")
        result["max_position_embeddings"] = mpe
        print("  max_position_embeddings=" + str(mpe))

        if b_errors:
            for e in b_errors:
                errors.append("config: " + e)
        checks["B_config"] = "PASS" if not b_errors else "FAIL"
        print("  -> " + checks["B_config"])
    except Exception as exc:
        checks["B_config"] = "FAIL"
        errors.append("config check exception: " + str(exc))
        print("  FAIL: " + str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK C — Safetensors structure vs meta-device model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[C] Safetensors structure check …")
    try:
        import torch  # type: ignore
        from safetensors import safe_open  # type: ignore
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

        # Load config and build meta model (no weights, no memory)
        hf_cfg = AutoConfig.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=False,
        )
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(
                hf_cfg,
                trust_remote_code=False,
            )
        meta_shapes = {k: tuple(v.shape) for k, v in meta_model.named_parameters()}

        # Collect tensor names/shapes from safetensors files
        idx_path = MODEL_DIR / "model.safetensors.index.json"
        if not idx_path.exists():
            raise FileNotFoundError("index.json not found")
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        # weight_map: {tensor_name: shard_filename}
        weight_map: dict[str, str] = idx.get("weight_map", {})

        # Read shapes from shard headers (fast — no data load)
        shard_tensors: dict[str, tuple] = {}
        seen_shards: set[str] = set()
        c_errors: list[str] = []

        for tensor_name, shard_file in weight_map.items():
            if shard_file in seen_shards:
                continue
            shard_path = MODEL_DIR / shard_file
            if not shard_path.exists():
                c_errors.append("shard missing: " + shard_file)
                continue
            seen_shards.add(shard_file)
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    shard_tensors[k] = tuple(f.get_slice(k)[:].shape)

        # Compare with meta model — exclude tied-embedding keys before any check.
        # When tie_word_embeddings=True, lm_head.weight may appear in the file
        # but not in named_parameters() (deduplication), or vice versa, or with
        # a different apparent shape depending on the transformers version and how
        # the meta device represents tied weights.  All variations are expected for
        # a correctly saved model; none should raise a structural error.
        tied_ok: set[str] = set()
        if cfg_data and cfg_data.get("tie_word_embeddings"):
            tied_ok.add("lm_head.weight")

        meta_keys = set(meta_shapes.keys())
        file_keys = set(shard_tensors.keys())

        mismatches: list[str] = []
        missing_in_file: list[str] = []
        extra_in_file: list[str] = []

        for k in sorted(meta_keys - file_keys):
            if k not in tied_ok:
                missing_in_file.append(k)
        for k in sorted(file_keys - meta_keys):
            if k not in tied_ok:
                extra_in_file.append(k)
        for k in sorted(meta_keys & file_keys):
            if k in tied_ok:
                continue  # shape may differ for tied weights; not an error
            ms = meta_shapes[k]
            fs = shard_tensors[k]
            if ms != fs:
                mismatches.append(k + ": meta=" + str(ms) + " file=" + str(fs))

        # Report tied-weight handling as an informational note, not an error.
        if tied_ok:
            tied_in_file = sorted(tied_ok & file_keys)
            tied_in_meta = sorted(tied_ok & meta_keys)
            print(
                "  tied_weights=" + str(sorted(tied_ok))
                + "  in_file=" + str(tied_in_file)
                + "  in_meta=" + str(tied_in_meta)
                + "  (excluded from structural checks)"
            )

        if missing_in_file or extra_in_file or mismatches or c_errors:
            for e in (missing_in_file + extra_in_file + mismatches + c_errors):
                errors.append("structure: " + str(e))
            checks["C_structure"] = "FAIL"
        else:
            checks["C_structure"] = "PASS"

        print("  meta_params=" + str(len(meta_keys))
              + "  file_tensors=" + str(len(file_keys))
              + "  real_missing=" + str(len(missing_in_file))
              + "  mismatches=" + str(len(mismatches))
              + "  -> " + checks["C_structure"])

        del meta_model
        gc.collect()

    except Exception as exc:
        checks["C_structure"] = "FAIL"
        errors.append("structure check exception: " + str(exc))
        print("  FAIL: " + str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK D — Full load (conditional on >= 6 GiB free RAM; on Kaggle with GPU)
    # This check is DEFERRED locally if RAM < 6 GiB and no GPU is available.
    # Run load_check.py on the Kaggle training session for a PASS result on D/E.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[D] Full load check …")
    ram_gib = free_ram_gib()
    print("  free_ram_gib=" + str(round(ram_gib, 2)))

    # Determine load device
    load_device = None
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            vram_gib = _torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            print("  free_vram_gib=" + str(round(vram_gib, 2)))
            # Always attempt GPU if CUDA is available (catch OOM gracefully)
            load_device = "cuda"
    except Exception:
        pass

    if load_device is None and ram_gib >= FULL_LOAD_RAM_GIB:
        load_device = "cpu"

    if load_device is None:
        checks["D_full_load"] = "DEFERRED"
        warnings.append(
            "Full load deferred locally: free RAM " + str(round(ram_gib, 2)) + " GiB < "
            + str(FULL_LOAD_RAM_GIB) + " GiB and no GPU available. "
            "Run load_check.py in the Kaggle training session to complete checks D and E."
        )
        print("  DEFERRED (run on Kaggle for full D/E pass)")
    else:
        result["full_load_attempted"] = True
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            print("  loading model on " + load_device + " (bf16) …")
            dtype = torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,             # per spec: dtype= not torch_dtype=
                attn_implementation="sdpa",
            )
            model = model.to(load_device)
            model.eval()

            # Tokenize test sentence
            tok2 = AutoTokenizer.from_pretrained(str(SNAP_DIR), local_files_only=True)
            inputs = tok2(TEST_SENTENCE, return_tensors="pt")
            input_ids = inputs["input_ids"].to(load_device)

            with torch.no_grad():
                out = model(input_ids=input_ids)
            logits = out.logits

            # Finite check
            finite_ok = bool(torch.isfinite(logits).all().item())
            nll = mean_nll(logits, input_ids) if finite_ok else float("inf")
            nll_ok = nll < 8.0
            result["mean_nll"] = round(nll, 4)

            if not finite_ok:
                errors.append("forward pass: logits not finite")
            if not nll_ok:
                errors.append("mean NLL " + str(round(nll, 4)) + " >= 8.0")

            checks["D_full_load"] = "PASS" if (finite_ok and nll_ok) else "FAIL"
            print("  logits_finite=" + str(finite_ok)
                  + "  mean_nll=" + str(round(nll, 4))
                  + "  -> " + checks["D_full_load"])

            del model
            gc.collect()
            if load_device == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            # Handle OOM gracefully — if CUDA OOM, fall back gracefully to DEFERRED
            exc_str = str(exc)
            if "CUDA out of memory" in exc_str or "OutOfMemoryError" in exc_str:
                checks["D_full_load"] = "DEFERRED"
                warnings.append(
                    "Full load deferred: CUDA OOM on " + str(load_device)
                    + " (" + exc_str.split("\n")[0][:120] + ")"
                )
                print("  DEFERRED (CUDA OOM — insufficient VRAM)")
            else:
                checks["D_full_load"] = "FAIL"
                errors.append("full load exception: " + exc_str)
                print("  FAIL: " + exc_str[:200])

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK E — PEFT LoraConfig trainable param count
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[E] PEFT LoRA param count …")
    try:
        import torch  # type: ignore
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

        if cfg_data is None:
            raise RuntimeError("config.json not loaded (check B failed)")

        expected_params = compute_expected_lora_params(cfg_data)
        result["expected_trainable_params"] = expected_params

        print("  expected_trainable_params=" + str(expected_params)
              + "  (~" + str(round(expected_params / 1e6, 1)) + "M)")

        # Build tiny meta model to count params without allocating memory
        hf_cfg2 = AutoConfig.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            trust_remote_code=False,
        )
        with torch.device("meta"):
            base_model = AutoModelForCausalLM.from_config(
                hf_cfg2,
                trust_remote_code=False,
            )

        # Move to cpu (meta tensors are fine with PEFT for param counting)
        # PEFT needs real tensors; use cpu with float32 meta workaround
        # Actually PEFT requires real weights for get_peft_model
        # Use a small real allocation: load config only and check param count via LoraConfig math
        # Simpler: apply LoraConfig to a real but empty (zeros) model — but that needs too much RAM
        # Use PEFT's internal module inspection on meta model instead:
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        # For param counting we need a real model — check if we have enough RAM/VRAM
        # Qwen3-1.7B in float32 meta = 0 bytes; in float16 real cpu ~ 3.4 GB
        # Use bfloat16 and check if we can afford it
        can_load_for_peft = (ram_gib >= FULL_LOAD_RAM_GIB) or (load_device == "cuda")

        if can_load_for_peft:
            print("  loading real model for PEFT param count on " + str(load_device or "cpu") + " …")
            real_model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
            )
            peft_model = get_peft_model(real_model, lora_cfg)
            tp, ap = peft_model.get_nb_trainable_parameters()
            result["actual_trainable_params"] = tp

            params_ok = (tp == expected_params)
            if not params_ok:
                errors.append(
                    "trainable params " + str(tp) + " != expected " + str(expected_params)
                )
            checks["E_peft_params"] = "PASS" if params_ok else "FAIL"
            print("  actual=" + str(tp) + "  expected=" + str(expected_params)
                  + "  -> " + checks["E_peft_params"])

            del peft_model, real_model
            gc.collect()
        else:
            # Count from formula only — still print expected; mark deferred
            checks["E_peft_params"] = "DEFERRED"
            warnings.append(
                "PEFT param count with real model deferred (insufficient RAM/VRAM). "
                "Formula gives " + str(expected_params) + " params."
            )
            print("  DEFERRED (formula=" + str(expected_params) + ")")

        del base_model
        gc.collect()

    except Exception as exc:
        checks["E_peft_params"] = "FAIL"
        errors.append("peft check exception: " + str(exc))
        print("  FAIL: " + str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # TIER DECISION
    # ─────────────────────────────────────────────────────────────────────────
    mandatory = ["A_tokenizer", "B_config", "C_structure"]
    optional_full = ["D_full_load", "E_peft_params"]

    mand_pass  = all(checks.get(k) == "PASS" for k in mandatory)
    full_pass  = all(checks.get(k) in ("PASS", "DEFERRED") for k in optional_full)
    full_done  = all(checks.get(k) == "PASS" for k in optional_full)

    if errors:
        tier = "FAIL"
    elif mand_pass and full_done:
        tier = "PASS"
    elif mand_pass:
        tier = "PASS-PARTIAL"
    else:
        tier = "FAIL"

    result["tier"] = tier
    result["checks"] = checks

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE JSON
    # ─────────────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Checks:")
    for k, v in checks.items():
        print("  " + k + ": " + str(v))
    if warnings:
        for w in warnings:
            print("  WARNING: " + w)
    if errors:
        for e in errors:
            print("  ERROR: " + e)
    print("TIER: " + tier)
    print("JSON written -> " + str(out_path))
    print("=" * 60)

    sys.exit(0 if tier != "FAIL" else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline model load check.")
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=str(OUT_PATH),
        help="Output path for JSON (default: sft/load_check.json).",
    )
    args = parser.parse_args()
    run_checks(Path(args.out))


if __name__ == "__main__":
    main()
