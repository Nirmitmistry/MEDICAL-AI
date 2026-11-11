"""
sft/kaggle_env_check.py — Kaggle training environment verifier
===============================================================
Run this at the start of every Kaggle training session, before installing
packages or loading the model.

Checks (in order):
  1. Kaggle environment: KAGGLE_KERNEL_RUN_TYPE must be set.
  2. CUDA available: torch.cuda.is_available() must return True.
  3. GPU device: name and compute capability are printed.
  4. VRAM: reports total and free; warns if free < MIN_VRAM_GB.
  5. Precision: selects bf16 (Ampere+ / cc >= 8.0) or fp16 (Turing / T4).
  6. Offline mode: warns if HF_HUB_OFFLINE is not set.
  7. Package versions: records torch/transformers/peft/accelerate; warns if
     transformers differs from the pinned CPU-env version in sft/freeze.txt.
  8. CUDA matmul+backward smoke test: catches "no kernel image" errors on
     older GPUs (P100) where certain CUDA kernels may not be compiled for
     the installed torch build.

Exit codes:
  0 — all required checks passed; training may proceed.
  1 — a required check failed; do NOT proceed.

Failures are loud, descriptive, and actionable. This script never falls
back silently to an unexpected environment.

Usage (inside Kaggle notebook):
    !python sft/kaggle_env_check.py
    !python sft/kaggle_env_check.py --json /kaggle/working/kaggle_env.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── freeze.txt reference (transformers version pinned in CPU env) ───────────
_SFT_DIR = Path(__file__).resolve().parent
_FREEZE_TRANSFORMERS = "5.20.2"   # authoritative pin from sft/freeze.txt

# ── configuration ──────────────────────────────────────────────────────────────
# Minimum free VRAM to attempt training (warning threshold, not hard stop).
# Qwen3-1.7B LoRA fp16 peak at micro-batch 1 is ~7.3 GB; allow some slack.
MIN_VRAM_GB = 10.0

# Minimum VRAM to even begin loading the model.
HARD_MIN_VRAM_GB = 6.0


def _bytes_to_gb_gib(n: int) -> tuple[float, float]:
    return round(n / 1e9, 3), round(n / (1024 ** 3), 3)


def run_checks() -> dict:
    """Run all Kaggle environment checks. Returns result dict."""
    result: dict = {
        "kaggle_env": None,
        "cuda_available": None,
        "device_name": None,
        "compute_capability": None,
        "bf16_native": None,
        "precision": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "torch_version": None,
        "cuda_version": None,
        "hf_hub_offline": None,
        "versions": {},
        "smoke_test_passed": None,
        "errors": [],
        "warnings": [],
        "ready": False,
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    # -- Check 1: Kaggle environment --------------------------------------------
    print("\n[1] Kaggle environment check ...")
    kernel_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
    if kernel_type:
        result["kaggle_env"] = kernel_type
        print(f"    KAGGLE_KERNEL_RUN_TYPE={kernel_type!r}  [OK]")
    else:
        result["kaggle_env"] = "NOT_SET"
        errors.append(
            "KAGGLE_KERNEL_RUN_TYPE is not set. "
            "This script must run inside a Kaggle notebook session. "
            "Do not run training on your local machine."
        )
        print("    FAIL: KAGGLE_KERNEL_RUN_TYPE not set.")

    # -- Check 2: torch import -------------------------------------------------
    print("\n[2] torch import ...")
    try:
        import torch  # type: ignore
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        print(f"    torch={torch.__version__}  cuda_build={torch.version.cuda}  [OK]")
    except ImportError:
        errors.append(
            "torch is not importable. "
            "Kaggle sessions include PyTorch by default -- check the accelerator setting."
        )
        print("    FAIL: torch not importable.")
        result["ready"] = False
        return result

    # -- Check 3: CUDA availability ---------------------------------------------
    print("\n[3] CUDA availability ...")
    if not torch.cuda.is_available():
        result["cuda_available"] = False
        reason = "torch.cuda.is_available() returned False"
        try:
            torch.zeros(1, device="cuda")
        except RuntimeError as exc:
            reason = str(exc).split("\n")[0]
        errors.append(
            "No CUDA GPU found: " + reason + ". "
            "Enable a GPU accelerator in Kaggle: Settings -> Accelerator -> GPU T4/P100."
        )
        print("    FAIL: " + reason)
        result["ready"] = False
        return result

    result["cuda_available"] = True
    device_name = torch.cuda.get_device_name(0)
    result["device_name"] = device_name

    cc = torch.cuda.get_device_capability(0)
    cc_str = f"{cc[0]}.{cc[1]}"
    result["compute_capability"] = cc_str

    bf16_native = (cc[0] > 8) or (cc[0] == 8 and cc[1] >= 0)
    result["bf16_native"] = bf16_native
    precision = "bf16" if bf16_native else "fp16"
    result["precision"] = precision

    print(f"    device={device_name!r}  cc={cc_str}  bf16_native={bf16_native}  [OK]")
    print(f"    -> selected training precision: {precision}")

    # -- Check 4: VRAM ----------------------------------------------------------
    print("\n[4] VRAM check ...")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_gb, total_gib = _bytes_to_gb_gib(total_bytes)
    free_gb, free_gib = _bytes_to_gb_gib(free_bytes)
    result["vram_total_gb"] = total_gb
    result["vram_free_gb"] = free_gb
    print(f"    total={total_gb} GB ({total_gib} GiB)  free={free_gb} GB ({free_gib} GiB)")

    if free_gb < HARD_MIN_VRAM_GB:
        errors.append(
            f"Insufficient VRAM: {free_gb} GB free < {HARD_MIN_VRAM_GB} GB minimum. "
            "The model cannot be loaded. Restart the Kaggle session to free VRAM."
        )
        print(f"    FAIL: free VRAM {free_gb} GB < hard minimum {HARD_MIN_VRAM_GB} GB.")
    elif free_gb < MIN_VRAM_GB:
        warnings.append(
            f"Low free VRAM: {free_gb} GB free < {MIN_VRAM_GB} GB recommended. "
            "Training may OOM at the configured micro-batch size. "
            "Restart the session and start training immediately to reclaim VRAM."
        )
        print(f"    WARNING: free VRAM {free_gb} GB < recommended {MIN_VRAM_GB} GB.")
    else:
        print(f"    free VRAM {free_gb} GB >= {MIN_VRAM_GB} GB  [OK]")

    # -- Check 5: HF_HUB_OFFLINE -----------------------------------------------
    print("\n[5] HF_HUB_OFFLINE check ...")
    hf_offline = os.environ.get("HF_HUB_OFFLINE", "")
    result["hf_hub_offline"] = hf_offline
    if hf_offline == "1":
        print("    HF_HUB_OFFLINE=1  (offline mode active)  [OK]")
    else:
        warnings.append(
            "HF_HUB_OFFLINE is not set to '1'. "
            "Set os.environ['HF_HUB_OFFLINE'] = '1' before loading models "
            "to prevent accidental network calls to HuggingFace Hub."
        )
        print("    WARNING: HF_HUB_OFFLINE not set; model loading may attempt network.")

    # -- Check 6: Package versions + freeze.txt alignment ----------------------
    print("\n[6] Package versions ...")
    versions: dict = {}
    for pkg in ("torch", "transformers", "peft", "accelerate"):
        try:
            import importlib.metadata as _meta
            versions[pkg] = _meta.version(pkg)
        except Exception:
            versions[pkg] = "NOT_INSTALLED"
    result["versions"] = versions

    for pkg, ver in versions.items():
        installed = ver if ver != "NOT_INSTALLED" else "(not installed)"
        print(f"    {pkg}=={installed}")

    # Warn if transformers differs from freeze.txt pin
    tf_ver = versions.get("transformers", "NOT_INSTALLED")
    if tf_ver != _FREEZE_TRANSFORMERS and tf_ver != "NOT_INSTALLED":
        warnings.append(
            f"transformers version mismatch: freeze.txt pins {_FREEZE_TRANSFORMERS!r} "
            f"but installed version is {tf_ver!r}. "
            "Renamed arguments between major versions may cause errors. "
            "Verify the training script is compatible with the installed version."
        )
        print(f"    WARNING: transformers {tf_ver!r} != freeze.txt {_FREEZE_TRANSFORMERS!r}")
    elif tf_ver == "NOT_INSTALLED":
        warnings.append(
            "transformers is not installed. "
            "Install with: pip install -r sft/requirements-kaggle.txt"
        )
        print("    WARNING: transformers not installed.")
    else:
        print(f"    transformers version matches freeze.txt ({_FREEZE_TRANSFORMERS!r})  [OK]")

    # -- Check 7: CUDA matmul+backward smoke test --------------------------------
    # Catches "no kernel image for this device" errors on P100 (sm_60) or other
    # older GPUs where some torch kernels may not be compiled for the device arch.
    print("\n[7] CUDA matmul+backward smoke test ...")
    try:
        import torch as _torch_smoke  # already imported above but re-bind safely
        if not _torch_smoke.cuda.is_available():
            result["smoke_test_passed"] = None
            print("    SKIPPED (no CUDA device)")
        else:
            _dev = _torch_smoke.device("cuda:0")
            # Use the precision already selected to match real training conditions
            _dtype = _torch_smoke.bfloat16 if result.get("precision") == "bf16" else _torch_smoke.float16
            # Small matmul that requires a compiled CUDA kernel
            _a = _torch_smoke.randn(64, 64, device=_dev, dtype=_dtype, requires_grad=False)
            _b = _torch_smoke.randn(64, 64, device=_dev, dtype=_dtype, requires_grad=True)
            _c = _torch_smoke.matmul(_a, _b)
            # Backward pass (catches "no kernel image" for grad kernels)
            _c.sum().backward()
            _grad_finite = bool(_torch_smoke.isfinite(_b.grad).all().item())
            _fwd_finite = bool(_torch_smoke.isfinite(_c).all().item())
            if not (_fwd_finite and _grad_finite):
                errors.append(
                    "CUDA smoke test: matmul or grad not finite "
                    f"(fwd={_fwd_finite}, grad={_grad_finite}). "
                    "Possible fp16 overflow — consider bf16 or gradient clipping."
                )
                result["smoke_test_passed"] = False
                print(f"    FAIL: fwd_finite={_fwd_finite}  grad_finite={_grad_finite}")
            else:
                result["smoke_test_passed"] = True
                print(f"    dtype={_dtype}  fwd_finite=True  grad_finite=True  [OK]")
            # Clean up
            del _a, _b, _c
            _torch_smoke.cuda.empty_cache()
    except RuntimeError as _exc:
        _msg = str(_exc)
        if "no kernel image" in _msg.lower() or "CUDA error" in _msg:
            errors.append(
                "CUDA smoke test failed with kernel-image error: " + _msg.split("\n")[0] + ". "
                "The installed PyTorch build may not support this GPU architecture. "
                "Try a different Kaggle GPU accelerator or a torch wheel built for this device."
            )
        else:
            errors.append("CUDA smoke test RuntimeError: " + _msg.split("\n")[0])
        result["smoke_test_passed"] = False
        print("    FAIL: " + _msg.split("\n")[0][:120])
    except Exception as _exc:
        warnings.append("CUDA smoke test exception (non-fatal): " + str(_exc).split("\n")[0])
        result["smoke_test_passed"] = False
        print("    WARNING (non-fatal): " + str(_exc).split("\n")[0][:120])

    # -- Summary ----------------------------------------------------------------
    result["ready"] = len(errors) == 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle training environment verifier. Exits 1 on any failure."
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write result JSON to this path (e.g. /kaggle/working/kaggle_env.json).",
    )
    args = parser.parse_args()

    result = run_checks()

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print("  WARNING: " + w)
    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print("  ERROR: " + e)

    if result["ready"]:
        print(
            "\nREADY: Kaggle GPU environment verified. "
            f"Precision={result['precision']}  "
            f"VRAM={result['vram_total_gb']} GB  "
            f"Device={result['device_name']!r}  "
            f"torch={result['versions'].get('torch', '?')}  "
            f"transformers={result['versions'].get('transformers', '?')}"
        )
    else:
        print(
            "\nNOT READY: Environment check failed. "
            "Fix the errors above before running training."
        )
    print("=" * 60)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nJSON written -> {out_path}")

    sys.exit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
