"""
sft/gpu_probe.py — Kaggle GPU verification utility
====================================================
Standalone GPU probe. stdlib + optional torch only. No network calls.
Designed to run at the start of a Kaggle training session to confirm
that a GPU is available and report its capabilities.

Output (first four lines always):
  device: <name> | none (<reason>)
  vram: <decimal_GB> GB  (<GiB> GiB)  free: <decimal_GB> GB (<GiB> GiB)
  compute_capability: <X.Y>
  bf16_native: <true|false>   # true only if cc >= 8.0

Then: torch/CUDA versions, SDPA backends, fp16/bf16 matmul finite checks.
No GPU: prints "device: none (<reason>)", still writes JSON, exits 0.

Usage (run in the Kaggle notebook or training script):
    python sft/gpu_probe.py
    python sft/gpu_probe.py --json /kaggle/working/gpu_probe_kaggle.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path


def _bytes_to_gb_gib(n: int) -> tuple[float, float]:
    """Return (decimal_GB, binary_GiB) for a byte count."""
    return round(n / 1e9, 3), round(n / (1024 ** 3), 3)


def probe() -> dict:
    """Run all checks and return a result dict."""
    result: dict = {
        "device": None,
        "vram_total_gb": None,
        "vram_total_gib": None,
        "vram_free_gb": None,
        "vram_free_gib": None,
        "compute_capability": None,
        "bf16_native": None,
        "torch_version": None,
        "cuda_version": None,
        "sdpa_backends": [],
        "fp16_matmul_finite": None,
        "bf16_matmul_finite": None,
        "no_cuda_reason": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # ── try importing torch ───────────────────────────────────────────────────
    try:
        import torch  # type: ignore
    except ImportError:
        result["device"] = "none"
        result["no_cuda_reason"] = "torch not importable"
        return result

    result["torch_version"] = torch.__version__
    result["cuda_version"] = torch.version.cuda

    # ── check CUDA availability ───────────────────────────────────────────────
    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() returned False"
        try:
            # Attempt to surface a better reason
            _ = torch.zeros(1, device="cuda")
        except RuntimeError as exc:
            reason = str(exc).split("\n")[0]
        result["device"] = "none"
        result["no_cuda_reason"] = reason
        return result

    # ── device info ───────────────────────────────────────────────────────────
    dev = torch.device("cuda:0")
    result["device"] = torch.cuda.get_device_name(0)

    cc = torch.cuda.get_device_capability(0)
    cc_str = str(cc[0]) + "." + str(cc[1])
    result["compute_capability"] = cc_str

    # bf16 native: cc >= 8.0  (Ampere+); cc < 8.0 has emulated bf16 only
    result["bf16_native"] = (cc[0] > 8) or (cc[0] == 8 and cc[1] >= 0)

    # ── VRAM ──────────────────────────────────────────────────────────────────
    mem = torch.cuda.mem_get_info(0)   # (free_bytes, total_bytes)
    free_bytes, total_bytes = mem
    total_gb, total_gib = _bytes_to_gb_gib(total_bytes)
    free_gb, free_gib = _bytes_to_gb_gib(free_bytes)
    result["vram_total_gb"]  = total_gb
    result["vram_total_gib"] = total_gib
    result["vram_free_gb"]   = free_gb
    result["vram_free_gib"]  = free_gib

    # ── torch/CUDA version detail ─────────────────────────────────────────────
    result["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

    # ── SDPA backends ─────────────────────────────────────────────────────────
    sdpa_working: list[str] = []
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
        # Map of name -> enum value
        backend_candidates = {
            "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
            "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
            "MATH": SDPBackend.MATH,
        }
        # Add CUDNN if available (torch >= 2.3)
        try:
            backend_candidates["CUDNN_ATTENTION"] = SDPBackend.CUDNN_ATTENTION
        except AttributeError:
            pass

        B, H, S, D = 1, 4, 16, 32
        q = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)

        for name, backend in backend_candidates.items():
            try:
                with sdpa_kernel(backend):
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                if out is not None and out.shape == (B, H, S, D):
                    sdpa_working.append(name)
            except Exception:
                pass
    except Exception:
        # Fallback: just record that SDPA check failed
        sdpa_working = ["SDPA_CHECK_FAILED"]

    result["sdpa_backends"] = sdpa_working

    # ── fp16 matmul finite check ──────────────────────────────────────────────
    try:
        a = torch.randn(64, 64, device=dev, dtype=torch.float16)
        b = torch.randn(64, 64, device=dev, dtype=torch.float16)
        c = torch.matmul(a, b)
        result["fp16_matmul_finite"] = bool(torch.isfinite(c).all().item())
    except Exception as exc:
        result["fp16_matmul_finite"] = False
        result["fp16_matmul_error"] = str(exc)

    # ── bf16 matmul finite check ──────────────────────────────────────────────
    try:
        a = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
        b = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        result["bf16_matmul_finite"] = bool(torch.isfinite(c).all().item())
    except Exception as exc:
        result["bf16_matmul_finite"] = False
        result["bf16_matmul_error"] = str(exc)

    return result


def format_lines(r: dict) -> list[str]:
    """Return the canonical first-4-line output plus extended info."""
    lines: list[str] = []

    # Line 1: device
    if r["device"] == "none":
        lines.append("device: none (" + (r.get("no_cuda_reason") or "unknown") + ")")
    else:
        lines.append("device: " + str(r["device"]))

    # Line 2: vram
    if r["vram_total_gb"] is not None:
        lines.append(
            "vram: " + str(r["vram_total_gb"]) + " GB"
            + "  (" + str(r["vram_total_gib"]) + " GiB)"
            + "  free: " + str(r["vram_free_gb"]) + " GB"
            + "  (" + str(r["vram_free_gib"]) + " GiB)"
        )
    else:
        lines.append("vram: N/A")

    # Line 3: compute_capability
    lines.append("compute_capability: " + (str(r["compute_capability"]) if r["compute_capability"] else "N/A"))

    # Line 4: bf16_native
    bf16 = r.get("bf16_native")
    lines.append("bf16_native: " + (str(bf16).lower() if bf16 is not None else "N/A"))

    # Extended info
    lines.append("---")
    lines.append("torch: " + (str(r.get("torch_version")) or "N/A"))
    lines.append("cuda_version: " + (str(r.get("cuda_version")) or "N/A"))
    lines.append("cudnn_version: " + (str(r.get("cudnn_version")) or "N/A"))
    lines.append("sdpa_backends: " + str(r.get("sdpa_backends", [])))
    lines.append("fp16_matmul_finite: " + str(r.get("fp16_matmul_finite")))
    lines.append("bf16_matmul_finite: " + str(r.get("bf16_matmul_finite")))
    lines.append("python: " + str(r.get("python")))
    lines.append("platform: " + str(r.get("platform")))

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU probe — standalone, no network.")
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Path to write JSON output (e.g. sft/gpu_probe_local.json).",
    )
    args = parser.parse_args()

    r = probe()
    lines = format_lines(r)

    for line in lines:
        print(line)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print("\nJSON written -> " + str(out_path))

    # Exit 0 regardless of GPU presence (per spec)
    sys.exit(0)


if __name__ == "__main__":
    main()
