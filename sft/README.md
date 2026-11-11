# sft/ — Training pipeline scaffold

Base model: `Qwen/Qwen3-1.7B` (Apache-2.0)
Revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

---

## Environment overview

| Location | Purpose |
|---|---|
| **Local PC** | CPU-only development: tokenizer checks, smoke tests, manifest verification, data preparation |
| **Kaggle** | GPU-dependent operations: model full load, LoRA fine-tuning, NLL evaluation |

The local GPU (RTX 3050, 4 GB) is **not used** for training — it is insufficient for LoRA fine-tuning.

---

## Local CPU environment

### Create the environment

```powershell
# From repo root (Windows PowerShell)
python -m venv sft/.venv
sft/.venv/Scripts/pip install --no-cache-dir -r sft/requirements-cpu.txt
```

### Run the smoke tests

```powershell
sft/.venv/Scripts/python -m unittest discover -v -s sft/tests
```

All tests must pass. The tokenizer test requires `sft/tokenizer_snapshot/` to exist.

### Run the manifest verifier

```powershell
sft/.venv/Scripts/python sft/verify_manifest.py
```

Exit 0 = all protected files match. Exit 1 = something changed or is missing.
Run this at the end of every task.

### Run P1a — G1 gate + E1 profile (Stage 1, step a)

```powershell
# Full run (writes sft/reports/g1_report.json and sft/reports/e1_profile.json):
sft\.venv-nb\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 sft/notebooks/P1a_g1_e1_profile.ipynb

# Fast dev run (first 2000 dialogues only):
$env:P1A_LIMIT_DIALOGUES=2000
sft\.venv-nb\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 sft/notebooks/P1a_g1_e1_profile.ipynb
$env:P1A_LIMIT_DIALOGUES=""
```

Uses `sft/.venv-nb`. Full run ~3.4 minutes on CPU. Outputs: `sft/reports/g1_report.json`, `sft/reports/e1_profile.json`, `sft/work/e1/*.tsv` (git-ignored).

### Run the local model check

```powershell
sft/.venv/Scripts/python sft/load_check.py
```

Expected local result: **tier = PASS-PARTIAL**
- A_tokenizer: PASS
- B_config: PASS
- C_structure: PASS
- D_full_load: DEFERRED (insufficient local RAM — run on Kaggle)
- E_peft_params: DEFERRED (run on Kaggle)

Checks D and E require a GPU with ≥6 GB VRAM. Run `load_check.py` on Kaggle at the start of training to get a full PASS.

### Rebuild the protected manifest (after intentional changes only)

```powershell
sft/.venv/Scripts/python sft/build_manifest.py
```

---

## Tokenizer snapshot (already downloaded)

Stored in `sft/tokenizer_snapshot/` (git-ignored). No weights — config and tokenizer files only.
Re-download at the pinned revision:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json",
                    "generation_config.json", "LICENSE"],
    ignore_patterns=["*.safetensors", "*.bin", "*.gguf"],
    local_dir="sft/tokenizer_snapshot",
)
```

---

## Kaggle training environment

For training, see the Kaggle setup instructions in `docs/IMPLEMENTATION_PLAN.md` → Stage 3.

**Key files for Kaggle:**

| File | Purpose |
|---|---|
| `sft/kaggle_env_check.py` | Verifies Kaggle session: GPU present, VRAM sufficient, precision selected |
| `sft/gpu_probe.py` | Reports GPU details for the run manifest |
| `sft/requirements-kaggle.txt` | Installs `peft` and `accelerate` on top of Kaggle's pre-installed torch |
| `sft/load_check.py` | Full model check (A–E) when run with GPU |

**Startup sequence (Kaggle notebook):**

```python
import subprocess, os

# 1. Must pass before proceeding
subprocess.run(["python", "sft/kaggle_env_check.py"], check=True)

# 2. Install missing packages (torch pre-installed by Kaggle)
subprocess.run(["pip", "install", "-q", "-r", "sft/requirements-kaggle.txt"], check=True)

# 3. Set offline mode before loading models
os.environ["HF_HUB_OFFLINE"] = "1"
```

---

## File inventory

| File | Runs on | Purpose |
|---|---|---|
| `kaggle_env_check.py` | Kaggle | Session verifier: GPU, VRAM, precision selection |
| `gpu_probe.py` | Kaggle | Detailed GPU diagnostic for run manifest |
| `load_check.py` | Local (A–C) / Kaggle (D–E) | Model tokenizer + structure + load check |
| `download_model.py` | Local | Download and SHA-256 verify Qwen3-1.7B weights |
| `build_manifest.py` | Local | Hash protected files into `protected_manifest.json` |
| `verify_manifest.py` | Local | Stdlib-only manifest verifier |
| `requirements-cpu.txt` | Local | CPU dev environment dependencies |
| `requirements-kaggle.txt` | Kaggle | Training dependencies (peft, accelerate) |
| `freeze.txt` | Local | Full CPU pip freeze |
| `freeze-gpu.txt` | — | Historical local GPU venv (RTX 3050; not for Kaggle) |
| `requirements-gpu.txt` | — | Superseded; RTX 3050-specific; not for Kaggle |
| `environment_manifest.json` | Local | CPU-stage environment record |
| `tests/test_smoke.py` | Local | CPU smoke tests |
