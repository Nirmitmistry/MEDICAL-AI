# Design Document: P4 — Final LoRA Training on Kaggle

## Overview

P4 wraps the frozen training artefacts from P3 without touching them. The design divides work into two execution contexts: a **local CPU stage (Part A)** that produces the acceptance checker, its unit tests, and the Kaggle upload bundle; and a **remote/post-transfer stage (Part B)** that runs after a human carries the Kaggle results back to the local machine.

The training script (`sft/train_lora.py`) and config (`sft/config/train_config.yaml`) are read-only inputs to P4. The notebook calls them as a subprocess; it never imports them or patches their contents.

---

## Architecture and Component Overview

```
Part A (local CPU, sft/.venv)
├── sft/p4_check.py              ← stdlib-only acceptance checker
├── sft/tests/test_p4_check.py   ← unit tests with synthetic fixtures
└── sft/kaggle/p4/
    ├── upload_manifest.json     ← SHA-256 manifest of all Kaggle upload files
    ├── p4_kaggle_notebook.ipynb ← 7-section execution notebook
    └── README.md                ← human steps

Part B (local CPU, after human transfer)
  sft/results/p4/ (git-ignored, human-placed)
  ├── files_sha256.json
  ├── run_manifest.json
  ├── log_history.json
  ├── session_log_<n>.json
  ├── checkpoint-*/              ← adapter files
  └── checkpoint_for_p5.json    ← written by B3

  sft/frozen_manifest_p4.json   ← written by B4
  sft/P4_REPORT.md              ← written at end of A and B
```

Data flow: frozen JSONL files → Kaggle GPU session → results/ → `p4_check.py` → `frozen_manifest_p4.json`.

---

## Component Designs

### 1. Acceptance Checker (`sft/p4_check.py`)

**Responsibility:** Read the results directory and report every acceptance criterion (§5 of requirements) as PASS or FAIL. Stdlib-only. No imports of torch, peft, accelerate, or any package outside the standard library.

#### 1.1 Invocation

```python
# CLI
python sft/p4_check.py [--results-dir PATH] [--frozen-manifest PATH] [--report-json PATH]

# Defaults
--results-dir      sft/results/p4
--frozen-manifest  sft/frozen_manifest_p3.json
```

Exit code 0 if all criteria pass; exit code 1 if any criterion fails; exit code 2 on input error (missing directory, malformed JSON).

#### 1.2 Criterion catalogue

Each criterion is implemented as a named function returning `(status: str, detail: str)` where `status` is `"pass"` or `"fail"`.

| ID | Name | Input | Rule |
|---|---|---|---|
| C1 | `frozen_hashes_ok` | `frozen_manifest_p3.json` | Re-compute SHA-256 of each listed file; all must match |
| C2 | `script_implements_best_macro` | `sft/train_lora.py` source text | Assert `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `SAVE_TOTAL_LIMIT = 3` appear in source |
| C3 | `run_completed` | `log_history.json` | Last step in log == planned total steps from frozen config |
| C4 | `val_nll_bounded` | `log_history.json` | Final macro val NLL ≤ min macro val NLL over all evals + 0.05 |
| C5 | `no_nan_inf` | `log_history.json` | No NaN/inf in `train_loss`, `grad_norm`, `eval_macro` at any step |
| C6 | `no_spike_loss` | `log_history.json` | No step where `train_loss - prev_train_loss > 0.5` |
| C7 | `no_spike_macro` | `log_history.json` | No eval where `eval_macro > 1.5 × min_eval_macro_so_far` |
| C8 | `resume_step_continuity` | all `session_log_<n>.json` | No duplicated or skipped step numbers across sessions |
| C9 | `resume_lr_no_restart` | `log_history.json` | LR does not drop below 10% of peak LR after session-1 warmup end |
| C10 | `run_manifest_fields` | `run_manifest.json` | All required top-level and nested keys present (§2.7) |
| C11 | `run_manifest_values` | `run_manifest.json` | Fixed values match frozen config (§2.8) |
| C12 | `data_hashes_match` | `run_manifest.json`, `frozen_manifest_p3.json` | `train_sha256`, `val_md_sha256`, `val_hc_sha256` match frozen values |
| C13 | `adapter_hyperparams` | adapter `adapter_config.json` per checkpoint | r=16, alpha=32, dropout=0.05, no lm_head/embed_tokens in target_modules |
| C14 | `adapter_dtype_f32` | adapter `adapter_model.safetensors` headers | LoRA weight tensors stored as F32 |
| C15 | `checkpoints_present` | results dir | At least 1 checkpoint dir present; at most 4 (last 3 + best) |
| C16 | `checkpoint_for_p5_recorded` | `checkpoint_for_p5.json` | File exists, has required fields, checkpoint dir it references exists |

#### 1.3 Spike detection logic

```python
def check_no_spike_loss(log: list[dict]) -> tuple[str, str]:
    # Only steps where train_loss is present (not NaN/inf after C5 passes)
    steps = [(e["step"], e["train_loss"]) for e in log if "train_loss" in e]
    for i in range(1, len(steps)):
        delta = steps[i][1] - steps[i-1][1]
        if delta > 0.5:
            return "fail", f"step {steps[i][0]}: loss rose {delta:.3f} from previous"
    return "pass", f"{len(steps)} train steps checked"

def check_no_spike_macro(log: list[dict]) -> tuple[str, str]:
    evals = [(e["step"], e["eval_macro"]) for e in log if "eval_macro" in e]
    min_so_far = float("inf")
    for step, macro in evals:
        if min_so_far < float("inf") and macro > 1.5 * min_so_far:
            return "fail", f"step {step}: eval_macro {macro:.4f} > 1.5× min {min_so_far:.4f}"
        min_so_far = min(min_so_far, macro)
    return "pass", f"{len(evals)} eval steps checked"
```

#### 1.4 Resume continuity logic

Session logs are sorted by session number. First/last step fields reconstruct the global sequence:

```python
def check_resume_step_continuity(session_logs: list[dict]) -> tuple[str, str]:
    # Sort by session_number
    logs = sorted(session_logs, key=lambda s: s["session_number"])
    for i in range(1, len(logs)):
        prev_last = logs[i-1]["last_step"]
        curr_first = logs[i]["first_step"]
        if curr_first == prev_last:
            return "fail", f"session {i} starts at step {curr_first} = session {i-1} last step (duplicate)"
        if curr_first > prev_last + 1:
            return "fail", f"gap: session {i-1} ended at {prev_last}, session {i} starts at {curr_first}"
    return "pass", f"{len(logs)} sessions, continuity ok"
```

#### 1.5 LR-restart detection

Warmup end for session 1 is `first_step + ceil(warmup_ratio * total_steps)`. After this point, LR should decrease monotonically (cosine). A restart is flagged if LR increases by > 5% of peak after warmup end.

#### 1.6 Safetensors header parsing

Adapter dtype checking reads only the safetensors header without loading tensors into memory:

```python
def _read_safetensors_metadata(path: Path) -> dict:
    """Read the JSON header from a safetensors file without loading tensors."""
    with open(path, "rb") as f:
        n_bytes = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(n_bytes)
    return json.loads(header_bytes.decode("utf-8"))
```

Each tensor entry in the metadata has a `dtype` field. All LoRA weight tensors (`lora_A`, `lora_B`, `lora_embedding_A`, `lora_embedding_B`) must have `dtype == "F32"`.

#### 1.7 Report output

When `--report-json` is given, writes:
```json
{
  "schema": "p4_check_report_v1",
  "results_dir": "/abs/path/to/results",
  "criteria": [
    {"id": "C1", "name": "frozen_hashes_ok", "status": "pass", "detail": "12 files verified"},
    ...
  ],
  "overall": "pass",
  "n_pass": 16,
  "n_fail": 0
}
```

---

### 2. Unit Tests (`sft/tests/test_p4_check.py`)

**Responsibility:** Test every checker function against synthetic fixtures, using no real data files.

#### 2.1 Fixture construction strategy

Each test method builds an in-memory or temporary-directory fixture representing the minimal set of files needed for the criterion under test. The `p4_check` module exposes each criterion as a standalone function that accepts explicit inputs (Python dicts or Path objects) rather than always reading from disk — this makes unit testing straightforward without mocking file I/O.

#### 2.2 Module-level design for testability

`p4_check.py` organizes checker functions into two layers:

- **Logic functions** (pure, testable): accept parsed data structures (dicts, lists) as arguments and return `(status, detail)`. These are the primary test targets.
- **IO functions**: read files from disk and call logic functions. Not unit-tested.

```python
# Example: logic function (tested)
def _check_val_nll_bounded(log_entries: list[dict]) -> tuple[str, str]: ...

# Example: IO wrapper (not unit-tested)
def check_val_nll_bounded(results_dir: Path) -> tuple[str, str]:
    log = _load_log_history(results_dir / "log_history.json")
    return _check_val_nll_bounded(log)
```

#### 2.3 Required test cases

| Test method | Scenario |
|---|---|
| `test_nan_in_train_loss` | `train_loss` = NaN → C5 fails |
| `test_inf_in_grad_norm` | `grad_norm` = inf → C5 fails |
| `test_nan_in_eval_macro` | `eval_macro` = NaN → C5 fails |
| `test_val_nll_bounded_pass` | final macro == min + 0.04 → C4 passes |
| `test_val_nll_bounded_fail` | final macro == min + 0.06 → C4 fails |
| `test_spike_loss_pass` | all deltas ≤ 0.5 → C6 passes |
| `test_spike_loss_fail` | one delta = 0.51 → C6 fails |
| `test_spike_macro_pass` | all macros < 1.5× running min → C7 passes |
| `test_spike_macro_fail` | one macro = 1.51× running min → C7 fails |
| `test_resume_duplicate_step` | session 2 starts at same step as session 1 ends → C8 fails |
| `test_resume_skipped_step` | session 2 starts 2 steps after session 1 ends → C8 fails |
| `test_resume_continuity_pass` | three sessions, clean handoff → C8 passes |
| `test_lr_restart_fail` | LR rises 10% after warmup end → C9 fails |
| `test_lr_restart_pass` | LR monotone after warmup end → C9 passes |
| `test_manifest_missing_field` | `run_manifest` lacks `seed` → C10 fails |
| `test_manifest_wrong_model_revision` | revision differs → C11 fails |
| `test_manifest_wrong_lora_r` | `lora.r == 32` → C11 fails |
| `test_data_hash_mismatch` | `train_sha256` differs from frozen → C12 fails |
| `test_adapter_wrong_r` | `adapter_config.json` has `r=32` → C13 fails |
| `test_adapter_lm_head_in_targets` | `lm_head` in `target_modules` → C13 fails |
| `test_adapter_dtype_bf16` | safetensors header shows `BF16` for a lora weight → C14 fails |
| `test_adapter_dtype_f32_pass` | all lora weights are `F32` → C14 passes |
| `test_missing_best_checkpoint` | only 1 checkpoint dir, it's not the best → C15 edge case |

---

### 3. Kaggle Upload Manifest (`sft/kaggle/p4/upload_manifest.json`)

**Responsibility:** List every file the Kaggle notebook needs, with SHA-256 and size, so `verify_uploads.py` can confirm integrity before the run.

#### 3.1 File groups

| Group | Kaggle dataset | Files |
|---|---|---|
| Code | `capstone-tokenizer-sft-code` (reuse or update) | `sft/train_lora.py`, `sft/tokenize_mask.py`, `sft/kaggle_env_check.py`, `sft/gpu_probe.py`, `sft/kaggle/verify_uploads.py`, `sft/kaggle/p4/upload_manifest.json` |
| Tokenizer | same code dataset | `sft/tokenizer_pkg/` (all files) |
| Config | same code dataset | `sft/config/train_config.yaml`, `sft/freeze-kaggle.txt` |
| Training data | `capstone-train-data` | `sft/work/final_mix.jsonl`, `sft/work/val_meddialog_tok.jsonl`, `sft/work/val_healthchat_tok.jsonl` |
| Dev probes | `capstone-train-data` | `sft/work/probes_dev.jsonl` (not probes_final) |
| Model weights | `capstone-model-weights` | `sft/model_snapshot/` (all weight files) |

#### 3.2 Reuse detection

When generating the manifest, the builder computes the SHA-256 of each file and compares it to the corresponding entry in `sft/kaggle/upload_manifest.json` (the P3 manifest). If the hash matches, the entry is marked `"reused_from_p3": true`. This tells the human they do not need to re-upload that file.

#### 3.3 Manifest schema

```json
{
  "schema": "upload_manifest_v1",
  "created_at": "<ISO-8601>",
  "files": [
    {
      "path": "sft/work/final_mix.jsonl",
      "sha256": "<hex>",
      "size_bytes": 123456789,
      "kaggle_dataset": "capstone-train-data",
      "reused_from_p3": false
    }
  ]
}
```

---

### 4. Kaggle Notebook (`sft/kaggle/p4/p4_kaggle_notebook.ipynb`)

**Responsibility:** Orchestrate the final training run on Kaggle. Each section writes one JSON to `/kaggle/working/results/`. Cell outputs contain only counts, hashes, step numbers, loss values, and enum strings.

#### 4.1 Notebook structure

| Section | Cell IDs | Saves |
|---|---|---|
| 0. Parameters | `p4_0001` | — |
| 1. Definitions | `p4_0002`–`p4_0005` | — |
| 2. In-notebook tests | `p4_0010`–`p4_0019` | asserts only |
| 3. env | `p4_0020`–`p4_0029` | `results/env.json` |
| 4. verify | `p4_0030`–`p4_0039` | `results/verify.json` |
| 5. dry_run | `p4_0040`–`p4_0049` | `results/dry_run.json` |
| 6. train | `p4_0050`–`p4_0059` | writes on every eval |
| 7. post_train | `p4_0060`–`p4_0069` | `results/session_log_<n>.json`, `results/log_history.json`, `results/files_sha256.json` |
| 8. smoke_test | `p4_0070`–`p4_0079` | `results/smoke_test.json` |

#### 4.2 In-notebook tests (section 2)

These are synthetic-only tests that must pass via `assert` before any data or model is loaded. They cover:
- JSON round-trip for session log format.
- SHA-256 computation on a known in-memory byte string.
- Spike detection logic (3 synthetic log vectors: clean, loss spike, macro spike).
- Resume continuity logic (clean and gap cases).

If any `assert` fails, execution halts before the env section.

#### 4.3 env section

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # must be before any torch import
import torch
assert torch.cuda.device_count() == 1, "Expected exactly 1 GPU"

# kaggle_env_check.py
result = subprocess.run([sys.executable, "sft/kaggle_env_check.py", "--json", "results/kaggle_env.json"], ...)
assert result.returncode == 0

# pip freeze comparison
freeze_actual = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
with open("sft/freeze-kaggle.txt") as f:
    freeze_expected = f.read()
_compare_pip_freeze(freeze_actual, freeze_expected)   # raises on any pinned-package diff

# GPU compatibility check
cc_major, cc_minor = torch.cuda.get_device_capability(0)
required_precision = _read_config_precision("sft/config/train_config.yaml")
if required_precision == "bf16" and not (cc_major * 10 + cc_minor >= 80):
    raise RuntimeError("bf16 required by config but GPU cc < 8.0")

env_data = { "device_name": ..., "compute_capability": ..., "precision_check": "pass", ... }
_write_json("results/env.json", env_data)
```

The `_compare_pip_freeze` function extracts only packages that appear in `freeze-kaggle.txt` and compares their versions. Packages in the actual freeze but absent from `freeze-kaggle.txt` are allowed (Kaggle pre-installs extras). Packages in `freeze-kaggle.txt` absent from the actual freeze, or present with a different version, are errors.

#### 4.4 verify section

```python
# verify_uploads.py check
result = subprocess.run([sys.executable, "sft/kaggle/verify_uploads.py",
                         "--manifest", "sft/kaggle/p4/upload_manifest.json",
                         "--root", "/kaggle/input"], ...)
assert result.returncode == 0

# Tokenizer integrity
from tokenize_mask import load_tokenizer
tok = load_tokenizer(Path("sft/tokenizer_pkg"))
# EOS/PAD assertions are inside load_tokenizer

# Template and system-prompt hashes
pkg_manifest = json.loads((Path("sft/tokenizer_pkg") / "tokenizer_pkg_manifest.json").read_text())
assert pkg_manifest["template_sha256"] == FROZEN_TEMPLATE_SHA256
assert pkg_manifest["system_prompt_sha256"] == FROZEN_SYSTEM_PROMPT_SHA256

# T1–T6 on Kaggle's transformers (synthetic fixtures, same as test_tokenize_mask.py)
# T2–T5 on every item of final_mix.jsonl and both val sets
...
_write_json("results/verify.json", {"t1_t6": "pass", "item_level": {"final_mix": n_ok, ...}})
```

`FROZEN_TEMPLATE_SHA256` and `FROZEN_SYSTEM_PROMPT_SHA256` are written as constants in the Parameters cell, taken from `sft/tokenizer_pkg/tokenizer_pkg_manifest.json` at notebook-build time.

#### 4.5 dry_run section

```python
if DRY_RUN:
    dry_dir = Path("/kaggle/working/dry_run_out")
    result = subprocess.run([
        sys.executable, "sft/train_lora.py",
        "--train-tok", str(FINAL_MIX_JSONL),
        "--val-md-tok", str(VAL_MD_JSONL),
        "--val-hc-tok", str(VAL_HC_JSONL),
        "--out", str(dry_dir),
        "--n-steps", "10",           # override only steps and output dir
        "--lr", str(FROZEN_LR),      # re-state frozen value (not changing it)
        "--micro-batch", str(FROZEN_MICRO_BATCH),
    ], ...)
    # Parse trainer logs for step-10 loss
    # Resume from dry_dir checkpoint for 5 more steps
    _write_json("results/dry_run.json", {...})
    print("dry run done — set DRY_RUN=False for the real run")
    raise SystemExit(0)   # halt the kernel gracefully
```

The frozen config is not passed as a file argument because we override `n-steps`. The frozen config's LR, micro_batch, etc. are re-stated from constants defined in the Parameters cell; the final `train` section passes no overrides and relies on `train_config.yaml`.

#### 4.6 train section

```python
if not DRY_RUN:
    cmd = [sys.executable, "sft/train_lora.py",
           "--config", "sft/config/train_config.yaml",   # frozen, unmodified
           "--train-tok", str(FINAL_MIX_JSONL),
           "--val-md-tok", str(VAL_MD_JSONL),
           "--val-hc-tok", str(VAL_HC_JSONL),
           "--out", f"/kaggle/working/{RUN_ID}"]
    if RESUME_FROM:
        # Verify adapter hash of latest checkpoint matches session log
        _verify_resume_integrity(RESUME_FROM, prev_session_log)
        cmd += ["--resume", RESUME_FROM]
    # Run training as subprocess; flush log_history.json via eval callback or post-process
```

Wall-clock logging: at each eval step, record `time.monotonic()` and compute `elapsed_s`, `steps_remaining`, `eta_s`.

#### 4.7 post_train section

Log history accumulation:

```python
# Load trainer's log_history from run_manifest or checkpoint state
# Merge with any prior session logs if RESUME_FROM
# Write accumulated log_history.json

# Adapter hash
def _hash_adapter(ckpt_dir: Path) -> str:
    return sha256_file(ckpt_dir / "adapter_model.safetensors")

session_log = {
    "schema": "session_log_v1",
    "run_id": RUN_ID,
    "session_number": SESSION_NUMBER,
    "gpu_model": gpu_info["device_name"],
    "first_step": first_step,
    "last_step": last_step,
    "wall_clock_seconds": elapsed,
    "retained_checkpoints": [
        {"checkpoint_id": ckpt.name, "adapter_sha256": _hash_adapter(ckpt)}
        for ckpt in retained
    ]
}
```

Output packing: retain only `adapter_model.safetensors` + `adapter_config.json` for all but the last checkpoint; retain the full optimizer/RNG state for the last checkpoint so the run can be resumed.

#### 4.8 smoke_test section

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

best_ckpt = _pick_best_checkpoint(log_history)
base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, ...)
model = PeftModel.from_pretrained(base, best_ckpt, ...)
tok = AutoTokenizer.from_pretrained(PKG_DIR, ...)

probes = _load_probes(PROBES_DEV_JSONL)   # never probes_final
results = {"n_total": len(probes), "n_eos_terminated": 0, "n_think_tokens": 0, "n_empty": 0}
for probe in probes:
    ids = tok.apply_chat_template(probe["messages"], add_generation_prompt=True, return_tensors="pt")
    out = model.generate(ids, max_new_tokens=256, do_sample=False)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    if "<|im_end|>" in text: results["n_eos_terminated"] += 1
    results["n_think_tokens"] += text.count("<think>")
    if not text.strip(): results["n_empty"] += 1

_write_json("results/smoke_test.json", results)   # no probe text, no response text
```

#### 4.9 Output integrity

`files_sha256.json` is written at the end of `post_train`:

```python
output_dir = Path(f"/kaggle/working/{RUN_ID}")
hashes = {}
for p in sorted(output_dir.rglob("*")):
    if p.is_file():
        hashes[str(p.relative_to(output_dir))] = sha256_file(p)
_write_json("results/files_sha256.json", hashes)
```

---

### 5. Notebook Format and Execution Constraints

**5.1** Valid nbformat v4; `nbformat.validate` must pass; all cell ids are unique strings.

**5.2** A `parameters` cell is the first executable cell (section 0). Every path, constant, and toggle the human must set is in this cell only.

**5.3** A definitions section follows immediately (section 1), containing only imports and helper function definitions — no data processing.

**5.4** The in-notebook tests section (section 2) contains only synthetic assertions and must run before any data is read or any subprocess is launched.

**5.5** The final cell is a status cell that prints "ALL SECTIONS PASSED" if no earlier cell raised an exception. It writes no files.

**5.6** No cell output may contain corpus text. All `print()` calls output counts, ids, hashes, step numbers, loss values, or enum strings only.

---

### 6. Part B — Local CPU Processing

#### 6.1 B1 transfer integrity

```python
# python -c inline or sft/work/check_transfer.py (not committed)
import json, hashlib, sys
from pathlib import Path

results_dir = Path("sft/results/p4")
manifest = json.loads((results_dir / "files_sha256.json").read_text())
errors = []
for rel_path, expected_sha in manifest.items():
    fp = results_dir / rel_path
    if not fp.exists():
        errors.append(f"MISSING: {rel_path}")
        continue
    actual = sha256_file(fp)
    if actual != expected_sha:
        errors.append(f"CHANGED: {rel_path}")
if errors:
    for e in errors: print(e, file=sys.stderr)
    sys.exit(1)
print(f"B1 OK: {len(manifest)} files verified")
```

After passing B1, the human is instructed to mark the directory read-only (documented in README).

#### 6.2 B2 acceptance check

```bash
sft\.venv\Scripts\python.exe sft\p4_check.py --results-dir sft/results/p4 --report-json sft/results/p4/check_report.json
```

The checker output is printed to stdout (one line per criterion). The JSON report is written for the frozen manifest.

#### 6.3 B3 checkpoint selection

```python
# Logic embedded in p4_check.py --select-checkpoint flag, or inline in P4_REPORT.md generation
log = json.loads((results_dir / "log_history.json").read_text(encoding="utf-8"))
evals = [(e["step"], e["eval_macro"]) for e in log if "eval_macro" in e]
best_step, best_macro = min(evals, key=lambda x: (x[1], -x[0]))
# locate checkpoint dir for best_step
checkpoint_for_p5 = {
    "schema": "checkpoint_for_p5_v1",
    "checkpoint_id": f"checkpoint-{best_step}",
    "step": best_step,
    "eval_macro": best_macro,
    "val_md_nll": ...,
    "val_hc_nll": ...,
    "adapter_sha256": sha256_file(...),
    "selection_rule": "lowest eval_macro; ties broken by later step"
}
```

#### 6.4 B4 frozen manifest

`sft/frozen_manifest_p4.json` follows the same schema as `frozen_manifest_p3.json`:

```json
{
  "schema": "frozen_manifest_v1",
  "created_at": "<ISO-8601>",
  "entries": [
    {"path": "sft/p4_check.py", "sha256": "...", "size": 12345},
    ...
  ]
}
```

Entries covering `sft/results/p4/` are included for the non-private files (manifests, logs, checkpoint config). Adapter weight files are listed by path for record but marked `"private": true`; the manifest itself is committed. The agent runs `git check-ignore -v` on each path before including it in the git-add list.

---

### 7. Privacy Architecture

All data-touching functions follow one rule: inputs flow in as file paths or counts; outputs flow out as counts, hashes, step numbers, loss values, and identifiers. No text string from the corpus ever leaves a function through a return value, a print statement, or a JSON field.

The privacy constraint is enforced at three layers:
1. **Notebook cell outputs**: every `print()` call in the notebook produces only structured data.
2. **JSON files**: all `_write_json` calls receive dicts containing only the allowed types.
3. **p4_check.py**: checker detail strings reference step numbers and numeric values, not text.

---

### 8. Error Handling

| Scenario | Behavior |
|---|---|
| Any entry in `frozen_manifest_p3.json` has changed | `p4_check.py` C1 fails; Part A halts |
| `freeze-kaggle.txt` version mismatch | env section halts kernel before training |
| Upload manifest verification fails | verify section halts kernel |
| T1–T6 fail on Kaggle's transformers | verify section halts kernel |
| `DRY_RUN=True` after dry_run passes | execution stops at "dry run done" message |
| NaN/inf in first eval | C5 fails; p4_check exits 1 |
| Checkpoint directory missing at B1 | C15 fails |
| Frozen file appears wrong | Halt and report (never change) |
| Transfer integrity mismatch | B1 fails; B2–B4 do not proceed |

---

### 9. Correctness Properties

#### Property 1: Checker is idempotent

For any results directory that passes all criteria, running `p4_check.py` twice with the same `--results-dir` produces identical stdout and identical `--report-json` output.

**Validates: Requirements 2.2, 2.11**

---

#### Property 2: Spike detection is monotone in threshold

For any log history, reducing the spike threshold (e.g., from 0.5 to 0.3 for loss, or from 1.5× to 1.2× for macro) causes C6/C7 to report at least as many failures as before — never fewer.

**Validates: Requirements 2.4, 2.5**

---

#### Property 3: Session continuity check is order-independent on input ordering

For any list of session log dicts, `check_resume_step_continuity` returns the same result regardless of the order in which the dicts are passed in (because the function sorts by `session_number`).

**Validates: Requirements 2.6**

---

#### Property 4: Upload manifest round-trip integrity

For any set of files for which `upload_manifest.json` was generated, running `verify_uploads.py` exits 0; modifying any listed file or deleting it causes `verify_uploads.py` to exit 1 with a descriptive error.

**Validates: Requirements 4.1, 5.3**

---

#### Property 5: Checkpoint selection is deterministic

For any `log_history.json`, B3 checkpoint selection returns the same `checkpoint_id` on every run (lowest macro, ties by later step).

**Validates: Requirements 6.3**

---

#### Property 6: No corpus text in any committed output

For any 200 utterances ≥40 characters sampled from the source CSV, none appears as a substring in any cell output of `p4_kaggle_notebook.ipynb` or in any JSON under `sft/reports/`.

**Validates: Requirements 7.1**

---

### 10. Testing Strategy

#### 10.1 Unit tests (CPU venv, `sft/tests/test_p4_check.py`)

All tests use synthetic fixtures constructed in-memory or in `tempfile.TemporaryDirectory()`. No real data files are read. Tests are `unittest.TestCase` subclasses, importable with no side effects. The full test suite runs in `sft/.venv` without torch.

#### 10.2 Notebook in-notebook tests (section 2 of notebook)

Synthetic-only assertions covering spike detection and resume continuity logic, executed before any data or subprocess is touched. These run on the Kaggle kernel and confirm the logic is consistent with the unit tests.

#### 10.3 Integration-level tests (Kaggle execution)

The `verify` section's T1–T6 on Kaggle's transformers, and T2–T5 on every item of the final mix and val sets, serve as integration tests that the frozen artefacts are consistent with the Kaggle environment.
