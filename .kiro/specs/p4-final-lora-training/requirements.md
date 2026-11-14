# Requirements: P4 — Final LoRA Training on Kaggle

## Overview

P4 executes the final production LoRA SFT run for the Medical Chat LLM using the frozen artifacts from P3. The phase splits into two parts: a local CPU preparation stage (Part A) that writes `sft/p4_check.py`, unit tests, and the Kaggle upload bundle; and a post-training verification stage (Part B) that runs after the human places Kaggle results in `sft/results/p4/`.

The frozen training script (`sft/train_lora.py`) and frozen config (`sft/config/train_config.yaml`) are **never modified**. P4 wraps them, verifies them, and records their outputs.

---

## Requirement Groups

### 1. Frozen-manifest integrity (Part A precondition)

**1.1** Before any Part A code is written, all entries in `sft/frozen_manifest_p3.json` must be re-verified by running `sft/verify_frozen.py`. If any entry fails, Part A halts and reports the mismatch.

**1.2** `sft/p4_check.py` must assert at startup that `sft/train_lora.py`, `sft/config/train_config.yaml`, and every file in `frozen_manifest_p3.json` have SHA-256 values matching the manifest. Any mismatch causes the checker to exit non-zero with a descriptive error.

**1.3** The checker must verify that `train_lora.py` implements the best-by-macro checkpoint selection (`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `SAVE_TOTAL_LIMIT=3` → at most 4 checkpoints).

---

### 2. `sft/p4_check.py` — acceptance checker (stdlib-only)

**2.1** The checker is stdlib-only (no torch, peft, accelerate, pandas, datasketch). It runs on `sft/.venv`.

**2.2** The checker reads `sft/results/p4/` and runs every acceptance criterion from §5 as a named PASS/FAIL item, printing one line per criterion.

**2.3** The checker accepts an optional `--results-dir` argument (default `sft/results/p4/`) and exits 0 only if every criterion PASSes.

**2.4** The checker implements the **spike rule** as follows: a training step is a spike if its loss is more than 0.5 above the previous step's loss, **or** if the macro val NLL at any evaluation is more than 1.5× the minimum macro val NLL seen so far in the run (across all sessions).

**2.5** The checker reads `log_history.json` (fields: `step`, `train_loss`, `lr`, `grad_norm`, `val_md_nll`, `val_hc_nll`, `eval_macro`) and applies the spike rule to both the step-level training loss and the eval-level macro NLL.

**2.6** For resume continuity the checker reads all `session_log_<n>.json` files, reconstructs the global step sequence, and flags: duplicated steps (same step number appearing in two sessions), skipped steps (gap > 1 between consecutive steps, excluding the eval-only final step), and LR restarting warmup after session 1 (LR drops below 10% of peak after the warmup period of session 1 is finished).

**2.7** The checker verifies `run_manifest.json` contains all required fields: `seed`, `model_revision`, `precision`, `lora` (with sub-keys `r`, `alpha`, `dropout`, `target_modules`), `training` (with sub-keys `learning_rate`, `micro_batch`, `gradient_accumulation`, `effective_batch`, `max_steps`, `max_len`), `data` (with sub-keys `train_jsonl`, `train_sha256`, `val_md_jsonl`, `val_md_sha256`, `val_hc_jsonl`, `val_hc_sha256`), `tokenizer_pkg` (with sub-keys `template_sha256`, `system_prompt_sha256`), `gpu`, `package_versions`, `trainable_params`, `total_params`.

**2.8** The checker verifies the following values in `run_manifest.json` match the frozen config: `model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"`; `lora.r == 16`, `lora.alpha == 32`, `lora.dropout == 0.05`; the `target_modules` list excludes `lm_head` and `embed_tokens`; `training.effective_batch == 32`; `training.max_len == 2048`; `seed == 42`.

**2.9** The checker verifies adapter safetensors headers for each retained checkpoint: `lora.r == 16`, `lora.alpha == 32`, `lora.dropout == 0.05`; LoRA weight tensors stored in F32 dtype; none of `lm_head` or `embed_tokens` appear in adapter tensor names.

**2.10** The checker verifies that the data SHA-256s in `run_manifest.json` match the SHA-256s of the corresponding files in `sft/frozen_manifest_p3.json`.

**2.11** The checker implements a `--report-json <path>` flag that writes a machine-readable JSON summary (sorted keys, LF newlines) to the given path. The JSON contains each criterion name, its status (`pass` or `fail`), and a brief detail string. No corpus text appears in this file.

---

### 3. `sft/tests/test_p4_check.py` — unit tests (stdlib-only)

**3.1** All tests use synthetic fixtures only (no real data, no file system reads of `sft/work/` or `sft/results/`).

**3.2** Tests must cover all checker functions using synthetic fixtures for: NaN/inf in training loss; NaN/inf in grad norm; final macro val NLL > min + 0.05; duplicated step number at resume boundary; skipped step number at resume boundary; LR restarting warmup after session 1; missing best checkpoint directory; SHA-256 hash mismatch (data, adapter); wrong adapter hyperparameter (r, alpha, dropout); wrong dtype in adapter safetensors header (`BF16` where `F32` is required); `lm_head` or `embed_tokens` appearing in adapter tensor names; spike in training loss (> 0.5 from previous step); spike in eval macro NLL (> 1.5× minimum seen).

**3.3** Tests are `unittest.TestCase` subclasses, importable with no side effects.

**3.4** All fixtures are constructed in-memory (temporary directories may be used but no actual model files are required).

---

### 4. Kaggle upload bundle (`sft/kaggle/p4/`)

**4.1** `sft/kaggle/p4/upload_manifest.json` lists every file needed for the Kaggle run with its SHA-256 and size in bytes. Files that are unchanged from P3 (identical hash) are noted with `"reused_from_p3": true` but their hash and size are still recorded.

**4.2** The bundle must never list the test split (`dialogues_test.jsonl`), `probes_final.jsonl`, `probes_all.jsonl`, or any raw CSV.

**4.3** `sft/kaggle/p4/README.md` describes the exact human steps to: verify local artifacts before uploading, upload to Kaggle (listing which files go into which dataset, with the `reused_from_p3` note), create/configure the notebook, run each section in order, download results, and place them in `sft/results/p4/`.

**4.4** The README lists every acceptance criterion from §5 so the human knows what to check before continuing to Part B.

---

### 5. Kaggle notebook (`sft/kaggle/p4/p4_kaggle_notebook.ipynb`)

**5.1 Top-variables cell.** The first executable cell defines: `RUN_ID` (a string set once, e.g. `"p4_run_001"`); `RESUME_FROM` (None or the path to the previous session's checkpoint output directory); `DRY_RUN` (boolean, default `True`). These are the only values the human changes between sessions.

**5.2 `env` section.** Sets `CUDA_VISIBLE_DEVICES=0` before any torch import. Asserts `torch.cuda.device_count() == 1`. Runs `kaggle_env_check.py` and `gpu_probe.py` and asserts exit 0. Compares `pip freeze` output with `freeze-kaggle.txt` line by line; any pinned package with a different version stops execution with a descriptive error. Asserts that the GPU's compute capability supports the precision required by `train_config.yaml` (bf16 requires cc ≥ 8.0, fp16 requires cc ≥ 7.0); if not, stops. If the GPU has less VRAM than what E0 measured, stops; if the GPU model differs but VRAM is equal or greater, records the difference and continues. Writes `env.json`.

**5.3 `verify` section.** Runs `verify_uploads.py` against the P4 upload manifest; any failure stops execution. Loads the tokenizer from `tokenizer_pkg` and asserts: `eos_token_id == 151645`; `pad_token_id == 151643`; `pad_token_id != eos_token_id`; template SHA-256 matches the frozen value; system-prompt SHA-256 matches the frozen value. Runs T1–T6 (imported from `tokenize_mask.py`) on Kaggle's installed transformers version; any failure stops execution. Runs T2–T5 (item-level assertions) on every item of `final_mix.jsonl` and both val sets; any failure stops execution. Writes `verify.json` with pass/fail counts (no corpus text).

**5.4 `dry_run` section.** Runs only when `DRY_RUN=True`. Calls the frozen entry point (`train_lora.py`) with `max_steps=10` and a separate output directory (`/kaggle/working/dry_run_out/`); the frozen `train_config.yaml` is never modified — overrides are passed as CLI arguments. Verifies: training loss at step 10 is finite; grad norm is finite; a checkpoint directory is written; resuming from that checkpoint for 5 more steps produces finite loss. Writes `dry_run.json` with the step-10 loss, grad norm, and resume step-15 loss. Stops execution after printing "dry run done — set DRY_RUN=False for the real run."

**5.5 `train` section.** Runs only when `DRY_RUN=False`. Calls the frozen entry point with the frozen `train_config.yaml` unmodified. If `RESUME_FROM` is not None, passes `--resume RESUME_FROM`; before training, verifies the `run_id` matches the current `RUN_ID`, the adapter hash of the latest checkpoint matches the value in the most recent `session_log_<n>.json`, and the `run_manifest` input hashes (if present) equal the frozen values. Flushes `log_history.json` to `results/` after every eval callback. Logs elapsed wall-clock time and estimated time to completion after every eval step. Writes an in-session `session_log_<n>.json` (session number, GPU model, first step, last step, wall-clock seconds, adapter hashes of retained checkpoints).

**5.6 `post_train` section.** If the final training step is not an eval step, runs a final evaluation on both val sets and records the result. Verifies `run_manifest.json` exists and contains all required fields (§2.7). Writes `session_log_<n>.json` (session number from `RESUME_FROM` count + 1; GPU model; first step; last step; wall-clock seconds; SHA-256 of each retained checkpoint adapter). Writes `log_history.json` accumulating all sessions (fields per step: `step`, `train_loss`, `lr`, `grad_norm`; fields per eval step: `val_md_nll`, `val_hc_nll`, `eval_macro`). Copies any earlier session logs from `RESUME_FROM` forward so all session logs travel with the results. Packs adapter-only files for each retained checkpoint plus the full optimizer/scheduler/RNG state of the last checkpoint. Writes `files_sha256.json` (path → SHA-256 for every file in the output directory, sorted keys). Reports total output directory size in bytes.

**5.7 `smoke_test` section.** Informational only — no gate. Loads the best checkpoint by logged macro val NLL. Runs 10 seeded dev probes (from `probes_dev.jsonl`, **not** `probes_final.jsonl`) with greedy decoding, `max_new_tokens=256`, using the frozen runner. Counts: responses terminated by `<|im_end|>` (expected 10), `<think>` token occurrences (expected 0), empty replies (expected 0). Writes `smoke_test.json`. Never prints probe text or response text.

---

### 6. Part B — post-transfer verification (local CPU)

**6.1 B1 — transfer integrity.** Verifies `sft/results/p4/files_sha256.json` against all files in `sft/results/p4/`. Any mismatch fails B1. After passing, marks the directory read-only (chmod on POSIX; attrib on Windows — document the command for the human if permissions cannot be set programmatically).

**6.2 B2 — acceptance check.** Runs `sft/p4_check.py --results-dir sft/results/p4/` and reports each criterion as PASS or FAIL. If any criterion FAILs, the Part B report notes it and does not proceed to B3.

**6.3 B3 — checkpoint selection.** Selects the checkpoint for P5 as the one with the lowest macro val NLL across all eval steps in `log_history.json`, with ties broken by later step number. The final-step evaluation counts. Records the selection in `sft/results/p4/checkpoint_for_p5.json` with fields: `checkpoint_id` (the checkpoint directory name), `step`, `eval_macro`, `val_md_nll`, `val_hc_nll`, `adapter_sha256` (SHA-256 of `adapter_model.safetensors`), `selection_rule`.

**6.4 B4 — frozen manifest.** Writes `sft/frozen_manifest_p4.json` with SHA-256 and size for: all adapter files for retained checkpoints; `run_manifest.json`; `log_history.json`; all `session_log_<n>.json` files; `files_sha256.json`; `checkpoint_for_p5.json`; `sft/kaggle/p4/p4_kaggle_notebook.ipynb`; `sft/p4_check.py`; `sft/tests/test_p4_check.py`. Prints git-add commands for non-private files only (adapter weights are git-ignored; `sft/results/p4/` is git-ignored).

---

### 7. Privacy and data constraints

**7.1** No corpus text, probe text, or model response text appears in any cell output, any JSON in `sft/reports/`, any committed file, or the chat report. Reports contain counts, ids, hashes, step numbers, loss values, and enum strings only.

**7.2** The test split (`dialogues_test.jsonl`), `probes_final.jsonl`, `probes_all.jsonl`, and raw CSVs are never uploaded to Kaggle and never read in P4 code.

**7.3** Adapter weights and `sft/results/p4/` are git-ignored. The agent must confirm with `git check-ignore -v <path>` before writing `frozen_manifest_p4.json`.

---

### 8. Immutability constraints

**8.1** `sft/train_lora.py` must not be modified.

**8.2** `sft/config/train_config.yaml` must not be modified.

**8.3** Every file in `sft/frozen_manifest_p3.json` must not be modified.

**8.4** If any frozen file appears incorrect, execution halts and reports the discrepancy without making changes.

---

### 9. Verification checks (must all pass before P4 is marked complete)

**9.1** `sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests` — all tests pass, including `test_p4_check.py`.

**9.2** `sft\.venv\Scripts\python.exe -m compileall -q sft` — exits 0.

**9.3** `sft\.venv\Scripts\python.exe sft\verify_manifest.py` — exits 0.

**9.4** `git status --short` shows only new files under `sft/` (no modifications to frozen files).

**9.5** Privacy scan: 200 random utterances ≥40 characters from the source CSV must not appear as substrings in any new notebook's JSON or any `sft/reports/*.json`.

**9.6** Each new notebook executes top-to-bottom without error cells.
