# Implementation Plan: P4 — Final LoRA Training on Kaggle

## Overview

P4 produces three code artefacts locally (the acceptance checker, its unit tests, and the Kaggle upload bundle), then hands off to a human for the Kaggle run, and finishes with a local Part B verification pass. The frozen training script and config are never touched. All new Python files are stdlib-only on the CPU side.

---

## Tasks

- [ ] 1. Write `sft/p4_check.py` — the stdlib-only acceptance checker
  - [ ] 1.1 Implement the safetensors-header reader and the SHA-256 helper
    - Write `_read_safetensors_metadata(path: Path) -> dict` that reads only the 8-byte length prefix + JSON header without loading tensors.
    - Write `sha256_file(path: Path) -> str` (identical to the one in `verify_manifest.py`; re-implement, do not import across modules).
    - Add type hints on both functions.
    - _Requirements: 2.1, 2.9_

  - [ ] 1.2 Implement the pure logic functions for each criterion (C1–C16)
    - Write `_check_no_nan_inf(log: list[dict]) -> tuple[str, str]` for C5.
    - Write `_check_no_spike_loss(log: list[dict]) -> tuple[str, str]` for C6 (threshold 0.5).
    - Write `_check_no_spike_macro(log: list[dict]) -> tuple[str, str]` for C7 (threshold 1.5×).
    - Write `_check_val_nll_bounded(log: list[dict]) -> tuple[str, str]` for C4.
    - Write `_check_resume_step_continuity(session_logs: list[dict]) -> tuple[str, str]` for C8.
    - Write `_check_lr_no_restart(log: list[dict], warmup_end_step: int, peak_lr: float) -> tuple[str, str]` for C9.
    - Write `_check_manifest_fields(manifest: dict) -> tuple[str, str]` for C10.
    - Write `_check_manifest_values(manifest: dict) -> tuple[str, str]` for C11.
    - Write `_check_data_hashes(manifest: dict, frozen_entries: dict) -> tuple[str, str]` for C12.
    - Write `_check_adapter_hyperparams(adapter_config: dict) -> tuple[str, str]` for C13.
    - Write `_check_adapter_dtype(safetensors_meta: dict) -> tuple[str, str]` for C14.
    - Write `_check_script_implements_best_macro(script_text: str) -> tuple[str, str]` for C2 (string search in source).
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 1.3_

  - [ ] 1.3 Implement the IO wrappers and the checkpoint-selection function
    - Write IO wrappers (`check_<name>(results_dir, ...)`) that load files and delegate to logic functions.
    - Write `select_checkpoint_for_p5(log_history: list[dict], results_dir: Path) -> dict` that returns the `checkpoint_for_p5` record (lowest `eval_macro`, ties by later step).
    - _Requirements: 2.2, 6.3_

  - [ ] 1.4 Implement the CLI (`main`) with argparse and `--report-json`
    - `--results-dir` (default `sft/results/p4`), `--frozen-manifest` (default `sft/frozen_manifest_p3.json`), `--report-json`.
    - Print one line per criterion (ID, name, PASS/FAIL, detail).
    - Write the JSON report with `sort_keys=True`, `ensure_ascii=False`, `\n` newlines.
    - Exit 0 if all pass, exit 1 if any fail, exit 2 on input error.
    - _Requirements: 2.3, 2.11_

- [ ] 2. Write `sft/tests/test_p4_check.py` — unit tests with synthetic fixtures
  - [ ] 2.1 Write fixtures and test cases for C5 (NaN/inf detection)
    - Synthetic log with `train_loss = float("nan")` → C5 fails.
    - Synthetic log with `grad_norm = float("inf")` → C5 fails.
    - Synthetic log with `eval_macro = float("nan")` → C5 fails.
    - Clean log with finite values → C5 passes.
    - _Requirements: 3.1, 3.2_

  - [ ] 2.2 Write test cases for C6/C7 (spike detection)
    - Loss deltas all ≤ 0.5 → C6 passes; one delta = 0.51 → C6 fails.
    - Macro NLLs all < 1.5× running min → C7 passes; one = 1.51× → C7 fails.
    - _Requirements: 3.2_

  - [ ] 2.3 Write test cases for C4 (val NLL bounded)
    - Final macro = min + 0.04 → pass; final macro = min + 0.06 → fail.
    - Single-eval log (final == min) → pass.
    - _Requirements: 3.2_

  - [ ] 2.4 Write test cases for C8/C9 (resume continuity and LR restart)
    - Three session logs with clean handoff → C8 passes.
    - Session 2 starts at same step as session 1 ends → C8 fails.
    - Session 2 starts 2 steps after session 1 ends → C8 fails.
    - LR monotone after warmup end → C9 passes; LR rises 10% of peak → C9 fails.
    - _Requirements: 3.2_

  - [ ] 2.5 Write test cases for C10–C12 (manifest fields, values, data hashes)
    - `run_manifest` missing `seed` key → C10 fails.
    - `model_revision` wrong → C11 fails; `lora.r = 32` → C11 fails.
    - `train_sha256` mismatch → C12 fails.
    - _Requirements: 3.2_

  - [ ] 2.6 Write test cases for C13/C14 (adapter config and dtype)
    - `adapter_config` with `r=32` → C13 fails.
    - `lm_head` in `target_modules` → C13 fails.
    - Safetensors header with a lora weight tensor dtype `BF16` → C14 fails.
    - All lora weight tensors dtype `F32` → C14 passes.
    - _Requirements: 3.2, 3.4_

- [ ] 3. Checkpoint: ensure unit tests pass before building the notebook
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Build `sft/kaggle/p4/upload_manifest.json`
  - [ ] 4.1 Compute SHA-256 and size for every upload file; mark P3 reuses
    - Enumerate all files from §3.1 of the design (code, tokenizer, config, training data, dev probes, model weights).
    - Compare each hash against the P3 `sft/kaggle/upload_manifest.json`; set `"reused_from_p3": true` where hashes match.
    - Write the manifest with `sort_keys=True`, `ensure_ascii=False`, `\n` newlines.
    - Assert the test split, `probes_final.jsonl`, `probes_all.jsonl`, and raw CSVs are not listed.
    - _Requirements: 4.1, 4.2_

- [ ] 5. Write `sft/kaggle/p4/p4_kaggle_notebook.ipynb`
  - [ ] 5.1 Write sections 0 (parameters) and 1 (definitions)
    - Section 0: `RUN_ID`, `RESUME_FROM`, `DRY_RUN=True`, all Kaggle path constants, `FROZEN_LR`, `FROZEN_MICRO_BATCH`, `FROZEN_TEMPLATE_SHA256`, `FROZEN_SYSTEM_PROMPT_SHA256` (read from `tokenizer_pkg_manifest.json` at notebook-build time).
    - Section 1: all imports and helper functions (`sha256_file`, `_write_json`, `_compare_pip_freeze`, `_read_safetensors_metadata`, `_hash_adapter`, `_pick_best_checkpoint`).
    - No data processing in either section.
    - _Requirements: 5.1_

  - [ ] 5.2 Write section 2 (in-notebook tests) using synthetic fixtures only
    - JSON round-trip for session log format.
    - SHA-256 on a known in-memory byte string.
    - Spike detection (3 synthetic log vectors: clean, loss spike, macro spike).
    - Resume continuity (clean and gap cases).
    - All wrapped in `assert`; any failure halts notebook.
    - _Requirements: 5.1, 5.2 (notebook structure requirement)_

  - [ ] 5.3 Write section 3 (env) — environment and package checks
    - Set `CUDA_VISIBLE_DEVICES=0` before any torch import.
    - Assert `torch.cuda.device_count() == 1`.
    - Run `kaggle_env_check.py` and `gpu_probe.py` as subprocesses; assert exit 0.
    - Implement `_compare_pip_freeze`; halt on any pinned-package version diff.
    - Assert GPU compute capability supports required precision.
    - Write `results/env.json`.
    - _Requirements: 5.2_

  - [ ] 5.4 Write section 4 (verify) — upload and tokenizer integrity
    - Run `verify_uploads.py`; assert exit 0.
    - Load tokenizer; assert EOS/PAD ids and template/system-prompt hashes.
    - Run T1–T6 (import from `tokenize_mask`) on synthetic inputs; halt on any failure.
    - Run T2–T5 on every item of `final_mix.jsonl` and both val sets; halt on any failure.
    - Write `results/verify.json` with pass/fail counts (no corpus text).
    - _Requirements: 5.3_

  - [ ] 5.5 Write section 5 (dry_run) — 10-step proof of life
    - Guard with `if DRY_RUN:`.
    - Call `train_lora.py` with `--n-steps 10`, separate output dir; pass frozen LR and micro_batch as CLI args (the frozen config file is never modified).
    - Verify finite loss and grad norm at step 10; verify checkpoint written; verify resume for 5 more steps produces finite loss.
    - Write `results/dry_run.json`.
    - Print "dry run done — set DRY_RUN=False for the real run" and `raise SystemExit(0)`.
    - _Requirements: 5.4_

  - [ ] 5.6 Write section 6 (train) — full run with resume support
    - Guard with `if not DRY_RUN:`.
    - Build CLI command for `train_lora.py` with `--config sft/config/train_config.yaml` (unmodified); if `RESUME_FROM` is set, verify adapter hash + run_id against previous session log before adding `--resume`.
    - After each eval callback (or via log polling), flush accumulated `log_history.json` to `results/`.
    - Log elapsed and ETA after every eval step.
    - _Requirements: 5.5_

  - [ ] 5.7 Write section 7 (post_train) — log accumulation, packing, hashing
    - If the final step is not an eval step, run a final evaluation on both val sets.
    - Verify `run_manifest.json` exists with all required fields.
    - Build and write `results/session_log_<n>.json`.
    - Accumulate `results/log_history.json` from all sessions.
    - Pack adapter-only files for all retained checkpoints + full state for the last.
    - Compute and write `results/files_sha256.json`.
    - Report total output size.
    - _Requirements: 5.6_

  - [ ] 5.8 Write section 8 (smoke_test) — informational only
    - Load best checkpoint by logged macro NLL; run 10 seeded dev probes (from `probes_dev.jsonl`).
    - Count `<|im_end|>` terminations, `<think>` occurrences, empty replies.
    - Write `results/smoke_test.json` (counts only — no probe or response text).
    - _Requirements: 5.7_

  - [ ] 5.9 Validate notebook format and cell ids
    - Confirm `nbformat.validate` passes.
    - Confirm all cell ids are unique strings following the `p4_XXXX` scheme.
    - Confirm the notebook has a final status cell as the last cell.
    - _Requirements: 5.1 (notebook structure)_

- [ ] 6. Write `sft/kaggle/p4/README.md`
  - Describe all prerequisites (local artefacts to verify before uploading).
  - List exact upload steps (which files go into which Kaggle dataset, noting P3 reuses).
  - List notebook creation and configuration steps (GPU, internet off, dataset attachments).
  - Describe what to do after the run (download `results/`, place in `sft/results/p4/`).
  - List all 16 acceptance criteria so the human knows what Part B will check.
  - _Requirements: 4.3, 4.4_

- [ ] 7. Checkpoint: run all verification checks (Part A complete)
  - Ensure all tests pass, ask the user if questions arise.
  - Run: `sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests`
  - Run: `sft\.venv\Scripts\python.exe -m compileall -q sft`
  - Run: `sft\.venv\Scripts\python.exe sft\verify_manifest.py`
  - Check `git status --short` for unexpected modifications.
  - Write Part A section of `sft/P4_REPORT.md`.

- [ ] 8. Part B — post-transfer verification (after human places `sft/results/p4/`)
  - [ ] 8.1 B1: verify transfer integrity
    - Run the `files_sha256.json` check against all files in `sft/results/p4/`.
    - If any mismatch, report and halt.
    - Document the read-only command for the human (attrib on Windows).
    - _Requirements: 6.1_

  - [ ] 8.2 B2: run acceptance checker
    - Run `sft\p4_check.py --results-dir sft/results/p4 --report-json sft/results/p4/check_report.json`.
    - Report each of the 16 criteria as PASS or FAIL.
    - If any FAIL, record it in `sft/P4_REPORT.md` and do not proceed to B3.
    - _Requirements: 6.2_

  - [ ] 8.3 B3: select checkpoint for P5
    - Run checkpoint selection from `log_history.json` (lowest macro, ties by later step).
    - Write `sft/results/p4/checkpoint_for_p5.json` with all required fields.
    - _Requirements: 6.3_

  - [ ] 8.4 B4: write `sft/frozen_manifest_p4.json` and `sft/P4_REPORT.md`
    - Run `git check-ignore -v` on adapter weight paths before including them in the git-add list.
    - Write `sft/frozen_manifest_p4.json` with SHA-256 and size for all non-private new files.
    - Print git-add commands covering only non-private files.
    - Write the final `sft/P4_REPORT.md` covering both Part A and Part B sections.
    - _Requirements: 6.4_

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP.
- Part B tasks (8.1–8.4) depend on the human placing results in `sft/results/p4/`; they cannot run until `HUMAN ACTIONS PENDING` is resolved.
- The frozen training script and config are never modified; the notebook passes all required parameters as CLI arguments.
- `p4_check.py` and `test_p4_check.py` are stdlib-only — no torch, peft, accelerate, or any package outside the standard library.
- Spike rule: training loss rise > 0.5 from previous step, **or** eval macro > 1.5× minimum seen so far.
- All reports contain only counts, ids, hashes, step numbers, loss values, and enum strings — no corpus text.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["1.4", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6"] },
    { "id": 4, "tasks": ["4.1", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "5.4"] },
    { "id": 6, "tasks": ["5.5", "5.6"] },
    { "id": 7, "tasks": ["5.7", "5.8"] },
    { "id": 8, "tasks": ["5.9"] },
    { "id": 9, "tasks": ["8.1"] },
    { "id": 10, "tasks": ["8.2"] },
    { "id": 11, "tasks": ["8.3"] },
    { "id": 12, "tasks": ["8.4"] }
  ]
}
```
