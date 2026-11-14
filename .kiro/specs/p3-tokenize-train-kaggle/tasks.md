# P3 — Tasks

## Task List

### STEP 0 — Preconditions

- [ ] 0.1 Run `sft/verify_manifest.py` and confirm exit 0
- [ ] 0.2 Run `sft/verify_frozen.py` and confirm exit 0
- [ ] 0.3 Verify `eval/gates.yaml` is tracked and unmodified in git (`git ls-files --error-unmatch eval/gates.yaml` and `git status --porcelain eval/gates.yaml` returns empty)
- [ ] 0.4 Verify `sft/results/p2/` exists

### Part A — Local CPU

#### A1 — Tokenizer package

- [ ] 1.1 Inspect `sft/build_tokenizer_pkg.py` for correctness: Jinja2 template syntax, transformers 5.x `tokenizer_config.json` key names, file copy logic, manifest output format
- [ ] 1.2 Run `build_tokenizer_pkg.py` to build `sft/tokenizer_pkg/`
- [ ] 1.3 Verify `sft/tokenizer_pkg/tokenizer_pkg_manifest.json` contains `template_sha256`, `system_prompt_sha256`, and per-file SHA-256 entries
- [ ] 1.4 Verify `generation_config.json` has `eos_token_id=[151645, 151643]` and `pad_token_id=151643`
- [ ] 1.5 Load tokenizer and assert `pad_token_id=151643`, `eos_token_id=151645`, `pad != eos`

#### A2 — tokenize_mask.py audit

- [ ] 2.1 Confirm `add_special_tokens=False` is the correct flag name in transformers 5.20.2 (check `inspect.signature(tok.encode)` or transformers changelog)
- [ ] 2.2 Confirm the `_SPECIAL_TOK_RE` regex covers all Qwen3 special tokens and correctly identifies literal special-token strings in content
- [ ] 2.3 Confirm that `_build_exchange_tokens` produces labels of `-100` for the assistant role header and the trailing `\n`, and of token IDs for the assistant content and closing `<|im_end|>`
- [ ] 2.4 Confirm that `tokenize_dialogue` truncation logic drops trailing exchanges at turn boundaries and returns `dropped_too_long` when exchange 1 alone does not fit
- [ ] 2.5 Confirm `training_prefix_string` and `apply_chat_template_to_prefix` produce byte-for-byte equal output for T6

#### A3 — Tests T1–T6 (CPU)

- [ ] 3.1 Run `HF_HUB_OFFLINE=1 sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests` and confirm all tests pass (record pass/skip/fail counts)
- [ ] 3.2 If any test fails, identify and fix the bug in the production code (never in the test); re-run until all pass
- [ ] 3.3 Confirm T7 is present as a `skipUnless(torch_available)` test with the full implementation documented as residing in `sft/kaggle/p3_kaggle_notebook.ipynb`

#### A4 — Build tokenized mixes

- [ ] 4.1 Confirm `sft/work/dialogues_train.jsonl` and `sft/work/dialogues_val.jsonl` exist (produced by P1/P2); if missing, record as a precondition failure and stop
- [ ] 4.2 Confirm `sft/work/behavior_dialogues.jsonl` exists; if missing, run with a warning and report 0% behavior share
- [ ] 4.3 Run `sft/build_mix.py --pilot` and confirm exit 0
- [ ] 4.4 Verify the four output files exist and have non-zero size: `pilot_mix.jsonl`, `overfit_set.jsonl`, `val_meddialog_tok.jsonl`, `val_healthchat_tok.jsonl`
- [ ] 4.5 Read `build_mix_report.json` and confirm pilot has 4,800 dialogues; overfit has 16; val subsets have 200 each; realised token shares are within 5 percentage points of targets (50/45/5)
- [ ] 4.6 Confirm `max_repeats_per_dialogue` is recorded (expected ≤3 from BEH_UPSAMPLE=3)

#### A3 extended — T2–T5 on produced sets

- [ ] 5.1 Run T2–T5 checks (as in the P3 notebook section 7) on `pilot_mix.jsonl`, `overfit_set.jsonl`, `val_meddialog_tok.jsonl`, `val_healthchat_tok.jsonl`; confirm 0 failures on all sets
- [ ] 5.2 Run T1 on 1,000 seeded-random items from `dialogues_train.jsonl`; confirm 0 mismatches
- [ ] 5.3 Run T6 on ≥500 user-ending prefixes from `dialogues_train.jsonl`; confirm 0 failures

#### A5 — Training script audit

- [ ] 6.1 Check `TrainingArguments` argument names against transformers 5.20.2: `eval_strategy` (not `evaluation_strategy`), `save_safetensors`, `load_best_model_at_end` compatibility with `save_strategy='steps'`
- [ ] 6.2 Replace `pkg_resources.get_distribution` with `importlib.metadata.version` if running Python 3.12+ (or add a try/except fallback that is already present)
- [ ] 6.3 Confirm `Trainer.evaluate()` override signature is compatible with transformers 5.20.2
- [ ] 6.4 Confirm `LoraConfig` from peft uses valid argument names (verify against `peft` version in `requirements-kaggle.txt`)
- [ ] 6.5 Confirm `LORA_TARGET_PATTERN` constant is used or remove it (actual target modules are computed dynamically from `named_modules`)
- [ ] 6.6 Confirm `run_manifest.json` is written with sorted keys and LF newlines

#### A6 — Kaggle upload package audit

- [ ] 7.1 Verify `sft/kaggle/verify_uploads.py` is stdlib-only, has `--help`, and exits 1 on hash mismatch
- [ ] 7.2 Verify `sft/kaggle/p3_kaggle_notebook.ipynb` passes `nbformat.validate()` (all cell IDs unique, valid metadata)
- [ ] 7.3 Confirm each notebook section writes exactly one JSON to `/kaggle/working/results/` with no corpus text
- [ ] 7.4 Confirm the E0 section builds max-length synthetic batches (worst case) for VRAM measurement and measures tok/s separately on realistic batches (non-pad token count)
- [ ] 7.5 Confirm the kill-and-resume section trains two independent runs and compares step K+1 loss (not step K)
- [ ] 7.6 Confirm the T7 section uses fp32, LoRA dropout=0, the same 4 dialogues, and divides each micro-batch loss by 4 before backward (to match gradient scale)
- [ ] 7.7 Confirm the E2 section evaluates on both val sets at every eval checkpoint and records the macro (mean of two NLLs)

#### Local CPU notebook execution

- [ ] 8.1 Ensure `sft/work/` and `sft/reports/` directories exist (create if missing)
- [ ] 8.2 Execute `sft/notebooks/P3_tokenize_mix.ipynb` with nbconvert and confirm no error cells
- [ ] 8.3 Confirm `sft/reports/p3_tokenize_report.json` is written and contains only counts, hashes, and enum strings (no corpus text)
- [ ] 8.4 Confirm `sft/kaggle/upload_manifest.json` is written with SHA-256 for all upload files

#### Privacy scan

- [ ] 9.1 Take 200 random utterances (≥40 chars) from `data/processed/integrated_conversations_ner_validated.csv` using a temp script under `sft/work/`
- [ ] 9.2 Assert none appears as a substring in `sft/notebooks/P3_tokenize_mix.ipynb` or any `sft/reports/*.json`
- [ ] 9.3 Delete the temp script after the scan passes

#### Verification

- [ ] 10.1 Run `sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests` — all pass
- [ ] 10.2 Run `sft\.venv\Scripts\python.exe -m compileall -q sft` — no syntax errors
- [ ] 10.3 Run `sft\.venv\Scripts\python.exe sft\verify_manifest.py` — exit 0
- [ ] 10.4 Run `sft\.venv\Scripts\python.exe sft\verify_frozen.py` — exit 0
- [ ] 10.5 Run `git status --short` and confirm only new files under `sft/` (no modifications to protected files)

#### Part A checkpoint report

- [ ] 11.1 Write `sft/P3_REPORT.md` with sections: files created, files modified, implementation summary, commands run with exit codes and key output lines, test results, measured results (counts/hashes/token shares/truncation stats), deviations, unresolved issues, **HUMAN ACTIONS PENDING**
- [ ] 11.2 The HUMAN ACTIONS PENDING section must list: exact upload steps, Kaggle notebook execution steps, where to place result JSONs, and that Part B starts only after all JSONs are in `sft/results/p3/`
- [ ] 11.3 Confirm the report contains no dataset text (counts, hashes, and enum strings only)

---

### Part B — After Kaggle Results (run only after sft/results/p3/ is populated)

#### B1 — Accept or reject Kaggle results

- [ ] 12.1 Read all JSONs from `sft/results/p3/` and report each acceptance criterion as PASS or FAIL
- [ ] 12.2 If E0 shows NaN/inf at all micro-batch sizes: apply remedy (halve LR to 5e-5 for E0 rerun) and document; if NaN persists on the rerun, stop and report (needs bf16 GPU)
- [ ] 12.3 If T7 fails: check the loss-division factor in the micro-batch-1 loop and re-examine the token-mean loss implementation
- [ ] 12.4 If overfit fails: run T2/T3 diagnostics on the overfit set before adjusting LR
- [ ] 12.5 If resume fails: check that the saved checkpoint includes the optimizer, scheduler, scaler, and RNG state files

#### B2 — LR selection

- [ ] 13.1 Define "spike" before reading curves: training loss increase > 0.5 from previous step, or val macro loss > 1.5× the minimum seen so far; write this definition to the report
- [ ] 13.2 Apply the spike definition to all three E2 LR runs and identify any spiking runs
- [ ] 13.3 Select the LR with the lowest final-step macro val loss among non-spiking runs; record it

#### B3 — Budget computation and final mix

- [ ] 14.1 Compute N_dialogues for ≤5 GPU-hours using the formula: `N = floor(5 * 3600 * toks_per_sec / (avg_seq_len * 1.15 + eval_overhead))` and report the derivation
- [ ] 14.2 Build the final mix: `sft/build_mix.py --n-dialogues <N>`
- [ ] 14.3 Re-run T2–T5 on `sft/work/final_mix.jsonl` and confirm 0 failures
- [ ] 14.4 Compute and record the SHA-256 of `final_mix.jsonl`
- [ ] 14.5 Extend `sft/kaggle/upload_manifest.json` to include `final_mix.jsonl`

#### B4 — Freeze

- [ ] 15.1 Create `sft/config/` directory if missing
- [ ] 15.2 Write `sft/config/train_config.yaml` with keys: `micro_batch`, `gradient_accumulation`, `learning_rate`, `max_len`, `n_dialogues`, `precision`, `seed`
- [ ] 15.3 Copy `pip_freeze.txt` from `sft/results/p3/` to `sft/freeze-kaggle.txt`
- [ ] 15.4 Write `sft/frozen_manifest_p3.json` with SHA-256 entries for: `sft/tokenizer_pkg/tokenizer_pkg_manifest.json`, `sft/work/final_mix.jsonl`, `sft/work/val_meddialog_tok.jsonl`, `sft/work/val_healthchat_tok.jsonl`, `sft/config/train_config.yaml`
- [ ] 15.5 Confirm `sft/frozen_manifest.json` (the original) is unmodified

#### B5 — Final verification and report

- [ ] 16.1 Run `sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests` — all pass
- [ ] 16.2 Run `sft\.venv\Scripts\python.exe -m compileall -q sft` — no syntax errors
- [ ] 16.3 Run `sft\.venv\Scripts\python.exe sft\verify_manifest.py` — exit 0
- [ ] 16.4 Run `sft\.venv\Scripts\python.exe sft\verify_frozen.py` — exit 0
- [ ] 16.5 Run `git status --short` and confirm only new files under `sft/` are shown
- [ ] 16.6 Update `sft/P3_REPORT.md` with Part B results, filling in all PASS/FAIL verdicts and measured values
- [ ] 16.7 Update `.kiro/specs/p3-tokenize-train-kaggle/tasks.md` to mark all completed tasks
