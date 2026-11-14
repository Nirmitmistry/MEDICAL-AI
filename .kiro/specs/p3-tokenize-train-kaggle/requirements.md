# Requirements Document

## Introduction

P3 covers the tokenize-and-mask stage, the LoRA SFT training script, and the Kaggle E0/E2 pilot experiments for the Medical Chat LLM capstone project. The system must (a) produce a custom-ChatML tokenizer package from the pinned Qwen3-1.7B tokenizer snapshot, (b) tokenize and label-mask the dialogue JSONL splits into training mixes and evaluation sets, (c) provide a Kaggle-ready LoRA SFT training script, and (d) produce a Kaggle notebook that runs the E0 GPU-memory check, the overfit sanity test, the kill-and-resume test, T7 gradient-accumulation equivalence, and the E2 three-way LR pilot. All GPU work runs exclusively on Kaggle. No dataset text may appear in any output, report, or notebook cell. All code that does not require a GPU must run on the CPU venv (`sft/.venv`) without importing torch, peft, or accelerate.

The following modules are already delivered from earlier phases and must not be modified or deleted:
- `sft/tokenize_mask.py` — tokenizer and masking logic
- `sft/train_lora.py` — LoRA training script
- `sft/build_tokenizer_pkg.py` — tokenizer package builder
- `sft/build_mix.py` — training mix builder
- `sft/tests/test_tokenize_mask.py` — T1–T7 test suite
- `sft/notebooks/P3_tokenize_mix.ipynb` — local CPU notebook
- `sft/kaggle/p3_kaggle_notebook.ipynb` — Kaggle GPU notebook
- All files listed in `sft/protected_manifest.json` and any `sft/frozen_manifest*.json`

P3 tasks complete the deliverables described in ARCHITECTURE.md G(P3) and IMPLEMENTATION_PLAN Stage 3a–3c.

---

## Glossary

- **TokenizedItem**: A Python dataclass holding `input_ids`, `labels`, `n_tokens`, `truncated`, `source`, and `dialogue_id` for one dialogue after ChatML serialization and label masking.
- **TokenizationStats**: A mutable counter tracking `kept`, `truncated`, `dropped_too_long`, and `dropped_special_token_in_content` per data source.
- **tokenizer_pkg**: The directory `sft/tokenizer_pkg/` containing the custom ChatML tokenizer derived from `sft/tokenizer_snapshot/` with our own `chat_template`, `generation_config`, and a `tokenizer_pkg_manifest.json` with file hashes.
- **tokenizer_snapshot**: The read-only Qwen3-1.7B tokenizer files in `sft/tokenizer_snapshot/`; ground rule 1 forbids any modification.
- **pilot_mix**: A 4,800-dialogue tokenized JSONL file in `sft/work/` used for E2 pilots (150 steps × 32 effective batch).
- **overfit_set**: A 16-dialogue tokenized JSONL file (8 MedDialog + 8 HealthChat) used for the overfit sanity test.
- **val_meddialog_tok / val_healthchat_tok**: 200-item tokenized JSONL subsets of the validation split, one per source, used for in-training evaluation.
- **upload_manifest**: `sft/kaggle/upload_manifest.json`; lists the SHA-256 and size of every file that must be uploaded to Kaggle before training.
- **SYSTEM_PROMPT_V1**: The ≈90-token system prompt defined in `sft/prompts/system_prompt_v1.txt` and replicated as a constant in `sft/tokenize_mask.py`; its SHA-256 is pinned.
- **MASKED**: The integer −100; the label value for all non-supervised token positions.
- **IM_END_ID**: Token id 151,645 corresponding to `<|im_end|>`; the EOS token and the closing supervised token for every assistant turn.
- **ENDOFTEXT_ID**: Token id 151,643 corresponding to `<|endoftext|>`; used as the pad token (must differ from EOS).
- **ChatML template**: The custom Jinja2 template stored in `tokenizer_config.json`; it injects SYSTEM_PROMPT_V1 when no system message is present and never emits `<think>` blocks.
- **macro val loss**: The arithmetic mean of the per-step evaluation losses on `val_meddialog_tok` and `val_healthchat_tok`; the metric used to select the best checkpoint and compare E2 LR pilots.
- **E0**: The GPU-memory and throughput check: 50 training steps at micro-batch 1, 2, and 4; selects the largest micro-batch with peak VRAM < 90% and no NaN/inf.
- **E2**: Three 150-step pilot runs at learning rates 5e-5, 1e-4, and 2e-4 on the identical pilot mix; selects the LR with the lowest macro val loss and no loss spikes.
- **Tokenizer_Package_Builder**: The module `sft/build_tokenizer_pkg.py`.
- **Mix_Builder**: The module `sft/build_mix.py`.
- **Tokenize_Mask_Module**: The module `sft/tokenize_mask.py`.
- **Training_Script**: The module `sft/train_lora.py`.
- **Local_Notebook**: `sft/notebooks/P3_tokenize_mix.ipynb`; runs on `sft/.venv-nb` (CPU).
- **Kaggle_Notebook**: `sft/kaggle/p3_kaggle_notebook.ipynb`; runs on Kaggle with a GPU.
- **verify_manifest**: `sft/verify_manifest.py`; must exit 0 after any change.
- **verify_frozen**: `sft/verify_frozen.py`; must exit 0 after any change.
- **sft_venv**: `sft/.venv`; CPU-only venv; allowed packages are those in `sft/requirements-cpu.txt` plus their transitive dependencies.
- **sft_venv_nb**: `sft/.venv-nb`; notebook venv for local notebook execution.

---

## Requirements

### Requirement 1 — Tokenizer Package

**User Story:** As a training engineer, I want a reproducible tokenizer package derived from the pinned snapshot so that training on Kaggle uses the identical template and special-token configuration as local CPU tests.

#### Acceptance Criteria

1. THE Tokenizer_Package_Builder SHALL copy every file from `sft/tokenizer_snapshot/` to `sft/tokenizer_pkg/` without modifying the source files.
2. THE Tokenizer_Package_Builder SHALL overwrite `tokenizer_config.json` in `sft/tokenizer_pkg/` with the custom ChatML template that injects SYSTEM_PROMPT_V1 as the system block when no system message is present and appends `<|im_start|>assistant\n` when `add_generation_prompt` is true.
3. THE Tokenizer_Package_Builder SHALL overwrite `generation_config.json` in `sft/tokenizer_pkg/` so that `eos_token_id` equals `[151645, 151643]` and `pad_token_id` equals `151643`.
4. THE Tokenizer_Package_Builder SHALL write `sft/tokenizer_pkg/tokenizer_pkg_manifest.json` with the SHA-256 of every file in the package, the template SHA-256, and the system-prompt SHA-256, using sorted keys and LF newlines.
5. WHEN `sft/tokenizer_pkg/` already exists, THE Tokenizer_Package_Builder SHALL overwrite it deterministically so that re-running produces byte-for-byte identical output.
6. THE Tokenizer_Package_Builder SHALL exit 0 on success and exit 1 on any failure, printing the error to stderr.
7. WHEN the tokenizer is loaded from `sft/tokenizer_pkg/`, THE Tokenize_Mask_Module SHALL assert that `eos_token_id` equals 151645, `pad_token_id` equals 151643, and `pad_token_id` differs from `eos_token_id`.
8. THE ChatML template SHALL NOT contain any `<think>` blocks.

---

### Requirement 2 — Label Masking

**User Story:** As a training engineer, I want every dialogue tokenized segment-by-segment with correct assistant-only supervision so that only assistant content and its closing `<|im_end|>` contribute to the training loss.

#### Acceptance Criteria

1. WHEN tokenizing a dialogue, THE Tokenize_Mask_Module SHALL assign label equal to the token id for every content token and the closing `<|im_end|>` of every assistant turn, and SHALL assign MASKED (−100) for all system tokens, all user tokens, all role-header tokens (`<|im_start|>{role}\n`), and all trailing `\n` tokens after each `<|im_end|>`.
2. WHEN tokenizing a dialogue, THE Tokenize_Mask_Module SHALL produce `input_ids` by concatenating segment-by-segment token lists (T1 parity requirement), and THE test T1 SHALL assert that segment-by-segment `input_ids` equal full-string tokenization on 1,000 seeded-random train items.
3. THE test T2 SHALL assert that the decoded supervised token ids equal the concatenation of `{assistant_content}<|im_end|>` for every assistant turn, in order.
4. THE test T3 SHALL assert that zero supervised labels appear in the system block, any user turn, any role header, or any padding position.
5. THE test T4 SHALL assert that every assistant turn contributes exactly one supervised `<|im_end|>` token at its close, and that `pad_token_id` differs from `eos_token_id`.
6. WHEN a dialogue is truncated to fit within `max_len`, THE Tokenize_Mask_Module SHALL retain the longest prefix ending on a complete assistant turn, and THE test T5 SHALL assert that the last supervised token of a truncated item is `<|im_end|>`.
7. THE test T6 SHALL assert that for every prefix ending in a user turn, `tok.apply_chat_template(messages, add_generation_prompt=True)` produces a string whose token ids equal the training sequence's prefix through the assistant role header, byte-for-byte.
8. WHEN content contains a literal special-token string (e.g. `<|im_start|>`), THE Tokenize_Mask_Module SHALL drop the dialogue with reason `dropped_special_token` and SHALL NOT produce a TokenizedItem for it.
9. IF a dialogue's first exchange alone exceeds `max_len` after including the system block, THEN THE Tokenize_Mask_Module SHALL drop it with reason `dropped_too_long` and SHALL NOT produce a TokenizedItem.

---

### Requirement 3 — Training Mix Construction

**User Story:** As a training engineer, I want a reproducible tokenized training mix with the specified token-share ratios and a small overfit set so that pilot experiments use a consistent, well-characterized dataset.

#### Acceptance Criteria

1. THE Mix_Builder SHALL produce `sft/work/pilot_mix.jsonl` containing 4,800 tokenized dialogues sampled with seed 42 at token-share ratios of 50% MedDialog, 45% HealthChat, and 5% behavior set (each behavior item upsampled 3×).
2. THE Mix_Builder SHALL produce `sft/work/overfit_set.jsonl` containing exactly 8 MedDialog and 8 HealthChat tokenized dialogues.
3. THE Mix_Builder SHALL produce `sft/work/val_meddialog_tok.jsonl` and `sft/work/val_healthchat_tok.jsonl` each containing exactly 200 tokenized items drawn from their respective sources in the validation split.
4. THE Mix_Builder SHALL produce `sft/work/build_mix_report.json` containing realised token shares, truncation and drop counts per source, and SHA-256 of each output file, using sorted keys and LF newlines.
5. WHEN re-run with the same inputs and seed, THE Mix_Builder SHALL produce output files with identical SHA-256 values.
6. THE Mix_Builder SHALL write tokenized JSONL items in the format `{"dialogue_id": str, "source": str, "input_ids": [...], "labels": [...], "n_tokens": int, "truncated": bool}` with `ensure_ascii=False` and LF newlines.
7. THE Mix_Builder SHALL NOT include any dataset utterance text in `build_mix_report.json` or any other report; only counts, hashes, ids, and enum strings are permitted.
8. WHILE `max_len` equals 2,048, THE Mix_Builder SHALL assert that at least 90% of eligible multi-exchange train dialogues are untruncated (Architecture E3 requirement).

---

### Requirement 4 — LoRA Training Script

**User Story:** As a training engineer, I want a Kaggle-only LoRA SFT training script that enforces GPU execution, selects precision automatically, and writes a reproducible run manifest so that training is auditable and resumable.

#### Acceptance Criteria

1. WHEN invoked without a CUDA GPU, THE Training_Script SHALL raise a RuntimeError with a message stating that local CPU training is not supported and exit with code 1.
2. WHEN invoked with more than one visible CUDA device, THE Training_Script SHALL raise a RuntimeError requiring `CUDA_VISIBLE_DEVICES=0` and exit with code 1.
3. THE Training_Script SHALL select `bf16` precision when GPU compute capability is ≥ 8.0, and `fp16` otherwise.
4. THE Training_Script SHALL apply LoRA with r=16, alpha=32, dropout=0.05, targeting all linear layers except `lm_head` and `embed_tokens`, with LoRA weights kept in fp32.
5. THE Training_Script SHALL load the base model with `attn_implementation="sdpa"`, `device_map={"": 0}`, and `local_files_only=True`, with `HF_HUB_OFFLINE=1` set before any model load.
6. THE Training_Script SHALL use an effective batch size of 32 dialogues, with gradient accumulation = 32 // micro_batch, cosine LR schedule, 3% warmup, weight decay 0, and gradient clip 1.0.
7. THE Training_Script SHALL evaluate every 100 steps on both `val_meddialog_tok` and `val_healthchat_tok`, compute macro val loss as their arithmetic mean, save checkpoints every 100 steps, and keep the last 3 plus the best checkpoint by macro val loss.
8. WHEN a `--resume` path is provided and exists, THE Training_Script SHALL resume from that checkpoint, producing a next-step loss within 2% of the uninterrupted run (kill-and-resume acceptance criterion).
9. WHEN training completes, THE Training_Script SHALL write `run_manifest.json` to the output directory containing sorted keys, SHA-256 of each input file, template SHA-256, system-prompt SHA-256, GPU info, package versions, trainable parameter count, and training hyperparameters.
10. THE Training_Script SHALL NOT import torch, peft, or accelerate at module level; all GPU-dependent imports SHALL be inside functions so that the CPU test suite can import the module without error.
11. THE Training_Script SHALL accept `--help` and SHALL exit with a nonzero code on argument errors.

---

### Requirement 5 — Gradient-Accumulation Equivalence (T7)

**User Story:** As a training engineer, I want to verify that the token-mean loss and collator are mathematically equivalent under different micro-batch / accumulation splits so that the effective batch size is consistent regardless of GPU memory configuration.

#### Acceptance Criteria

1. THE test T7 SHALL run only when torch is installed (decorated with `@unittest.skipUnless(_TORCH_AVAILABLE, ...)`), and SHALL be skipped silently on the CPU venv.
2. WHEN torch is available, THE test T7 SHALL assert that micro-batch 1 × accumulation 4 and micro-batch 4 × accumulation 1 produce parameter gradients with maximum absolute difference less than 1e-3, using 4 synthetic tokenized items in fp32 with LoRA dropout set to 0.
3. THE Kaggle_Notebook section "T7" SHALL implement the full T7 run using real tokenized data from `overfit_set.jsonl`, report the max absolute gradient difference, and assert it is less than 1e-3.

---

### Requirement 6 — E0 GPU Memory and Throughput Check

**User Story:** As a training engineer, I want an automated E0 check that measures VRAM usage and training throughput at each candidate micro-batch size so that the largest safe micro-batch can be selected before E2 pilots begin.

#### Acceptance Criteria

1. THE Kaggle_Notebook section "E0" SHALL run 50 training steps each at micro-batch 1, 2, and 4, measuring peak VRAM allocated and counting NaN/inf occurrences in loss and gradient norm.
2. WHEN any micro-batch size produces peak VRAM ≥ 90% of total VRAM or any NaN/inf, THE Kaggle_Notebook SHALL mark that micro-batch size as failing the `fit_90pct` criterion.
3. THE Kaggle_Notebook section "E0" SHALL measure realistic tokens-per-second on length-grouped batches drawn from `pilot_mix.jsonl` using the best passing micro-batch size.
4. THE Kaggle_Notebook section "E0" SHALL write `e0_results.json` to `/kaggle/working/results/` containing peak VRAM, VRAM percentage, NaN/inf count, fit status, and tokens-per-second for each micro-batch size, with no corpus text.
5. IF no micro-batch size passes the fit criterion, THEN THE Kaggle_Notebook SHALL halt with an assertion error rather than continuing to E2.

---

### Requirement 7 — Overfit Sanity Test

**User Story:** As a training engineer, I want an overfit test on 16 dialogues to verify that the model and training loop can memorize a tiny set, confirming that gradients flow correctly before committing to full E2 pilots.

#### Acceptance Criteria

1. THE Kaggle_Notebook section "Overfit" SHALL train on exactly 16 dialogues from `overfit_set.jsonl` for 150 steps at a learning rate of 1e-3 with LoRA dropout set to 0.
2. THE Kaggle_Notebook section "Overfit" SHALL record the loss at step 0 and the loss at step 149 and assert that `loss_step_149 / loss_step_0 ≤ 0.30`.
3. THE Kaggle_Notebook section "Overfit" SHALL write `overfit_results.json` to `/kaggle/working/results/` containing initial loss, final loss, ratio, pass status, and loss curve at 25-step intervals, with no corpus text.

---

### Requirement 8 — Kill-and-Resume Test

**User Story:** As a training engineer, I want a kill-and-resume test to verify that training can be interrupted and restarted from a checkpoint without diverging, because Kaggle sessions can be interrupted at any time.

#### Acceptance Criteria

1. THE Kaggle_Notebook section "Kill-and-Resume" SHALL run an uninterrupted training of K+1 steps and record the loss at step K+1.
2. THE Kaggle_Notebook section "Kill-and-Resume" SHALL run a separate training of K steps, save a checkpoint with optimizer and RNG state, resume from that checkpoint, run one more step, and record the loss at step K+1.
3. THE Kaggle_Notebook section "Kill-and-Resume" SHALL assert that the relative difference between the resumed loss and the uninterrupted loss at step K+1 is less than 2%.
4. THE Kaggle_Notebook section "Kill-and-Resume" SHALL write `resume_results.json` to `/kaggle/working/results/` containing both loss values, the relative difference, and the pass status, with no corpus text.

---

### Requirement 9 — E2 Learning-Rate Pilots

**User Story:** As a training engineer, I want three 150-step LR pilot runs at learning rates 5e-5, 1e-4, and 2e-4 on the identical pilot mix so that the best learning rate can be selected before the full training run.

#### Acceptance Criteria

1. THE Kaggle_Notebook section "E2" SHALL run three sequential 150-step training runs at learning rates 5e-5, 1e-4, and 2e-4, each initialized from the base model weights with seed 42.
2. THE Kaggle_Notebook section "E2" SHALL evaluate macro val loss every 50 steps on `val_meddialog_tok` and `val_healthchat_tok` for each pilot.
3. THE Kaggle_Notebook section "E2" SHALL write `e2_results.json` to `/kaggle/working/results/` containing val loss curves (step and macro val loss only), final macro val loss per LR, and train loss curves (step and loss only), with no corpus text.
4. THE Kaggle_Notebook section "E2" SHALL use the micro-batch size selected by E0 and the same effective batch of 32 for all three pilots.

---

### Requirement 10 — Local CPU Notebook

**User Story:** As a data engineer, I want a local CPU notebook that builds the tokenizer package, runs T1–T6 on real train data, builds all training mixes, and produces an upload manifest so that all Kaggle inputs can be verified before uploading.

#### Acceptance Criteria

1. THE Local_Notebook SHALL have a parameters cell at the top, a definitions section, an in-notebook tests section that executes before any dataset is read, and a final status cell.
2. THE Local_Notebook in-notebook tests section SHALL pass all assertions before proceeding to data processing; WHEN any assertion fails, THE Local_Notebook SHALL halt via `assert`.
3. THE Local_Notebook SHALL run T1 on 1,000 seeded-random train items from `dialogues_train.jsonl` and assert zero mismatches.
4. THE Local_Notebook SHALL run T2–T5 on every item in `pilot_mix.jsonl`, `overfit_set.jsonl`, `val_meddialog_tok.jsonl`, and `val_healthchat_tok.jsonl` and assert all pass.
5. THE Local_Notebook SHALL assert that at least 90% of eligible multi-exchange train dialogues are untruncated at `max_len=2048`.
6. THE Local_Notebook SHALL write `sft/kaggle/upload_manifest.json` listing the SHA-256 and byte size of every file needed for Kaggle upload, with no corpus text in any value.
7. THE Local_Notebook SHALL write `sft/reports/p3_tokenize_report.json` containing tokenizer hashes, mix statistics, T1–T5 pass/fail counts, and the multi-exchange untruncated percentage, with no corpus text.
8. WHEN executed top-to-bottom with `sft/.venv-nb`, THE Local_Notebook SHALL produce no error cells.
9. THE Local_Notebook SHALL be valid nbformat (passes `nbformat.validate`) with unique cell ids.

---

### Requirement 11 — Kaggle Notebook

**User Story:** As a training engineer, I want a complete Kaggle notebook that runs environment checks, T1–T6, E0, overfit, kill-and-resume, T7, and E2 in sequence so that all P3 acceptance criteria can be verified in one Kaggle session.

#### Acceptance Criteria

1. THE Kaggle_Notebook SHALL have a parameters cell at the top, a definitions section, an in-notebook tests section that runs before model loading, and a final status cell.
2. THE Kaggle_Notebook in-notebook tests section SHALL run `kaggle_env_check.py` and assert it exits 0 before any further cells execute.
3. THE Kaggle_Notebook SHALL run all T1–T6 test classes from `sft/tests/test_tokenize_mask.py` and write results to `t1_t6_results.json`; WHEN any test fails, THE Kaggle_Notebook SHALL halt via `assert`.
4. THE Kaggle_Notebook SHALL execute sections E0, Overfit, Kill-and-Resume, T7, and E2 in that order, writing a separate JSON result file to `/kaggle/working/results/` for each section.
5. THE final status cell of THE Kaggle_Notebook SHALL list all result files with their sizes and SHA-256 prefixes, and SHALL print a human-readable acceptance summary showing pass/fail for each section.
6. WHEN executed top-to-bottom on a Kaggle GPU session, THE Kaggle_Notebook SHALL produce no error cells.
7. THE Kaggle_Notebook SHALL set `CUDA_VISIBLE_DEVICES=0` and `HF_HUB_OFFLINE=1` before any torch or transformers import.
8. THE Kaggle_Notebook SHALL be valid nbformat (passes `nbformat.validate`) with unique cell ids.

---

### Requirement 12 — Upload and Verification Manifest

**User Story:** As a training engineer, I want a verified upload manifest that records SHA-256 hashes of every file destined for Kaggle so that I can confirm the correct versions are attached to the Kaggle notebook before training begins.

#### Acceptance Criteria

1. THE Local_Notebook SHALL produce `sft/kaggle/upload_manifest.json` listing path, SHA-256, and byte size for every file in the three Kaggle datasets (tokenizer_pkg files, code modules, and training data files), with sorted keys and LF newlines.
2. WHEN `sft/kaggle/verify_uploads.py` is run against `sft/kaggle/upload_manifest.json`, THE verify_uploads script SHALL exit 0 if all listed files exist and their SHA-256 values match, and exit 1 with a descriptive error for any mismatch or missing file.
3. THE Kaggle_Notebook SHALL verify SHA-256 of all attached input files against the values in the embedded or attached upload manifest before loading any model or data, and SHALL halt if any hash mismatches.

---

### Requirement 13 — Privacy and Determinism

**User Story:** As a data steward, I want all outputs to contain only counts, ids, hashes, and enum strings so that no dataset text is exposed in committed files, logs, or notebook cell outputs.

#### Acceptance Criteria

1. THE Local_Notebook cell outputs SHALL NOT contain any dataset utterance text; only counts, ids, hashes, and enum strings are permitted.
2. THE Kaggle_Notebook cell outputs SHALL NOT contain any dataset utterance text; only counts, ids, hashes, and enum strings are permitted.
3. WHEN producing any JSON report, THE Mix_Builder, THE Tokenizer_Package_Builder, and THE Training_Script SHALL write JSON with `ensure_ascii=False`, sorted keys, and LF newlines.
4. THE Mix_Builder, THE Tokenizer_Package_Builder, and THE Training_Script SHALL produce deterministic output given the same inputs and seed; re-running SHALL produce files with identical SHA-256 values.
5. IF a privacy scan samples 200 utterances of at least 40 characters from the source CSV and checks for substring presence in any new notebook JSON or `sft/reports/*.json`, THEN no utterance SHALL appear as a substring.

---

### Requirement 14 — CPU-Only Compatibility

**User Story:** As a developer running CPU-only checks, I want all non-GPU modules to be importable and testable without torch, peft, or accelerate so that the full CPU test suite passes without installing GPU packages.

#### Acceptance Criteria

1. THE Tokenize_Mask_Module SHALL be importable on `sft/.venv` without torch, peft, or accelerate being installed.
2. THE Mix_Builder SHALL be importable on `sft/.venv` without torch, peft, or accelerate being installed.
3. THE Tokenizer_Package_Builder SHALL be importable on `sft/.venv` without torch, peft, or accelerate being installed.
4. THE Training_Script SHALL be importable on `sft/.venv` without raising any import error; all torch imports SHALL be inside functions decorated with lazy import patterns or inside `train()`.
5. WHILE running on `sft/.venv`, THE test suite SHALL run all T1–T6 test cases and THE test T7 SHALL be automatically skipped via `@unittest.skipUnless(_TORCH_AVAILABLE, ...)`.
6. WHEN `HF_HUB_OFFLINE=1` is set, THE entire CPU test suite SHALL pass without any network request.

---

### Requirement 15 — Verification Gates

**User Story:** As a project owner, I want all existing verification gates to remain green after P3 so that no earlier deliverable is inadvertently broken.

#### Acceptance Criteria

1. AFTER all P3 deliverables are complete, THE `sft/verify_manifest.py` script SHALL exit 0.
2. AFTER all P3 deliverables are complete, THE `sft/verify_frozen.py` script SHALL exit 0.
3. AFTER all P3 deliverables are complete, running `sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests` SHALL report zero failures and zero errors.
4. AFTER all P3 deliverables are complete, running `sft\.venv\Scripts\python.exe -m compileall -q sft` SHALL exit 0 with no syntax errors.
5. THE `sft/tokenizer_snapshot/` directory SHALL remain unmodified; its files SHALL match the hashes in `sft/protected_manifest.json`.
6. THE `data/` directory contents SHALL remain unmodified; source CSV files SHALL match their recorded hashes.
