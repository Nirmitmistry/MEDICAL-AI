---

inclusion: always

---

# CAPSTONE standing rules (apply to every task)

## 1. Authority order

1. docs/ARCHITECTURE.md: source of truth. Changed only by a written amendment from the project owner.

2. docs/IMPLEMENTATION_PLAN.md: stages, environments, ground rules.

3. docs/DATASET.md: CSV schema and documented counts.

4. docs/AUDIT.md: background only.

Any other planning file in the repo (DATA_PIPELINE, PROJECT_PLAN, FINE_TUNING_RL_STRATEGY, EVALUATION_SAFETY or similar) is superseded. Do not implement from it. It contradicts the approved design (8B model, TRL, rewrite-to-assistant, DPO/GRPO, entity conditioning, merging or renumbering turns, using the clean CSV).

If a task prompt conflicts with ARCHITECTURE.md, stop and report the conflict. Do not choose a side.

## 2. Fixed decisions (never change silently)

- Model Qwen/Qwen3-1.7B, pinned revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e. LoRA SFT through transformers Trainer + PEFT. No TRL, Unsloth, DeepSpeed, QLoRA, RLHF or DPO.

- Own ChatML template, no <think> blocks, assistant-only loss, pad id different from <|im_end|>.

- data/processed/integrated_conversations_ner_validated.csv is the only data source and is read-only. integrated_conversations_clean.csv and any existing SFT export are not inputs.

- Dialogue integrity: no row-level deletion, no renumbering, no merging of turns. Cut at the first defect (rule R3).

- Entity columns are not used for training or serialization.

- Splits by group_id. Truncation only at turn boundaries. No packing.

- Data and weights stay private.

## 3. Environment

- Local machine: Windows 10, PowerShell, CPU only. Interpreter: sft\.venv\Scripts\python.exe. About 15 GB RAM total but only about 3 GB free: stream or chunk, never hold duplicate copies of the full dataset.

- The local RTX 3050 is not a training target. Never write code that assumes a local GPU. All GPU work is Kaggle-only (IMPLEMENTATION_PLAN, Stage 3). CPU-stage modules must not import torch, peft or accelerate.

- Allowed dependencies: those pinned in sft/requirements-cpu.txt (transformers, pandas, datasketch, langid), their already-installed dependencies, and the standard library. Do not install or add packages. No linter or type checker is installed; do not install one.

- Notebook execution uses sft\.venv-nb. The original sft\.venv remains the environment for the existing smoke tests and verify_manifest.py.

- Open files with explicit encoding="utf-8". Write JSONL with newline="\n" and ensure_ascii=False. All outputs deterministic (sorted keys, stable ordering, fixed seeds).

## 4. Code rules

- Inspect existing files before modifying. Reuse existing code.

- New deliverables are Jupyter notebooks (.ipynb) under sft/notebooks/. Do not create new .py modules unless the task explicitly names one. Existing P0a .py files stay. Notebooks are saved WITH outputs, and every output must contain counts, hashes, ids and enum strings only. Each notebook has a parameters cell at the top, a definitions section, an in-notebook tests section that must pass before any real data is read (halt via assert), and a final status cell. Notebooks must be valid nbformat (nbformat.validate) with cell ids.

- No placeholders, stubs, mock logic or unexplained TODOs in production code. Tests use synthetic fixtures only.

- Type hints on public functions. argparse CLIs with --help and meaningful nonzero exit codes.

## 5. Privacy

- Never print, log, or put dataset text in chat reports or committed files. Reports contain counts, ids, hashes and enum strings only. Any artifact containing dataset text goes under sft/work/ (git-ignored) and is named in the report by path only.

- Notebook cell outputs count as reports. Never display dataframe rows, samples or head() of dataset text.

## 6. Verification before claiming done (run from repo root; all must pass)

1. sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests   (all earlier tests must still pass)

2. sft\.venv\Scripts\python.exe -m compileall -q sft

3. sft\.venv\Scripts\python.exe sft\verify_manifest.py   (exit 0)

4. git status --short: only files the task allows.

5. Execute each new notebook top to bottom with the sft\.venv-nb python: `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <path>` (if that exact form fails, use the equivalent that works and report it). It must finish with no error cells.

6. Privacy scan: take 200 random utterances of at least 40 characters from the source CSV and assert that none appears as a substring in any new notebook's JSON or any sft/reports/*.json. Do this with an inline python -c or a temp script under sft/work/, never a committed file. (Required from P1a onward; skip for notebooks that do not touch the dataset.)

Never mark a task complete if a check fails. Fix failures you caused. Report anything else as unresolved. Never weaken or delete tests to make them pass.

## 7. Execution style

- Batch independent actions in parallel only when they touch disjoint files. Respect dependencies.

- Do not start the next phase. After the checks, stop.

- Update .kiro/specs/capstone-impl/tasks.md only after verification passes.

## 8. Final report format (mandatory, plain markdown)

1 Files created. 2 Files modified. 3 What was implemented. 4 Commands run, with exit codes and key output lines. 5 Tests: passed/failed/skipped counts and names of failures. 6 Measured results the task asked for. 7 Deviations from the task or architecture, with reasons. 8 Unresolved issues. 9 Manual steps still required. 10 Output of git status --short.

Report only what you actually ran. If something was not run, say so.
