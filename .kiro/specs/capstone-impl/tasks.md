# CAPSTONE implementation task checklist

- [x] Task 0: workspace setup (check after Step 4 passes)

## Stage 1 — Data preparation (local CPU)

- [x] P1a G1 + R1-R3/R6 rules + E1 profiling
- [ ] P1b dialogue builder (R4, R5, R7) and prep_report.json

## Stage 2 — Splits, behavior set, probes, gates

- [ ] (sub-tasks added as prompts are issued)

## Stage 3 — Kaggle training

- [x] P3a Tokenize and mask (tokenize_mask.py, train_lora.py, build_mix.py, build_tokenizer_pkg.py) — T1-T6 pass, T7 skeleton on CPU; fix: pkg_resources → importlib.metadata in train_lora.py
- [x] P3b Kaggle notebook (sft/kaggle/p3_kaggle_notebook.ipynb) and local notebook (sft/notebooks/P3_tokenize_mix.ipynb) — both valid nbformat; local notebook must be executed by human (HA1)
- [ ] P3c Kaggle E0/E2 pilots — HUMAN ACTION PENDING (HA2: upload datasets; HA3: run Kaggle notebook; HA4: report results for Part B)

## Stage 4 — Evaluation, inference, finalization

- [ ] (sub-tasks added as prompts are issued)
