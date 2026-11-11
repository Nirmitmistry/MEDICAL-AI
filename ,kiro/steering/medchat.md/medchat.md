# MedChat (CAPSTONE) — Kiro steering

## 1. Authority and precedence
Authoritative, in this order:
1. docs/ARCHITECTURE.md — design decisions (section H lists what must not change silently).
2. docs/IMPLEMENTATION_PLAN.md — stages, deliverables, acceptance criteria, repo layout.
3. docs/DATASET.md and docs/AUDIT.md — source facts and risks. Counts are documented, not verified; gate G1 verifies them.
4. P0a_REPORT.md — record of completed P0a (environment, protected manifest, tokenizer snapshot).

Superseded, do not implement from: PROJECT.md, DATAPIPE.md, FINETUNE_RL.md, EVAL.md. They describe an 8B plan (Qwen3-8B, r=64, DPO/GRPO, vLLM, 8×GPU). Their ideas are not in scope unless ARCHITECTURE.md re-approves them. Known conflicts resolved in ARCHITECTURE.md's favour:
- Model: Qwen3-1.7B, not 8B.
- Dialogue builder: cut at first defect, never merge same-role turns, no 8-exchange truncation (ARCHITECTURE D3/E3). DATAPIPE §3 merges and truncates; ignore that.
- Input: validated CSV is the source. `integrated_conversations_clean.csv` is quarantined, not the preferred input (DATAPIPE §0 says otherwise).
- Splits: by `group_id` with quotas (ARCHITECTURE D/P2), not DATAPIPE §7 quotas.
- Test: `unittest` in `sft/tests` (as in P0a), not pytest.

## 2. Phase mapping
ARCHITECTURE uses P0–P5. IMPLEMENTATION_PLAN uses Stage 1–4. Map them as follows:
- P0 (environment) = done, per P0a_REPORT.
- P1 = Stage 1, Data preparation.
- P2 = Stage 2, Splits/behavior/probes/frozen gates.
- P3 + P4 = Stage 3, Kaggle training (tokenize/mask, E0/E2 pilots, final run).
- P5 = Stage 4, Evaluation/inference.
When a phase is named, quote the matching ARCHITECTURE section and IMPLEMENTATION_PLAN stage.

## 3. Architecture and module responsibilities
All new code lives under `sft/`. Do not create modules outside the layout in IMPLEMENTATION_PLAN "Repository layout" without a written amendment.
- Dialogue builder (P1): reads the validated CSV read-only, applies R1–R7, writes `dialogues_all.jsonl` and `prep_report.json`.
- Splits and behavior/probes (P2): group-aware split, MinHash audit, behavior set, probes, `eval/gates.yaml`.
- Tokenize and mask (P3): own ChatML template, assistant-only labels, tests T1–T7.
- Training (P3–P4): HF Trainer + PEFT LoRA, resumable checkpoints, `run_manifest.json`. Kaggle only.
- Evaluation (P5): V1–V7 harness, probes, rater sheets.
- Inference (P5): offline `ChatSession` with the same template and window rule; deterministic regex red-flag banner, not a model.

## 4. Technology stack
- Python 3.13 on the local CPU machine. Pinned in `sft/requirements-cpu.txt` and `sft/freeze.txt`.
- Model: `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0, local snapshot only.
- Libraries: transformers, pandas, datasketch, langid (CPU); torch, peft, accelerate (Kaggle). `bitsandbytes` only on the QLoRA rung. TRL, Unsloth, DeepSpeed, vLLM are excluded.
- Do not add libraries not listed in ARCHITECTURE E1 without an amendment.

## 5. Hard constraints
- The validated CSV is read-only. Load IDs with `dtype=str`. Never delete, rename or overwrite raw data, notebooks, processed CSVs, or SFT exports.
- No row-level deletion, renumbering or merging of turns. Cut at the first defect.
- Loss only on assistant content plus `<|im_end|>`, for every assistant turn. Pad ≠ EOS. No `<think>` anywhere.
- Entity columns are not used in training or serialization.
- Split by `group_id`. Test split used exactly once. Probe families disjoint from behavior-set families.
- Gates are frozen before the final training run and never loosened.
- GPU work runs only on Kaggle. The local RTX 3050 is not a training target.
- Data and weights stay private. No dataset text in logs, reports, or chat.
- Versions are frozen after the E0 smoke test. No mid-project upgrades.

## 6. Coding and testing conventions
- Stdlib-only verifier scripts (`verify_manifest.py` pattern) for anything run in CI or gating.
- Tests in `sft/tests/` using `unittest`. Tests T1–T7 are mandatory; T1–T6 run locally, T7 on GPU.
- Every stage writes a JSON report (counts, drops by reason, hashes). Conservation must close (R7).
- Seed 42 everywhere. Record hashes for data, splits, template, system prompt, tokenizer, versions, config.
- Keep modules small and single-purpose. Reuse existing helpers (`sft/` tree) before writing new ones.

## 7. Rules for executing the phased plan
1. Work one phase at a time. Do not start a phase until its predecessor's accept criteria are met and recorded in the IMPLEMENTATION_PLAN status table.
2. Begin each phase by restating its ARCHITECTURE section and acceptance criteria.
3. Run `sft/verify_manifest.py` at the end of every task; it must exit 0.
4. A failed gate stops the phase. Report the failure with counts. Do not patch around it or loosen a threshold.
5. A G1 mismatch stops P1 and is reported, not explained away.
6. Any change to a decision in ARCHITECTURE section H needs a written amendment and re-running the dependent gates.
7. Report what was measured, what was assumed, and what remains unverified. Do not claim a check passed unless it was run.