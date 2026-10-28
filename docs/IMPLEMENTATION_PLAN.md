# Medical Chat LLM (CAPSTONE)

A separately fine-tuned, open-weight chat model that answers general health questions across several turns using only its weights and the running conversation. No retrieval and no external LLM at inference.

> **Not a medical device and not clinical advice.** Training data includes LLM-written answers that were never clinically validated. This project makes behavioral claims only, not claims of medical correctness.

## Status

| Item | Status |
|---|---|
| Repository and dataset audit | Complete |
| Architecture and model selection | Complete, approved (`ARCHITECTURE.md`) |
| Implementation phases P0–P5 | In progress. Update this table as each phase is accepted |

Source of truth for design decisions: `ARCHITECTURE.md`. Change it only through a written amendment.

## Approved design (summary)

- **Model:** `Qwen/Qwen3-1.7B` (post-trained checkpoint, Apache-2.0), pinned to an exact revision.
- **Method:** LoRA SFT (r=16, all linear layers) with `transformers` Trainer + PEFT on one GPU. No TRL, no RLHF/DPO.
- **Data:** the validated CSV is read-only. A dialogue builder reconstructs complete dialogues as JSONL with dialogue-aware cleaning.
- **Format and loss:** fixed ChatML template with one system prompt, no `<think>` blocks, loss on assistant tokens only.
- **Splits:** by `group_id` (exact and near-duplicate clusters), so a dialogue never spans splits.
- **Behavior supplement:** about 240 hand-reviewed dialogues (identity, escalation, uncertainty, context carry-over).
- **Evaluation:** held-out NLL, history ablation, scripted probes, blind human rubric. Gates are frozen before the final training run.
- **Inference:** offline `ChatSession` with the same template, whole-exchange history trimming, and a deterministic red-flag banner (regex, not a model).

## Data

Source of record: `data/processed/integrated_conversations_ner_validated.csv` (documented counts, to be re-verified by gate G1).

| Source | Turns | Dialogues |
|---|---|---|
| MedDialog | 224,328 | 112,164 (2 turns each) |
| HealthChat-LMSYS | 28,183 | 6,828 |
| HealthChat-WildChat | 23,128 | 4,264 |
| **Total** | **275,639** | **123,256** |

- MedDialog is single-exchange doctor–patient Q&A. All multi-exchange signal comes from HealthChat (real user–LLM conversations).
- NER columns are not used for training (labels are limited to DISEASE/CHEMICAL and are known to be mislabeled).
- `integrated_conversations_clean.csv` and any existing SFT exports are quarantined: never used as inputs, never modified or deleted.

### Licensing and privacy

| Source | Terms |
|---|---|
| MedDialog | No open license declared; research use only; do not redistribute raw text |
| HealthChat-LMSYS | LMSYS-Chat-1M license agreement; no redistribution or re-identification |
| HealthChat-WildChat | AI2 ImpACT Low Risk license; attribution required |

Data and trained weights stay private. Get legal review before any release.

## Repository layout

```text
data/processed/     source CSVs (read-only)
notebooks/          04-06: NER, validation, cleaning (do not modify)
sft/                all new code for this project
  <env, pins, manifests, tests>   TODO: fill in after phase P0a
ARCHITECTURE.md     approved architecture and plan
```

## Implementation phases

| Phase | Goal |
|---|---|
| P0 | Environment and access (P0a: CPU env + tokenizer; P0b: weights + GPU probe) |
| P1 | Dialogue builder, gate G1, `dialogues_all.jsonl`, `prep_report.json` |
| P2 | Splits, behavior set, probes, baseline eval, frozen gates |
| P3 | Tokenize and mask (tests T1–T7), E0/E2 pilots |
| P4 | Final LoRA training |
| P5 | Evaluation (V1–V7), inference CLI, report |

## Setup

TODO: replace with exact commands after P0a. Expected shape:

```text
1. Create the environment from the pinned requirements in sft/
2. Run the test suite
3. Run the protected-manifest verifier (must exit 0)
```

Training needs a GPU with at least 15 GB. Data building and the demo run on CPU.

## Ground rules

- Never delete or modify original datasets, notebooks, SFT exports or checkpoints.
- Run the protected-manifest verifier at the end of every task.
- Split by `group_id`, never by row. Use the test split once.
- Do not paste dataset text into logs, reports or chats.

## Limitations

- Multi-turn data is thin and LLM-generated.
- A 1.7B model can hallucinate. Uncertainty and escalation behavior are tested, not guaranteed.
- No clinical validation. Safety probes are rated by non-clinicians unless a clinician is available.

## License

TODO: choose a license for the code. Model and data terms are listed above.
