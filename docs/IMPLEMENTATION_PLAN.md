# Medical Chat LLM (CAPSTONE)

A separately fine-tuned, open-weight chat model that answers general health questions across several turns using only its weights and the running conversation. No retrieval and no external LLM at inference.

> **Not a medical device and not clinical advice.** Training data includes LLM-written answers that were never clinically validated. This project makes behavioral claims only, not claims of medical correctness.

---

## Status

| Item | Status |
|---|---|
| Repository and dataset audit | Complete |
| Architecture and model selection | Complete, approved (`ARCHITECTURE.md`) |
| Implementation stages S1–S4 | In progress. Update this table as each stage is accepted. |

Source of truth for design decisions: `ARCHITECTURE.md`. Change it only through a written amendment.

---

## Architecture overview

| Environment | Responsibility |
|---|---|
| **Local PC (CPU only)** | Development, data preparation, dataset inspection, validation, CPU-compatible testing, documentation |
| **Kaggle (GPU)** | All GPU-dependent operations: model loading for inspection, LoRA fine-tuning, GPU-accelerated evaluation, checkpoint generation |

The local RTX 3050 (4 GB VRAM) is **not used for training**. It is insufficient for LoRA fine-tuning of Qwen3-1.7B, which requires approximately 7–11 GB VRAM. All training runs on Kaggle.

---

## Approved design (summary)

- **Model:** `Qwen/Qwen3-1.7B` (post-trained checkpoint, Apache-2.0), pinned to revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Method:** LoRA SFT (r=16, all linear layers) with `transformers` Trainer + PEFT on one GPU. No TRL, no RLHF/DPO.
- **Data:** the validated CSV is read-only. A dialogue builder reconstructs complete dialogues as JSONL with dialogue-aware cleaning.
- **Format and loss:** fixed ChatML template with one system prompt, no `<think>` blocks, loss on assistant tokens only.
- **Splits:** by `group_id` (exact and near-duplicate clusters), so a dialogue never spans splits.
- **Behavior supplement:** about 240 hand-reviewed dialogues (identity, escalation, uncertainty, context carry-over).
- **Evaluation:** held-out NLL, history ablation, scripted probes, blind human rubric. Gates are frozen before the final training run.
- **Inference:** offline `ChatSession` with the same template, whole-exchange history trimming, and a deterministic red-flag banner (regex, not a model).

---

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

---

## Repository layout

```text
data/processed/     source CSVs (read-only, protected by manifest)
notebooks/          01–08: EDA, NER, validation, cleaning (do not modify)
scripts/            dataset preparation and retrieval pipeline scripts
sft/                all new code for training and evaluation
  requirements-cpu.txt       local CPU development dependencies
  requirements-kaggle.txt    Kaggle training dependencies (peft, accelerate only)
  freeze.txt                 full CPU env pin
  freeze-gpu.txt             historical local GPU env (RTX 3050; not for Kaggle)
  tokenizer_snapshot/        Qwen3-1.7B tokenizer files (git-ignored weights)
  model_snapshot/            Qwen3-1.7B weight files (git-ignored)
  tests/                     CPU smoke tests
  gpu_probe.py               Kaggle GPU verification utility
  kaggle_env_check.py        Kaggle session environment verifier
  load_check.py              Offline model structure/tokenizer checks
  download_model.py          Model weight download and SHA-256 verification
  build_manifest.py          Protected-file manifest builder
  verify_manifest.py         Protected-file manifest verifier (stdlib-only)
docs/               Architecture and technical documentation
```

---

## Implementation stages

### Stage 1 — Data Preparation *(local, CPU)*

**Goal:** produce clean, validated, split-ready dialogue JSONL from the source CSV.

| Deliverable | Description |
|---|---|
| Gate G1 pass | Source CSV row/dialogue counts match documented figures |
| `dialogues_all.jsonl` | All valid dialogues after builder rules R1–R7 |
| `prep_report.json` | Drop counts by reason, per source (conservation check) |
| E1 histograms | Length, language, template distribution → sets `max_len`, regex thresholds |

**Accept criteria:**
- G1 passes, or any discrepancy is explained and approved.
- Schema valid on 100% of records; alternation and contiguity 100%.
- R7 conservation closes per source.
- Residual template/`chatdoctori` ≤ 0.1% of assistant turns.
- Drop counts reported by reason.

**Environment:** Local PC, CPU only.

---

### Stage 2 — Splits, Behavior Set, Frozen Gates *(local, CPU)*

**Goal:** produce final split manifest, behavior dialogues, probes, and freeze evaluation gates before any training.

| Deliverable | Description |
|---|---|
| Split manifest | Train/val/test splits by `group_id` with MinHash audit |
| Behavior set | ~240 hand-reviewed dialogues (identity, escalation, uncertainty) |
| Probe set | ~150 scripted evaluation probes |
| `eval/gates.yaml` | Frozen evaluation thresholds (committed before training) |

**Accept criteria:**
- Zero `group_id` overlap across splits and zero exact normalized first-turn overlap.
- MinHash audit (Jaccard 0.8): zero val/test dialogues matching any train dialogue.
- Quotas met: val and test each have ≥500 MedDialog + ≥500 HealthChat (≥400 multi-exchange each).
- Manifest hash reproducible on re-run.
- Behavior and probe scenario families are disjoint.
- Gates committed before any training.

**Environment:** Local PC, CPU only. Baseline Qwen3-1.7B evaluation (for gate-freezing) may run on Kaggle if CPU is too slow.

---

### Stage 3 — Kaggle Training Environment + Fine-Tuning *(Kaggle, GPU)*

**Goal:** set up the Kaggle environment, run pilots, execute final training.

#### 3a — Kaggle setup

Before running training:
1. Upload the `sft/` directory (excluding `.venv*` and `model_snapshot/`) as a Kaggle dataset, or use the Kaggle notebook's attached dataset feature.
2. Upload model weights (`sft/model_snapshot/Qwen3-1.7B/`) as a private Kaggle dataset.
3. Upload training data (`dialogues_train.jsonl`, `dialogues_val.jsonl`) as a private Kaggle dataset.
4. Enable GPU accelerator in Kaggle: Settings → Accelerator → GPU T4 x2 or P100.

**Notebook startup sequence:**

```python
# Cell 1: Environment check (must pass before proceeding)
import subprocess
subprocess.run(["python", "sft/kaggle_env_check.py", "--json", "/kaggle/working/kaggle_env.json"], check=True)

# Cell 2: Install missing packages (torch is pre-installed by Kaggle)
subprocess.run(["pip", "install", "-q", "-r", "sft/requirements-kaggle.txt"], check=True)

# Cell 3: GPU probe for the run manifest
subprocess.run(["python", "sft/gpu_probe.py", "--json", "/kaggle/working/gpu_probe_kaggle.json"], check=True)

# Cell 4: Model structure check (runs load_check.py checks A–E with GPU)
import os; os.environ["HF_HUB_OFFLINE"] = "1"
subprocess.run(["python", "sft/load_check.py", "--out", "/kaggle/working/load_check.json"], check=True)
```

**kaggle_env_check.py will exit 1 (blocking further cells) if:**
- Not running inside a Kaggle session.
- No CUDA GPU is found.
- Free VRAM < 6 GB.

#### 3b — Tokenize and mask

| Test | Assertion |
|---|---|
| T1 | Concatenated ids equal full-string tokenization on 1,000 random items. |
| T2 | Decoded supervised tokens equal the assistant contents plus `<\|im_end\|>`, in order. |
| T3 | Zero supervised tokens in system, user, headers or padding. |
| T4 | Every assistant turn ends in a supervised `<\|im_end\|>`, and pad id ≠ `<\|im_end\|>` id. |
| T5 | Truncated items still end on an assistant `<\|im_end\|>`. |
| T6 | Train/inference parity: `apply_chat_template` output equals training string byte-for-byte. |
| T7 | Gradient-accumulation equivalence within 1e-3. |

T1–T6 can be verified locally (CPU); T7 requires a GPU run.

#### 3c — Pilot experiments (E0, E2)

**E0** (≤0.5 GPU-hours): 50 steps at micro-batch 1/2/4, pick largest batch with peak < 90% VRAM and no NaN.

**E2** (3 × ~0.75 GPU-hours): 150-step pilots at LR {5e-5, 1e-4, 2e-4}. Pick lowest macro val loss with no spikes.

#### 3d — Final training

| Deliverable | Description |
|---|---|
| Adapter checkpoints | Saved every 100 steps; keep last 3 + best by val loss |
| `run_manifest.json` | Hashes for data, splits, template, system prompt, tokenizer, versions, GPU |

**Accept criteria:**
- Run completes (resumes handled).
- Final val loss ≤ min val loss + 0.05.
- The manifest holds hashes for all inputs.

**Environment:** Kaggle GPU (T4, P100, or whatever accelerator is available). Precision selected automatically by `kaggle_env_check.py` based on compute capability.

---

### Stage 4 — Evaluation, Inference, Finalization *(mixed)*

**Goal:** evaluate the fine-tuned model, build inference CLI, finalize documentation.

| Deliverable | Description |
|---|---|
| `eval_report.json` | V1–V7 evaluation results with CIs |
| Rater sheets | V5 safety probes, V7 blind pairwise |
| `ChatSession` + CLI | Offline inference with the same template and window rule |
| Final README + docs | Updated to reflect actual implementation |

**Evaluation suite:**

| ID | Question | Method | Gate |
|---|---|---|---|
| V1 | Did SFT learn the target distribution? | Test split NLL, per source | Not evidence of correctness |
| V2 | Does it use history? | Multi-exchange NLL: full vs truncated history | Gap has CI above 0, ≥ base's gap |
| V3 | Constraint carry-over | 40 scripted 3-exchange probes | Pass rate ≥ base + 15 points |
| V4 | Calibrated uncertainty | 40 probes (insufficient info, diagnosis requests) | Credential claims = 0 |
| V5 | Safety | 60 probes (emergencies, self-harm, dosing) | Escalation ≥ 90%; 0 harmful specifics |
| V6 | Format and leakage | All generations | EOS-terminated ≥ 98%; `<think>` = 0 |
| V7 | Preferred over base? | 50 blind pairwise comparisons | Win/tie/loss counts |

**Accept criteria:**
- V1–V7 pass, or failures documented with E3 remedy executed (≤1 rerun).
- A scripted 10-turn CLI conversation shows history use.
- Template-hash assertion and offline-load test pass.
- Test split used exactly once.

**Environment:** Kaggle GPU for V1/V2 NLL evaluation; local CPU for probes and CLI.

---

## Local development setup

```powershell
# From repo root (Windows PowerShell)
python -m venv sft/.venv
sft/.venv/Scripts/pip install --no-cache-dir -r sft/requirements-cpu.txt

# Run smoke tests
sft/.venv/Scripts/python -m unittest discover -v -s sft/tests

# Run the manifest verifier (must exit 0)
sft/.venv/Scripts/python sft/verify_manifest.py

# Download tokenizer files only (no weights)
sft/.venv/Scripts/python sft/download_model.py  # or --verify-only if already downloaded

# Run local model structure check (checks A–C; D/E deferred to Kaggle)
sft/.venv/Scripts/python sft/load_check.py
```

Expected local results:
- All smoke tests: PASS.
- Manifest verifier: exit 0.
- `load_check.py`: tier = PASS-PARTIAL (A/B/C pass; D/E deferred — run on Kaggle).

---

## Kaggle training setup

1. Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x2 or P100.
2. Attach datasets: model weights, training JSONL, `sft/` code.
3. In the first notebook cell, run `sft/kaggle_env_check.py` — it must exit 0.
4. Install extra packages: `pip install -q -r sft/requirements-kaggle.txt`.
5. Set `HF_HUB_OFFLINE=1` before loading any model.
6. Run `sft/gpu_probe.py` and save the result for the run manifest.
7. Proceed with training.

The training script should select `bf16` precision if `compute_capability >= 8.0`, else `fp16`. `kaggle_env_check.py` prints this decision at startup.

---

## Ground rules

- Never delete or modify original datasets, notebooks, SFT exports, or checkpoints.
- Run the protected-manifest verifier at the end of every task.
- Split by `group_id`, never by row. Use the test split once.
- Do not paste dataset text into logs, reports, or chats.
- All GPU training runs on Kaggle. The local RTX 3050 is not a training target.

---

## Limitations

- Multi-turn data is thin and LLM-generated.
- A 1.7B model can hallucinate. Uncertainty and escalation behavior are tested, not guaranteed.
- No clinical validation. Safety probes are rated by non-clinicians unless a clinician is available.

---

## License

TODO: choose a license for the code. Model and data terms are listed above.
