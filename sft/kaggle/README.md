# P3 Kaggle Training — Human Steps

This directory contains the files needed to run E0 (GPU memory/speed check),
the overfit test, the kill-and-resume test, T7 (gradient accumulation), and
E2 (3 × 150-step LR pilots) on Kaggle.

**Architecture references:** ARCHITECTURE.md sections C, E1–E5, G(P3), E8;
IMPLEMENTATION_PLAN Stage 3a–3c.

---

## 0. Prerequisites (complete locally first)

Before uploading to Kaggle, confirm all these exist on your local machine:

| File | Location | Description |
|---|---|---|
| `tokenizer_pkg/` | `sft/tokenizer_pkg/` | Custom ChatML tokenizer (built by A1) |
| `pilot_mix.jsonl` | `sft/work/pilot_mix.jsonl` | 4,800-dialogue pilot mix (A4) |
| `overfit_set.jsonl` | `sft/work/overfit_set.jsonl` | 16-dialogue overfit set (A4) |
| `val_meddialog_tok.jsonl` | `sft/work/val_meddialog_tok.jsonl` | 200-item val subset (A4) |
| `val_healthchat_tok.jsonl` | `sft/work/val_healthchat_tok.jsonl` | 200-item val subset (A4) |
| `upload_manifest.json` | `sft/kaggle/upload_manifest.json` | SHA-256 manifest of all uploads |
| Model weights | `sft/model_snapshot/` | Qwen3-1.7B local snapshot |

Verify: `sft\.venv\Scripts\python.exe sft\verify_manifest.py` — must exit 0.

---

## 1. Upload private Kaggle datasets

Create **private** datasets on kaggle.com/datasets:

### Dataset 1 — `capstone-tokenizer-sft-code`
Upload these files:
```
sft/tokenizer_pkg/            (all files)
sft/kaggle/verify_uploads.py
sft/kaggle/upload_manifest.json
sft/tokenize_mask.py
sft/train_lora.py
sft/build_mix.py
sft/gpu_probe.py
sft/kaggle_env_check.py
```

### Dataset 2 — `capstone-model-weights`
Upload:
```
sft/model_snapshot/Qwen3-1.7B/   (all weight files — may be multi-file upload)
```

### Dataset 3 — `capstone-train-data`
Upload:
```
sft/work/pilot_mix.jsonl
sft/work/overfit_set.jsonl
sft/work/val_meddialog_tok.jsonl
sft/work/val_healthchat_tok.jsonl
```

After each upload, note the **dataset version URL** and record it in `upload_manifest.json`.

---

## 2. Create the Kaggle notebook

1. Go to kaggle.com → Code → New Notebook.
2. Settings → Language: Python.
3. Settings → Accelerator: **GPU T4 x2** (we lock to 1 GPU in code).
4. Settings → Internet: **Off** (required for HF_HUB_OFFLINE=1).
5. Attach all 3 datasets created above.
6. Upload `sft/kaggle/p3_kaggle_notebook.ipynb` as the notebook, **or** copy the
   percent-format `.py` notebook and paste cells manually.

---

## 3. Run the notebook sections in order

The notebook is structured as:

| Section | Description | Saves to `/kaggle/working/results/` |
|---|---|---|
| **Env** | `kaggle_env_check.py` + GPU probe + pip freeze | `env.json` |
| **T1-T6** | Rerun tests on Kaggle's transformers version | `t1_t6_results.json` |
| **E0** | 50 steps at micro-batch 1/2/4; VRAM + tok/s | `e0_results.json` |
| **Overfit** | 150 steps on 16 dialogues | `overfit_results.json` |
| **Kill-resume** | Train to step k, hard-kill, resume, compare step k+1 loss | `resume_results.json` |
| **T7** | Gradient accumulation equivalence (fp32) | `t7_results.json` |
| **E2** | 3 × 150 steps at LR 5e-5 / 1e-4 / 2e-4 | `e2_results.json` |

---

## 4. After the notebook completes

1. Download all JSONs from `/kaggle/working/results/` to `sft/results/p3/`.
2. Run locally: `sft\.venv\Scripts\python.exe sft\verify_manifest.py` — must exit 0.
3. Notify the agent to continue at Part B.

---

## Acceptance criteria (pre-B1 check)

| Criterion | Target |
|---|---|
| T1-T6 | All pass on Kaggle transformers |
| E0 max micro-batch | Peak < 90% VRAM, no NaN/inf |
| E0 tok/s | Recorded in `e0_results.json` |
| Overfit | NLL ≤ 30% of initial within 150 steps |
| Kill-resume | Loss at step k+1 within 2% of uninterrupted run |
| T7 | Max gradient diff < 1e-3 |
| E2 | 3 runs complete; LR chosen from results |

---

## Notes

- **Never** upload the test split (`dialogues_test.jsonl`) or raw CSVs.
- **Never** put dataset text in notebook outputs.
- All results JSONs contain counts, hashes, ids, and enum strings only.
- The model weights dataset must be **private** at all times.
