# Design Document: P3 — Tokenize and Mask, Training Script, Kaggle E0/E2 Pilots

## Overview

P3 delivers the complete tokenize-and-mask pipeline, the LoRA SFT training script, and the Kaggle experiment notebooks for the Medical Chat LLM capstone. The pipeline is split strictly between a CPU stage (tokenizer package build, label masking, mix construction, local notebook) and a GPU stage (LoRA training, E0/E2 pilot experiments on Kaggle). No GPU-dependent packages are imported at module level anywhere in the CPU stage.

The design follows ARCHITECTURE.md sections A–E8 and IMPLEMENTATION_PLAN Stage 3a–3c. All fixed decisions—Qwen3-1.7B at pinned revision, ChatML template without `<think>`, assistant-only loss, pad≠EOS, LoRA r=16/alpha=32, no TRL/QLoRA/DPO—are enforced by assertions in the code, not by convention.

---

## Architecture and Component Overview

```
CPU stage (sft/.venv)
├── build_tokenizer_pkg.py   → sft/tokenizer_pkg/   (custom ChatML tokenizer)
├── tokenize_mask.py         → TokenizedItem (input_ids, labels, n_tokens, truncated, source, dialogue_id)
├── build_mix.py             → sft/work/*.jsonl  (pilot_mix, overfit_set, val subsets)
├── notebooks/P3_tokenize_mix.ipynb
│      └── runs T1–T6 on real data, produces p3_tokenize_report.json + upload_manifest.json
└── kaggle/verify_uploads.py → verify upload_manifest.json SHA-256 integrity

GPU stage (Kaggle only)
├── train_lora.py            → LoRA SFT trainer (HF Trainer + PEFT)
└── kaggle/p3_kaggle_notebook.ipynb
       ├── kaggle_env_check
       ├── T1–T6 on tokenized mixes
       ├── E0: micro-batch VRAM sweep
       ├── Overfit sanity (16 dialogues, 150 steps)
       ├── Kill-and-resume checkpoint test
       ├── T7: gradient-accumulation equivalence
       └── E2: three 150-step LR pilots (5e-5, 1e-4, 2e-4)
```

Data flows left-to-right: `integrated_conversations_ner_validated.csv` → JSONL splits (Phase 2) → tokenized JSONL mixes (Phase 3 CPU) → LoRA adapters + eval metrics (Phase 3 GPU).

---

## Component Designs

### 1. Tokenizer Package Builder (`sft/build_tokenizer_pkg.py`)

**Responsibility:** Produce `sft/tokenizer_pkg/` from the read-only snapshot with the custom ChatML template and pinned generation config.

**Key design decisions:**
- Copy every non-directory file from `tokenizer_snapshot/` using `shutil.copy2` to preserve metadata.
- Patch `tokenizer_config.json` in-place: set `chat_template` to `CUSTOM_CHAT_TEMPLATE`, `eos_token` to `<|im_end|>`, `pad_token` to `<|endoftext|>`.
- Patch `generation_config.json`: set `eos_token_id=[151645, 151643]`, `pad_token_id=151643`.
- Write `tokenizer_pkg_manifest.json` with SHA-256 of every output file, the template SHA-256, and the system-prompt SHA-256, using sorted keys and LF newlines.
- The snapshot directory is never written to; `build_tokenizer_pkg` only writes to `out_dir`.
- Idempotent: running twice produces byte-for-byte identical output because all JSON is written with `sort_keys=True` and deterministic content.

**Interface:**
```python
def build_tokenizer_pkg(
    snapshot_dir: Path,
    out_dir: Path,
    system_prompt_path: Path | None = None,
) -> dict:
    """Returns manifest dict. Raises FileNotFoundError / ValueError on failure."""
```

**ChatML template rules (Architecture D4):**
- No `<think>` blocks anywhere in the template.
- If `messages[0].role != 'system'`, inject `SYSTEM_PROMPT_V1` as the system block.
- Generation prompt: `<|im_start|>assistant\n`.
- EOS: `<|im_end|>` (token id 151645); pad: `<|endoftext|>` (token id 151643).

---

### 2. Tokenize-and-Mask Module (`sft/tokenize_mask.py`)

**Responsibility:** Convert raw JSONL dialogue records into `TokenizedItem` objects with ChatML-structured `input_ids` and assistant-only `labels`.

#### 2.1 Data classes

```python
@dataclass
class TokenizedItem:
    input_ids:    List[int]   # len <= max_len
    labels:       List[int]   # same len; -100 for masked positions
    n_tokens:     int
    truncated:    bool
    source:       str
    dialogue_id:  str

@dataclass
class TokenizationStats:
    kept:                           int = 0
    truncated:                      int = 0
    dropped_too_long:               int = 0
    dropped_special_token_in_content: int = 0
```

#### 2.2 Tokenizer loading

`load_tokenizer(pkg_dir)` calls `AutoTokenizer.from_pretrained` with `local_files_only=True`, then immediately asserts:
- `tok.eos_token_id == 151645`
- `tok.pad_token_id == 151643`
- `tok.pad_token_id != tok.eos_token_id`

#### 2.3 Segment-by-segment tokenization

Each dialogue is tokenized segment by segment (never as a single concatenated string) to guarantee label alignment:

```
Segment sequence per exchange:
  [sys_header][sys_content][IM_END][NL]           ← MASKED
  [user_header][user_content][IM_END][NL]          ← MASKED
  [asst_header][asst_content][IM_END][NL]          ← header+NL MASKED; content+IM_END SUPERVISED
```

T1 asserts that this segment-by-segment `input_ids` equals full-string tokenization on 1,000 seeded items.

#### 2.4 Label-masking rules (Architecture E2)

| Token position | Label |
|---|---|
| System block (header + content + `<|im_end|>` + `\n`) | `MASKED` (-100) |
| User turn (header + content + `<|im_end|>` + `\n`) | `MASKED` |
| Assistant role header (`<|im_start|>assistant\n`) | `MASKED` |
| Assistant content tokens | token id (supervised) |
| Assistant closing `<|im_end|>` | token id = 151645 (supervised) |
| Assistant trailing `\n` after `<|im_end|>` | `MASKED` |

#### 2.5 Truncation (Architecture E3)

```
max_len = 2048, hard_cap = 3072
Algorithm:
  Try n_ex = len(exchanges) down to 1:
    if len(build_tokens(first_n_ex)) <= max_len: return item (truncated=True if n_ex < len)
  If even n_ex=1 > max_len: return (None, "dropped_too_long")
```

T5 asserts that the last supervised token of a truncated item is `<|im_end|>`.

#### 2.6 Special-token sanitisation

`_has_literal_special_token(text)` applies a pre-compiled regex over all Qwen3 special-token patterns. Any content matching → `(None, "dropped_special_token")`.

#### 2.7 T6 parity function

`apply_chat_template_to_prefix(tok, messages)` calls `tok.apply_chat_template(..., add_generation_prompt=True)`. When jinja2 is unavailable (CPU venv), it falls back to `render_chat_template_no_jinja` which implements the same template in pure Python. T6 asserts byte-for-byte equality between the inference prefix and the training sequence prefix.

---

### 3. Training Mix Builder (`sft/build_mix.py`)

**Responsibility:** Stream-tokenize JSONL splits and produce output files in `sft/work/`.

#### 3.1 Output files

| File | Content | Size |
|---|---|---|
| `pilot_mix.jsonl` | 4,800 dialogues at token-share ratios MD=50%/HC=45%/BEH=5% | ~9.6M tokens |
| `overfit_set.jsonl` | Exactly 8 MedDialog + 8 HealthChat | 16 items |
| `val_meddialog_tok.jsonl` | First 200 MedDialog items from val split | 200 items |
| `val_healthchat_tok.jsonl` | First 200 HealthChat items from val split | 200 items |
| `build_mix_report.json` | Counts, hashes, shares — no corpus text | JSON |

#### 3.2 Token-share algorithm

```python
# Greedy deficit-based sampling:
# At each step, pick the source with the largest positive gap = target_share - current_share
# using a fixed seed=42 RNG for initial shuffle.
# Behavior items are upsampled 3× before shuffling.
```

#### 3.3 JSONL item format

```json
{"dialogue_id": "...", "source": "...", "input_ids": [...], "labels": [...], "n_tokens": 42, "truncated": false}
```

Written with `ensure_ascii=False` and `\n` newlines.

#### 3.4 Determinism constraints

- All random operations use `random.Random(seed)` — not `random.seed()` global state.
- Output files are always fully overwritten.
- JSON reports use `sort_keys=True`.
- Given identical inputs and seed, SHA-256 of every output file is stable across re-runs.

#### 3.5 Architecture E3 assertion

After building the pilot mix, `build_mix.py` asserts that at least 90% of eligible multi-exchange train dialogues are untruncated at `max_len=2048`.

---

### 4. LoRA Training Script (`sft/train_lora.py`)

**Responsibility:** Kaggle-only LoRA SFT using HF Trainer + PEFT. All torch/peft/accelerate imports are inside functions.

#### 4.1 GPU environment guard

```python
def check_gpu_environment() -> dict:
    import torch
    n = torch.cuda.device_count()
    if n == 0: raise RuntimeError("No CUDA GPU found ...")
    if n != 1: raise RuntimeError(f"Found {n} GPUs. Set CUDA_VISIBLE_DEVICES=0 ...")
    # returns {device_name, compute_capability, vram_gb, bf16_supported, precision}
```

#### 4.2 LoRA configuration (Architecture E4)

```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=<all_linear_except_lm_head_and_embed_tokens>,
    bias="none",
)
```

LoRA weights are kept in fp32 regardless of training precision.

#### 4.3 Model loading

```python
AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    attn_implementation="sdpa",
    device_map={"": 0},
    local_files_only=True,
)
```

`HF_HUB_OFFLINE=1` is set before this call.

#### 4.4 Training configuration

| Parameter | Value |
|---|---|
| Effective batch | 32 dialogues |
| Gradient accumulation | 32 // micro_batch |
| LR schedule | cosine |
| Warmup ratio | 3% |
| Weight decay | 0 |
| Gradient clip | 1.0 |
| Eval steps | 100 |
| Save steps | 100 |
| Save total limit | 3 (last 3 + best by macro val loss) |

#### 4.5 Token-mean loss (Architecture E4 / T7)

```python
class TokenMeanLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Divide by number of supervised tokens, not sequence length,
        # so effective batch loss is invariant to micro-batch size.
        num_items = inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if num_items is not None:
            # Normalize by token count not item count
            ...
        return (loss, outputs) if return_outputs else loss
```

#### 4.6 Resume from checkpoint

When `--resume <path>` is given and the path exists:
- Passes `resume_from_checkpoint=path` to `trainer.train()`.
- The kill-and-resume test asserts the resumed loss at step K+1 is within 2% of the uninterrupted run.

#### 4.7 Run manifest

After training completes, `run_manifest.json` is written with:
- SHA-256 of each input file (tokenizer_pkg, mix files)
- Template SHA-256 and system-prompt SHA-256
- GPU info dict from `check_gpu_environment()`
- Package versions (transformers, peft, torch)
- Trainable parameter count
- Full training hyperparameters
- Written with `sort_keys=True`, `ensure_ascii=False`, LF newlines.

---

### 5. Local CPU Notebook (`sft/notebooks/P3_tokenize_mix.ipynb`)

**Structure (runs on `sft/.venv-nb`, CPU only):**

| Section | Purpose |
|---|---|
| `0. Parameters` | `REPO_ROOT`, paths, `MAX_LEN`, `SEED` — all in one cell |
| `1. Definitions` | Imports and helper functions |
| `2. In-notebook tests (synthetic)` | T-shape checks using synthetic fixtures; must all pass via `assert` before data is read |
| `3. Build tokenizer_pkg` | Calls `build_tokenizer_pkg.build_tokenizer_pkg()` |
| `4. Tokenizer smoke checks` | Loads tokenizer, verifies EOS/PAD ids, no `<think>` in template |
| `5. Build mixes` | Calls `build_mix.main()` |
| `6. T1 on real train data` | 1,000 seeded-random items, assert 0 mismatches |
| `7. T2–T5 on built files` | Runs on pilot_mix, overfit_set, val subsets; assert all pass |
| `8. E3 assertion` | ≥90% multi-exchange items untruncated |
| `9. Upload manifest` | Writes `sft/kaggle/upload_manifest.json` |
| `10. Report` | Writes `sft/reports/p3_tokenize_report.json` (counts, hashes, ids only) |
| `11. Final status` | Prints PASS/FAIL summary |

**Privacy constraints:** No cell output may contain dataset text. All `print()` calls output counts, ids, hashes, or enum strings only.

**Nbformat requirements:** Valid nbformat v4, `nbformat.validate` passes, all cell ids are unique strings.

---

### 6. Kaggle Notebook (`sft/kaggle/p3_kaggle_notebook.ipynb`)

**Structure (runs on Kaggle T4/P100 GPU):**

| Section | Purpose |
|---|---|
| `0. Parameters` | `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, input/output paths |
| `1. Environment check` | Runs `kaggle_env_check.py`, asserts exit 0 |
| `2. T1–T6` | Imports and runs test classes; writes `t1_t6_results.json`; halts on failure |
| `3. E0` | 50-step VRAM sweep at micro-batch 1/2/4; writes `e0_results.json` |
| `4. Overfit` | 150 steps on `overfit_set.jsonl` at LR=1e-3, dropout=0; writes `overfit_results.json` |
| `5. Kill-and-Resume` | Two runs (K+1 uninterrupted vs K+checkpoint+1); writes `resume_results.json` |
| `6. T7` | Gradient-accumulation equivalence on `overfit_set.jsonl`; writes `t7_results.json` |
| `7. E2` | Three 150-step pilots at LR=5e-5/1e-4/2e-4; writes `e2_results.json` |
| `8. Final status` | Lists all result files with sizes and SHA-256 prefixes; prints acceptance summary |

All output paths are under `/kaggle/working/results/`. No corpus text in any output.

---

### 7. Upload Manifest and Verify Script (`sft/kaggle/upload_manifest.json`, `sft/kaggle/verify_uploads.py`)

The local notebook writes `upload_manifest.json` listing path, SHA-256, and byte size for all Kaggle upload files (tokenizer_pkg files, code modules, training data files). The verify script:

```python
def verify_manifest(manifest_path: Path, root: Path) -> tuple[bool, list[str]]:
    # For each entry: check file exists, compute sha256, compare to manifest value
    # Returns (all_ok, list_of_error_messages)
```

The Kaggle notebook performs the same SHA-256 verification at runtime before loading any model or data.

---

## Data Models

### TokenizedItem (Python dataclass)
```python
@dataclass
class TokenizedItem:
    input_ids:    List[int]   # 1 <= len <= max_len
    labels:       List[int]   # same length; MASKED=-100 for non-supervised positions
    n_tokens:     int         # == len(input_ids)
    truncated:    bool        # True if any exchange was dropped to fit max_len
    source:       str         # "MedDialog" | "HealthChat-..." | "Behavior"
    dialogue_id:  str         # from source JSONL
```

### JSONL item (on-disk format for tokenized mixes)
```json
{
  "dialogue_id": "MedDialog_00123",
  "input_ids": [151644, 9530, 198, ...],
  "labels": [-100, -100, -100, ..., 198, 151645, -100],
  "n_tokens": 342,
  "source": "MedDialog",
  "truncated": false
}
```

### build_mix_report.json (privacy-safe)
```json
{
  "overfit_set":  {"hc_count": 8, "md_count": 8, "n": 16, "sha256": "...", "path": "..."},
  "pilot_mix":    {"n_dialogues_requested": 4800, "n_dialogues_produced": 4800,
                   "total_tokens": ..., "token_shares": {...}, "tokenization_stats": {...}},
  "val_subsets":  {"meddialog": {"n": 200, "sha256": "..."}, "healthchat": {"n": 200, ...}}
}
```

### upload_manifest.json (privacy-safe)
```json
{
  "files": [
    {"path": "sft/tokenizer_pkg/tokenizer_config.json", "sha256": "...", "size_bytes": 12345},
    ...
  ],
  "created_at": "...",
  "schema": "upload_manifest_v1"
}
```

### run_manifest.json (Kaggle output, privacy-safe)
```json
{
  "gpu_info": {"device_name": "Tesla T4", "compute_capability": "7.5", ...},
  "hyperparameters": {"effective_batch": 32, "lr": 1e-4, ...},
  "input_sha256": {"pilot_mix.jsonl": "...", "tokenizer_config.json": "...", ...},
  "package_versions": {"transformers": "4.47.0", "peft": "...", "torch": "..."},
  "template_sha256": "...",
  "system_prompt_sha256": "...",
  "trainable_parameters": 3407872
}
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `tokenizer_snapshot/` missing | `build_tokenizer_pkg` raises `FileNotFoundError`; main exits 1 |
| System prompt mismatch | `build_tokenizer_pkg` raises `ValueError`; main exits 1 |
| `train_jsonl` missing | `build_mix.main` prints error to stderr, returns 1 |
| Dialogue with literal special token in content | `tokenize_dialogue` returns `(None, "dropped_special_token")` |
| Dialogue too long even at n_ex=1 | `tokenize_dialogue` returns `(None, "dropped_too_long")` |
| No CUDA GPU at training time | `check_gpu_environment` raises `RuntimeError`; `train.main` exits 1 |
| More than 1 visible CUDA device | Same as above |
| Checkpoint path given but missing | Training starts from scratch with a warning |
| NaN/inf in E0 | Mark micro-batch size as failing; halt if no size passes |
| SHA-256 mismatch in Kaggle upload verify | Halt with descriptive error before any model loading |
| In-notebook test failure | `assert` halts notebook execution |

---

## Transformers 5.x Compatibility Notes

Transformers 5.x changed several trainer and tokenizer APIs. The following adaptations are required:

1. **`compute_loss` signature:** In transformers 5.x, `Trainer.compute_loss` receives `num_items_in_batch` as a kwarg rather than via `inputs`. The collator passes it explicitly; `TokenMeanLossTrainer.compute_loss` accepts `**kwargs` and pops `num_items_in_batch`.

2. **`add_special_tokens=False` in `encode()`:** Confirmed unchanged in 5.x — `tok.encode(text, add_special_tokens=False)` works as before.

3. **`AutoTokenizer.from_pretrained` with `local_files_only=True`:** Unchanged; still the correct API.

4. **`apply_chat_template(tokenize=False, add_generation_prompt=True)`:** Unchanged in 5.x. Falls back to `render_chat_template_no_jinja` if jinja2 is absent.

5. **`TrainingArguments.load_best_model_at_end`:** Requires `metric_for_best_model` to be set when using a custom eval metric (macro val loss). In 5.x this remains a string key expected in the eval metrics dict.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Snapshot files are not modified by the package builder

*For any* set of files in `sft/tokenizer_snapshot/`, after running `build_tokenizer_pkg`, the SHA-256 of every file in the snapshot directory is unchanged from before the build.

**Validates: Requirements 1.1**

---

### Property 2: Manifest SHA-256 values match on-disk files

*For any* successfully built `sft/tokenizer_pkg/`, the SHA-256 value recorded in `tokenizer_pkg_manifest.json` for each file equals the SHA-256 computed from the actual file on disk.

**Validates: Requirements 1.4**

---

### Property 3: Tokenizer package build is idempotent

*For any* snapshot directory and output directory, running `build_tokenizer_pkg` twice with the same inputs produces output files with identical SHA-256 values on both runs.

**Validates: Requirements 1.5**

---

### Property 4: Segment-by-segment token ids equal full-string token ids (T1)

*For any* valid dialogue record, the `input_ids` produced by concatenating segment-by-segment tokenization (system header, system content, `<|im_end|>`, newline, user header, user content, `<|im_end|>`, newline, assistant header, assistant content, `<|im_end|>`, newline) are identical to the token ids produced by tokenizing the equivalent full ChatML string in one call.

**Validates: Requirements 2.2**

---

### Property 5: Supervised tokens decode to assistant content plus closing token (T2)

*For any* dialogue record, decoding only the token ids where `labels[i] != MASKED` yields exactly the string `{assistant_content_1}<|im_end|>{assistant_content_2}<|im_end|>...` for all assistant turns in order — with no system text, user text, role headers, or padding.

**Validates: Requirements 2.1, 2.3**

---

### Property 6: No supervised labels outside assistant content (T3)

*For any* dialogue record, every token position in the system block, every user turn (header, content, `<|im_end|>`, trailing `\n`), and every assistant role header (`<|im_start|>assistant\n`) has `labels[i] == MASKED`.

**Validates: Requirements 2.1, 2.4**

---

### Property 7: Every assistant turn contributes exactly one supervised `<|im_end|>` (T4)

*For any* dialogue record with N assistant turns, the number of positions where `labels[i] == IM_END_ID` equals exactly N, with one such position at the end of each assistant turn's content.

**Validates: Requirements 2.5**

---

### Property 8: Truncated items end on a complete assistant turn (T5)

*For any* dialogue record whose full tokenization exceeds `max_len`, the resulting `TokenizedItem` has `truncated=True` and its last supervised token (the last position where `labels[i] != MASKED`) is `<|im_end|>` (token id 151645).

**Validates: Requirements 2.6**

---

### Property 9: Training prefix equals inference prefix byte-for-byte (T6)

*For any* dialogue record and any valid exchange index, the training sequence prefix through the assistant role header of that exchange is byte-for-byte identical to `tok.apply_chat_template(prefix_messages, add_generation_prompt=True)`.

**Validates: Requirements 2.7**

---

### Property 10: Mix builder output is deterministic (idempotent)

*For any* fixed set of input JSONL files and fixed seed, running `build_mix.main` twice produces all output JSONL files with identical SHA-256 values on both runs.

**Validates: Requirements 3.5, 13.4**

---

### Property 11: At least 90% of multi-exchange train dialogues are untruncated

*For any* run of `build_mix` over the full train split at `max_len=2048`, the fraction of eligible multi-exchange dialogues that are kept without truncation is at least 0.90.

**Validates: Requirements 3.8**

---

### Property 12: Upload manifest round-trip integrity

*For any* set of files for which `upload_manifest.json` was generated, running `verify_uploads.py` against the same files exits 0; if any file is modified or missing, `verify_uploads.py` exits 1 with a descriptive error message.

**Validates: Requirements 12.1, 12.2**

---

### Property 13: Gradient-accumulation equivalence (T7)

*For any* set of 4 synthetic tokenized items in fp32 with LoRA dropout set to 0, the maximum absolute difference between parameter gradients computed via micro-batch=1/accumulation=4 and micro-batch=4/accumulation=1 is less than 1e-3.

**Validates: Requirements 5.2**

---

### Property 14: Kill-and-resume loss convergence

*For any* training run interrupted at step K and resumed from the saved checkpoint, the training loss at step K+1 of the resumed run differs from the loss at step K+1 of an identical uninterrupted run by less than 2% relative.

**Validates: Requirements 4.8, 8.3**

---

### Property 15: No dataset text in any report or notebook output

*For any* 200 utterances of at least 40 characters sampled from `integrated_conversations_ner_validated.csv`, none of those utterances appears as a substring in any committed report JSON file under `sft/reports/` or in any notebook cell output JSON.

**Validates: Requirements 13.5, 3.7, 3.4**

---

## Testing Strategy

### Dual Testing Approach

**Unit/example tests** (run on CPU venv, no GPU):
- T1–T6: `sft/tests/test_tokenize_mask.py` — synthetic fixtures only; must pass before any real data processing.
- T7: same file, `@unittest.skipUnless(_TORCH_AVAILABLE, ...)` — skipped on CPU venv.
- Build script import checks: `python -m compileall -q sft` and `python -m unittest discover`.

**Property tests** (enforced by T1–T7 and notebook assertions):
- Properties 4–9 are directly exercised by T1–T6 on synthetic inputs in the test suite and on real data in the local notebook.
- Property 11 is verified in the notebook's E3 assertion cell.
- Properties 1–3 are verified by the build script's own manifest comparison.
- Properties 10, 12–15 are verified by notebook cells and the privacy scan.

### Test configuration

- T1–T6 on synthetic data: at least 100 synthetic records per test class.
- T1 on real data: 1,000 seeded-random train items (seed=42).
- T2–T5 on real data: all items in pilot_mix, overfit_set, and val subsets.
- T7 on real data (Kaggle): full 16-item overfit_set, fp32, dropout=0, max abs diff < 1e-3.

### CPU-only execution guarantee

All test classes T1–T6 and all CPU-stage modules import cleanly on `sft/.venv` (transformers, pandas, datasketch, langid only). The only test that requires torch is T7, which skips automatically via `@unittest.skipUnless(_TORCH_AVAILABLE, ...)`.
