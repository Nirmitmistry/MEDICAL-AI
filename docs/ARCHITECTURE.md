# Medical Chat LLM — Architecture, Data Contract, Training and Evaluation Plan

---

## A. One-page architecture overview

**Goal.** A separately fine-tuned open-weight chat model that answers health questions across several turns using only its weights and the running conversation. No retrieval and no external LLM at inference.

### Seven decisions

1. **Model:** `Qwen/Qwen3-1.7B` (post-trained checkpoint), Apache-2.0, stored locally.
2. **Method:** LoRA SFT (r=16, all linear layers) with `transformers` Trainer + PEFT on one GPU. No TRL, no QLoRA by default, no full fine-tune.
3. **Data:** the validated CSV stays read-only. A new builder reconstructs complete dialogues as JSONL with dialogue-aware cleaning. Existing SFT exports (including the ≈90-line file) and `_clean.csv` are not inputs.
4. **Format and loss:** ChatML with one fixed system prompt and one dialogue per sequence. Loss is on assistant tokens only (content + `<|im_end|>`), for every assistant turn.
5. **Splits:** by `group_id` (exact + near-duplicate clusters, so a `dialogue_id` never spans splits). Quota-stratified so val/test contain real multi-exchange dialogues.
6. **Behavior supplement:** ≈240 hand-reviewed dialogues (identity, escalation, uncertainty, context carry-over), because the corpus cannot teach these.
7. **Evaluation and inference:** held-out NLL, history ablation, ≈150 scripted probes, blind human rubric, gates frozen before final training. Inference is an offline `ChatSession` with the identical template, whole-exchange history trimming, and a deterministic red-flag banner.

### What the corpus supports (documented figures)

- MedDialog is 224,328 turns over 112,164 dialogues, exactly 2 turns each, so it is single-exchange. All multi-exchange signal comes from HealthChat: 11,092 dialogues (9.0% of dialogues, 18.6% of turns). Its assistant turns are LLM-written and not clinically validated.
- **Sufficient for:** response style, general health-information coverage, held-out-loss measurement.
- **Insufficient or unsuitable for:** calibrated uncertainty, emergency escalation, refusals, multi-turn constraint carry-over, and any claim of medical correctness. The NER columns are not supervision (documented mislabeling) and are not used.

**Compute.** The audit machine (CPU-only, ≈5.8 GiB RAM, no ML packages) can build data and run the demo. Training needs a ≥15 GB GPU (see E8).

---

## B. Component and data-flow diagram

```mermaid
flowchart TD
    CSV["integrated_conversations_ner_validated.csv<br/>read-only source of record"]
    G1{"Gate G1<br/>measured counts match<br/>documented counts?"}
    STOP["Stop and report mismatch"]

    subgraph CPU["CPU stage - no GPU needed"]
        B1["1. Dialogue builder<br/>group by dialogue_id, order by turn_id,<br/>validate, trim at first defect"]
        B2["2. Cleaner and filters<br/>strip templates, English only,<br/>length and format filters"]
        BEH["Authored behavior set and eval probes<br/>separate scenario families"]
        B3["3. Group and split<br/>exact + MinHash groups,<br/>quota strata, split by group"]
        JSONL[("dialogues train / val / test JSONL<br/>splits manifest + prep report")]
        B4["4. Tokenize and mask<br/>ChatML, assistant-only labels,<br/>prefix-parity tests"]
        TOK[("tokenized train and val sets")]
    end

    subgraph GPU["GPU stage - single GPU"]
        B5["5. LoRA SFT, Qwen3-1.7B<br/>HF Trainer + PEFT,<br/>resumable checkpoints"]
        ADP[("adapter checkpoints<br/>+ run manifest")]
    end

    subgraph EVAL["Evaluation"]
        B6["6. Eval harness<br/>held-out NLL, history ablation,<br/>probes, human rubric"]
        REP[("eval report + frozen gates")]
    end

    subgraph INF["Offline inference"]
        B7["7. Merge adapter<br/>local safetensors"]
        B8["8. ChatSession<br/>same template and system prompt,<br/>history window"]
        BAN["Red-flag banner<br/>regex, not a model"]
        CLI["CLI chat"]
    end

    CSV --> G1
    G1 -- pass --> B1
    G1 -- mismatch --> STOP
    B1 --> B2 --> B3
    BEH --> B3
    B3 --> JSONL --> B4 --> TOK --> B5 --> ADP
    ADP --> B6 --> REP
    ADP --> B7 --> B8 --> CLI
    BAN --> B8
```

---

## C. Model selection

| Candidate | License | Size / context | Verdict |
|---|---|---|---|
| **Qwen3-1.7B** | Apache-2.0 | 1.7B total (1.4B non-embedding), 28 layers, grouped-query attention 16/8, 32,768 context | **Default.** Plain dense attention and mature tooling. fp16 LoRA fits 16 GB with margin (fit table below). Merged bf16 is ≈3.4 GB and fits the audit machine. |
| Qwen3-4B | Apache-2.0 (Qwen3 family; confirm at download) | ≈4B (≈3.6B non-embedding), 32k native; from memory, confirm at download | Not default. fp16 LoRA is ≈12.3 GB at micro-batch 1 and over 16 GB at 2, so QLoRA would be required. Compute per token is ≈2.5×. bf16 weights (≈8 GB) do not fit the audit machine. Deferred upgrade, outside this architecture. |
| MiniCPM5-2B | Apache-2.0 | 2.52B total (1.98B non-embedding), 131,072 context, released 7 Sep 2026 | Not default. Standard Llama architecture is a plus. But it is under 7 weeks old, benchmark claims are vendor-reported, and bf16 weights (≈5.0 GB) leave ≈1.2 GB headroom on the audit machine. Revisit once the pipeline works. |
| Qwen3.5-2B/4B, Gemma 4 E2B/E4B | Qwen3.5: Apache-2.0; Gemma 4: Apache-2.0 | Qwen3.5 small series: released 2 Mar 2026, hybrid gated-DeltaNet attention; Gemma 4: released 2 Apr 2026, multimodal | **Excluded.** Qwen3.5's fast path needs flash-linear-attention and causal-conv1d compiled against the local torch/CUDA, otherwise it falls back to a slow PyTorch path, and transformers ≥5.9. Gemma 4's "effective" sizes describe active parameters, not total footprint. Both conflict with the offline, minimal-dependency, unknown-GPU constraints. |

**Llama 3.2** is excluded because its Acceptable Use Policy lists unlicensed practice of medical/health professions as prohibited, and distributed derivative models must carry "Llama" in their name. That is avoidable compliance ambiguity for a medical chatbot (not legal advice). Models of 7B and above exceed the T4 budget and the CPU demo.

### Recommendation

- Use `Qwen/Qwen3-1.7B` (**not** `-Base`). It starts instruction-following.
- Qwen3's default template toggles thinking mode. We replace it with our own thinking-free ChatML template, and eval gate V6 asserts zero `<think>` leakage.
- **Fallback**, only if no ≥8 GB GPU is obtainable: Qwen3-0.6B with identical code. It is a pipeline proof, not a quality result.

### Fit check

Arithmetic from config, 2,048-token sequences, assuming ≈10 bytes per vocabulary entry per token for logits + loss, **not measured**.

| Component | 1.7B LoRA fp16 (default) | 1.7B QLoRA | 4B LoRA fp16 | 4B QLoRA |
|---|---|---|---|---|
| Weights | 3.4 GB | 1.4 GB | 8.0 GB | 2.7 GB |
| LoRA + grads + Adam (fp32) | 0.3 | 0.3 | 0.5 | 0.5 |
| Checkpointed activations / seq | 0.5 | 0.5 | 0.7 | 0.7 |
| Logits + loss / seq (V=151,936) | 3.1 | 3.1 | 3.1 | 3.1 |
| **Total, micro-batch 1** | **7.3 GB** | **5.3 GB** | **12.3 GB** | **7.0 GB** |
| **Total, micro-batch 2** | **10.9 GB** | **8.9 GB** | **16.1 GB** | **10.8 GB** |

- Add ≈0.4 GB CUDA context plus allocator slack. E0 replaces these estimates with measurements.
- r=16 on all linear layers is ≈17M trainable parameters (about 1%), computed from config. E0 prints the exact count.
- Inference KV cache is ≈0.11 MB per token (≈0.23 GB at 2,048 tokens).

---

## D. Data contract

### D1. Source and gate G1

- Read `integrated_conversations_ner_validated.csv` **read-only** with `dtype=str` for both IDs.
- G1 recomputes every documented figure and stops on any unexplained mismatch.
  - **Counts:** 275,639 rows; 123,256 dialogues; 137,880 assistant and 137,759 user turns; per-source rows and dialogues; 198 null utterances; 46,889 rows with empty entities.
  - **Structure:** `dialogue_id` maps 1:1 to `(source_dataset, original_id)`; `(dialogue_id, turn_id)` is unique; every MedDialog dialogue has exactly 2 turns.
- Existing SFT exports and `integrated_conversations_clean.csv` are **quarantined** (neither consumed nor deleted). The clean file's row-level drops can break dialogue continuity (audit P1). We rebuild with dialogue-level operations.

### D2. Canonical record (JSONL, one per dialogue)

| Field | Content |
|---|---|
| `dialogue_id`, `source`, `original_id` | carried from CSV; `source` is one of the 3 values |
| `group_id`, `split` | assigned in step 3 |
| `messages[]` | `{turn_id, role, content}`; roles `user` / `assistant`; `turn_id` is the CSV position |
| `n_exchanges` | number of (user, assistant) pairs |
| `meta` | `templates_stripped`, `truncated_exchanges`, `n_tokens` |

Entity columns are not carried; they remain in the CSV, joinable on `(dialogue_id, turn_id)`.

### D3. Build rules (in order)

- **R1** Group by `dialogue_id`, sort by `turn_id`.
- **R2** *Valid* means: `turn_id` contiguous from 0, roles alternate starting with user, and every utterance is non-null and non-empty after strip.
- **R3** Never renumber or merge. **Cut at the first defect** and keep the longest prefix ending on an assistant turn. Drop the dialogue if no full exchange remains. This covers the 198 nulls, gaps from the row-level exact dedup, and same-role runs.
- **R4** Template stripping, inline, never deleting a turn.
  - Sign-off and greeting patterns come from frequency mining of first and last sentences of MedDialog assistant turns, reviewed by hand, plus `chatdoctori` / "Chat Doctor".
  - Residual first-person credential claims ("I am Dr…", "as your doctor") drop the dialogue. If this removes >25% of MedDialog, stop and review the regex.
  - Minimum length after stripping is 5 user and 20 assistant tokens (final thresholds from E1).
- **R5** Filters: English only (langid on user turns); no code fences or markdown tables; first exchange must fit `max_len`.
- **R6** Normalization: Unicode NFC, strip, collapse runs of blank lines. `NAME_k` placeholders are left as is.
- **R7** Conservation: input dialogues = kept + Σ dropped-by-reason, per source, in `prep_report.json`. P1 does not close if the sum doesn't.

### D4. Serialization (our ChatML template, saved in the tokenizer, hash pinned)

```text
<|im_start|>system\n{SYSTEM_PROMPT_V1}<|im_end|>\n
<|im_start|>user\n{content}<|im_end|>\n
<|im_start|>assistant\n{content}<|im_end|>\n
```

- No `<think>` blocks anywhere.
- EOS is `<|im_end|>`, and pad is **not** EOS.
- `generation_config.eos_token_id` is `[<|im_end|>, <|endoftext|>]`.

**SYSTEM_PROMPT_V1** (≈90 tokens, hash-pinned):

> You are a general health-information assistant. You are an AI, not a doctor: you cannot examine the user, diagnose, or prescribe. Use everything the user has told you in this conversation. If key details are missing, ask one or two focused questions. State uncertainty plainly. If there are signs of an emergency, tell the user to contact local emergency services now. Recommend seeing a qualified clinician for severe, persistent, or worsening symptoms, and for any prescription medicine or dosing decision.

### D5. One complete multi-turn item

*Synthetic illustration, not from the dataset, not medical advice.*

```json
{
  "dialogue_id": "e4a1f0c27b5d4e0f9a31c6b8d2f07a15",
  "source": "HealthChat-WildChat",
  "original_id": "illustrative",
  "group_id": "g_000123",
  "split": "train",
  "n_exchanges": 3,
  "messages": [
    {"turn_id": 0, "role": "user", "content": "I've had a throbbing headache on one side for two days. Ibuprofen helps a bit but it comes back. Should I be worried?"},
    {"turn_id": 1, "role": "assistant", "content": "A one-sided throbbing headache that eases with ibuprofen and then returns is often migraine-type or tension-type, but I can't diagnose it from a message. A few details would help: any fever, stiff neck, vision changes, weakness, or confusion? Is this the worst headache of your life, or different from your usual ones? If any of those are present, please get urgent medical care."},
    {"turn_id": 2, "role": "user", "content": "No fever or vision changes, and it isn't the worst I've had. I'm 29, I sleep about 5 hours a night, and I'm allergic to penicillin."},
    {"turn_id": 3, "role": "assistant", "content": "Thanks, that helps. Without red-flag symptoms, short sleep is a common trigger for recurring headaches, along with dehydration, skipped meals, and long screen time. A simple diary of timing, sleep, food, and stress can show a pattern. Your penicillin allergy doesn't affect ibuprofen, but mention it to any clinician or pharmacist. If the headaches keep coming back over the next week or two, book a routine visit."},
    {"turn_id": 4, "role": "user", "content": "Can I just take it every day if it keeps coming back?"},
    {"turn_id": 5, "role": "assistant", "content": "I'd avoid that. Using ibuprofen most days can irritate the stomach and can itself cause rebound headaches. With only about 5 hours of sleep, fixing sleep is the better first step, and if you find yourself needing a pain reliever on more than 2-3 days a week, see a clinician. Follow the label for dose and spacing, and ask a pharmacist if you're unsure."}
  ],
  "meta": {"templates_stripped": false, "truncated_exchanges": 0, "n_tokens": 0}
}
```

**Serialized, with label mask:**

```text
<|im_start|>system\n{SYSTEM_PROMPT_V1}<|im_end|>\n     labels = -100
<|im_start|>user\n{u1}<|im_end|>\n                      labels = -100
<|im_start|>assistant\n                                 labels = -100
{a1}<|im_end|>                                          labels = token ids (supervised)
\n                                                      labels = -100
...same pattern for u2/a2 and u3/a3
```

Turn 5 only answers well if the model uses ibuprofen (turn 0) and the sleep detail (turn 2). That is what V2/V3 test.

---

## E. Training, evaluation, inference design

### E1. Training method

- **LoRA, not QLoRA.** At 1.7B the fp16 base fits with margin, so quantization adds noise and a `bitsandbytes` dependency for no gain. QLoRA (NF4) is the rung for 8–12 GB GPUs only. Full fine-tuning needs roughly 14–16 bytes per parameter with AdamW (≈24 GB+ here), which rules out a T4.
- **Plain Trainer, not TRL.**
  - TRL's assistant-only loss depends on `{% generation %}` markers in the chat template, which very few models ship natively.
  - Current TRL docs steer toward prompt–completion data, which supervises one completion per example, the wrong shape for multi-turn single-sequence supervision.
  - Assistant masks can also be lost to truncation or ignored by the collator.
  - Our own masking is ≈40 lines with unit tests and doesn't change when libraries do.
- **Version pinning.** The transformers 5.x line is current, so 4.x tutorials may use renamed arguments. Freeze exact versions after the E0 smoke test, with no mid-project upgrades.
- **Dependencies (complete):** `torch`, `transformers`, `peft`, `accelerate`, `pandas`, `datasketch`, `langid`. Add `bitsandbytes` only on the QLoRA rung.

### E2. Objective and masking

- **Supervised:** for every assistant turn, content tokens plus the closing `<|im_end|>`.
- **Masked (-100):** system block, user turns, role headers, the `\n` after each `<|im_end|>`, and padding.
- **Construction:** tokenize segment by segment (header, content, `<|im_end|>`, newline), concatenate ids, and copy ids into labels only for supervised segments. No offset mapping and no template markers.
- Loss is token-mean over supervised tokens across the accumulated batch.

**Tests that must stay green:**

| Test | Assertion |
|---|---|
| T1 | Concatenated ids equal full-string tokenization on 1,000 random items. |
| T2 | Decoded supervised tokens equal the assistant contents plus `<|im_end|>`, in order. |
| T3 | Zero supervised tokens in system, user, headers or padding. |
| T4 | Every assistant turn ends in a supervised `<|im_end|>`, and the pad id differs from the `<|im_end|>` id. |
| T5 | Truncated items still end on an assistant `<|im_end|>`. |
| T6 | Train/inference parity. For every prefix ending in a user turn, `apply_chat_template(add_generation_prompt=True)` equals the training string through the assistant header, byte for byte. |
| T7 | Gradient-accumulation equivalence. Micro-batch 1 × accumulation 4 equals micro-batch 4 × accumulation 1 within 1e-3. |

### E3. Sequence handling

| Case | Rule |
|---|---|
| 1 exchange (all MedDialog) | Keep as is. |
| Multi-exchange that fits `max_len` | Keep whole; all assistant turns supervised. |
| Longer than `max_len` | Drop trailing exchanges at a turn boundary, keeping the longest prefix that ends on an assistant turn and fits. Never cut inside a message. Drop the dialogue if exchange 1 alone doesn't fit. Count both. |
| Missing, null or out-of-order turn; same-role run; leading assistant turn | R3 (cut at first defect). |
| Too short after stripping | Cut per R3. |
| Packing | Not used. Length-grouped batches with dynamic padding. |
| `max_len` | 2,048 default. E1 sets it so ≥90% of eligible multi-exchange dialogues are untruncated; hard cap 3,072 from E0. |

### E4. Training configuration

| Parameter | Value | Status |
|---|---|---|
| Base | `Qwen/Qwen3-1.7B`, pinned revision, local | Fixed |
| Precision | bf16 if the GPU supports it, else fp16; LoRA weights fp32 | Fixed rule. T4 has no native bf16, so fp16 there |
| Attention | PyTorch SDPA (FlashAttention-2 needs Ampere or newer) | Fixed |
| LoRA | r=16, α=32, dropout 0.05, all linear layers, not lm_head or embeddings | Fixed (r=64 only via E3 remedy) |
| Optimizer | AdamW, weight decay 0, clip 1.0, cosine, 3% warmup | Fixed |
| Learning rate | 1e-4 | E2 pilot |
| Epochs | 1 over a budget-sized sample | Fixed |
| Effective batch | 32 dialogues | Fixed; micro-batch from E0 |
| `max_len` | 2,048 | E1 |
| Train size | ≈25k dialogues indicative (≈800 steps), set so the run is ≤5 GPU-h at measured tok/s | E0 |
| Mix by tokens | 50% MedDialog / 45% HealthChat / 5% behavior set (×3 upsample) | Fixed start; adjusted only by E3 rules |
| Eval / checkpoints | In-run eval every 100 steps on a fixed 400-dialogue val subset (200 + 200); save every 100 steps with optimizer + RNG; keep last 3 + best by macro val loss (mean of the two source-group NLLs) | Fixed |
| Seed | 42 | Fixed |

Training on ≈25k of the eligible pool is a compute-budget choice, not a limit of the design. The rest of the pool is reserve.

### E5. Short experiments

**E0 (GPU, ≤0.5 h).**
- Run 50 steps at micro-batch 1/2/4 at `max_len`.
- Pick the largest micro-batch with peak <90% VRAM and no NaN/inf in loss or grad-norm.
- On NaN, the usual remedies are a lower learning rate, bf16 instead of fp16, gradient clipping and shorter sequences. We halve the LR first, then move to a bf16 GPU.
- tok/s sets the training budget.

**E1 (CPU).** Length, template, language and persona-claim histograms → `max_len`, regex list, thresholds.

**E2 (GPU, 3 × ≈0.75 h).** 150-step pilots at LR {5e-5, 1e-4, 2e-4} on identical data. Pick the lowest macro val loss with no spikes.

**E3 (conditional, one rerun max).** Run only if a gate fails, with a pre-registered remedy:

| Failure | Remedy |
|---|---|
| Multi-turn gates fail | Raise the HealthChat share by 10 points |
| Safety gates fail | Double the behavior-set weight |
| Dosing probes fail | Add a dose-and-frequency pattern filter on MedDialog |
| Capacity-limited | r=64 |

### E6. Evaluation suite

Greedy decoding; each system also run on base Qwen3-1.7B as baseline.

| ID | Question | Method | Metric / gate |
|---|---|---|---|
| V1 | Did SFT learn the target distribution? | Test split, assistant tokens only; per source and first vs later turns | NLL with bootstrap CI. Not evidence of correctness. |
| V2 | Does it use history? | HealthChat multi-exchange test items: NLL of assistant turn ≥2 with full history vs earlier turns removed vs another dialogue's history | Gap (full vs removed) has CI above 0 and is ≥ base's gap |
| V3 | Constraint carry-over in generation | 40 scripted 3-exchange probes; a fact (allergy, age, current medicine, duration) planted in exchange 1, used in exchange 3; automatic string/regex check | Pass rate ≥ base + 15 points (initial floor) |
| V4 | Calibrated uncertainty | 40 probes (insufficient info, "diagnose me", "exact dose", prognosis): asks a clarifying question or states uncertainty, no definitive diagnosis wording, no credential claims | Floor frozen vs base; credential claims = 0 |
| V5 | Safety | 60 probes (emergency red flags, self-harm, child/pregnancy/prescription dosing, controlled-drug acquisition, overdose); human pass/fail rubric | Escalation ≥90% model-only and 100% with banner; 0 harmful specifics in refusal categories |
| V6 | Format and leakage | All generations | EOS-terminated ≥98%; template/sign-off/chatdoctori ≤0.5%; `<think>` = 0; hallucinated next-turn headers = 0 |
| V7 | Preferred over base? | 50 single + 20 multi-turn test prompts, blind pairwise (base / fine-tuned / reference) by ≥2 raters | Win/tie/loss counts and rater agreement |

**Rules:**
- Probes are split 40/60 dev/final. The final 60% and the test split are each used once.
- Probe scenario families are disjoint from the behavior training set.
- Thresholds are written to `eval/gates.yaml` after the base-model baseline and before the final fine-tune. They can be tightened afterwards, never loosened.
- No external LLM judge.
- A clinician rating the V5 probes is preferred. Without one, the report states that limit and claims behavioral checks only.

### E7. Inference

- **Artifacts:** LoRA adapter, tokenizer with pinned template, `inference_config.json` (system-prompt hash, decode parameters), and an optional merged bf16 safetensors export for CPU.
- **Load:** `from_pretrained(local_dir, local_files_only=True)` with `HF_HUB_OFFLINE=1`. dtype is fp16/bf16 on CUDA and bf16 on CPU. A template-hash assertion runs at load.
- **Turn loop:** append the user message, build the prompt (system + window + generation prompt) with the same template, generate, strip, append the reply.
- **Window:** prompt budget is 2,048 − 512 = 1,536 tokens. Drop the oldest whole exchanges until it fits. The system prompt is never dropped. If the latest user message alone exceeds the budget, ask the user to shorten it. No summarization.
- **Decoding:** stop at `<|im_end|>`, `max_new_tokens` 512. Demo defaults are temperature 0.4, top-p 0.9, repetition penalty 1.05, with no tuning budget.
- **Red-flag banner:** a fixed message prepended when the user text matches a lexicon (chest pain with breathlessness, stroke signs, suicidal intent, overdose, severe bleeding, anaphylaxis). It is not model output and is evaluated separately in V5.
- **Interface:** CLI. Anything else must wrap the same `ChatSession`.
- **CPU feasibility:** merged bf16 ≈3.4 GB + KV cache fits 5.8 GiB. Speed is unmeasured. Accept if ≥2 tok/s is measured, otherwise demo on a GPU session. GGUF/llama.cpp packaging only if that fails.

### E8. Environment and cost

| GPU available | Path |
|---|---|
| ≥15 GB, Turing (Kaggle/Colab T4) | LoRA fp16, micro-batch 1–2. Checkpoint-and-resume is mandatory. |
| ≥15 GB, Ampere or newer (rented 3090/4090/L4/A100) | LoRA bf16, same config. Fewer hours; removes the fp16 NaN risk. |
| 8–12 GB | QLoRA NF4 (≈5.3 GB at micro-batch 1 per the fit table), same pipeline. |
| None | Qwen3-0.6B, data pipeline plus a tiny run only. Pipeline proof, not a quality result. |

- **Free tiers.** Kaggle offers T4/P100-class GPUs with 12-hour sessions and a floating weekly quota (≈30 h is commonly reported but undocumented). Colab's free tier publishes no quota, doesn't name the GPU, and may interrupt idle runtimes. Treat both as unreliable, so resumable training is required. The derived JSONL must be uploaded privately (MedDialog terms), and model weights and wheels pre-staged and checksummed.
- **Paid fallback.** On 23 Oct 2026 the cheapest in-stock on-demand prices were RTX 3090 $0.29/h, L4 $0.59/h and RTX 4090 $0.89/h, with no T4 in stock. Stock is thin, so check at booking.
- **GPU-hour plan (T4-equivalent).** E0 0.5 + E2 2.25 + final ≤5 + eval ≈1.5 + one contingency rerun ≤5 ≈ **14 h**. That fits one Kaggle quota week if the quota holds. On a rented 3090/4090 it should take about half the hours, roughly US$2–7 total, an estimate that depends on the real speedup.
- **Throughput.** ≈10 GFLOPs/token puts the T4 planning range at 0.8–1.5k tok/s, a FLOP-derived guess that E0 replaces.

---

## F. Risks, constraints, non-goals

| Risk | Mitigation / gate |
|---|---|
| Multi-turn data is thin and LLM-generated (≤11,092 dialogues) | V2/V3 gates; behavior set carries carry-over; E3 raises the HealthChat share. If val/test can't reach 300 multi-exchange each, declare the multi-turn objective unsupported by this corpus. |
| Doctor-persona, prescriptive MedDialog style conflicts with the system prompt | R4 credential filter, behavior set, V4/V5, conditional dose-pattern filter. |
| Small-model hallucination | Uncertainty probes; no correctness claims. |
| Licensing and terms: MedDialog has no open license; LMSYS/WildChat have their own terms; HealthChat assistant text comes from third-party LLMs | Data and weights stay private, research use only. Legal review before any release (not legal advice). |
| fp16 instability on T4 | E0 NaN probe; bf16 GPU fallback. |
| Framework churn | Pinned environment; `run_manifest.json`. |
| Free-tier interruption | Resumable training; kill-and-resume test. |
| Leakage across splits | Group split plus MinHash audit. |
| Small-n evaluation, non-clinician raters | Report CIs; limit claims. |
| PII and `NAME_k` remnants, memorization | No release; no extra scrubbing claimed. |

**Non-goals:** RAG, retrieval or vector DB; any external LLM at inference or as judge; fine-tuning Claude; RLHF/DPO; entity-aware training; multilingual or multimodal support; clinical validation; public release; long context beyond 2k; packing; multi-GPU; web UI or GGUF serving unless a measured need appears.

---

## G. Phased implementation plan

All new code lives in a new `sft/` tree. Notebooks 04–06 and existing training code are untouched.

### P0: Environment and access *(no dependencies)*

- **Deliverables:** pinned environment files, local model snapshot, GPU path chosen from the E8 table.
- **Accept:** `pip freeze` saved. Model loads with `local_files_only`. GPU probe prints device, VRAM, compute capability and bf16 support.

### P1: Dialogue builder *(needs P0 CPU environment)*

- **Deliverables:** `dialogues_all.jsonl`, `prep_report.json`, E1 histograms.
- **Accept:**
  - G1 passes, or the discrepancy is explained and approved.
  - Schema valid on 100% of records; contiguity and alternation 100%.
  - R7 conservation closes per source.
  - Residual template/`chatdoctori` ≤0.1% of assistant turns.
  - All MedDialog items are single-exchange.
  - Drop counts reported by reason.

### P2: Splits, behavior set, probes, baseline *(needs P1)*

- **Deliverables:** split manifest, ≈240 behavior dialogues, ≈150 probes, baseline eval of untuned Qwen3-1.7B, frozen `eval/gates.yaml`.
- **Accept:**
  - 0 `group_id` overlap across splits and 0 exact normalized first-turn overlap.
  - MinHash audit (Jaccard 0.8): 0 val/test dialogues matching any train dialogue.
  - Quotas met: val and test each have 500 MedDialog and 500 HealthChat (≥400 multi-exchange). If infeasible, shrink to ≥300 multi-exchange each or declare insufficiency.
  - Manifest hash reproducible on re-run.
  - Behavior and probe families disjoint.
  - Gates committed before any training.

### P3: Tokenize and mask, E0/E2 *(needs P2 and a GPU)*

- **Deliverables:** tokenizer package, tokenized sets, E0 report, E2 pilot results.
- **Accept:** T1–T7 pass. E0: peak <90% VRAM, no NaN/inf, tok/s recorded. Kill-and-resume matches the uninterrupted run's next-step loss within 2%. Overfit test: 16 dialogues reach ≤30% of initial loss within 150 steps. LR chosen.

### P4: Final training *(needs P3)*

- **Deliverables:** adapter checkpoints, `run_manifest.json`.
- **Accept:** run completes (resumes handled). Final val ≤ min val + 0.05. The manifest holds hashes for data, splits, template, system prompt, tokenizer, versions, config and GPU.

### P5: Evaluation, inference, report *(needs P4)*

- **Deliverables:** `eval_report.json`, rater sheets, `ChatSession` + CLI, short report.
- **Accept:**
  - V1–V7 pass, or failures are documented with the E3 remedy executed (≤1 rerun).
  - A scripted 10-turn CLI conversation shows history use.
  - Template-hash assertion and offline-load test pass.
  - CPU memory and speed recorded on the 5.8 GiB-class machine.
  - Test split used exactly once.

---

## H. Decisions that must not change silently

Any change needs a written amendment and a re-run of the dependent gates.

1. **Base model and pinned revision:** Qwen3-1.7B.
2. **Method and stack:** LoRA via Trainer + PEFT, single GPU. No TRL, Unsloth, Liger, DeepSpeed or QLoRA unless E0 proves a need.
3. **Own ChatML template** with no `<think>`; `SYSTEM_PROMPT_V1` text and hash; EOS `<|im_end|>`; pad ≠ EOS.
4. **Loss mask:** assistant content + `<|im_end|>` only, all assistant turns; T1–T7 stay green.
5. **The validated CSV is the read-only source.** Existing SFT exports and `_clean.csv` are not inputs.
6. **Dialogue integrity:** no row-level deletion, renumbering or merging; cut-at-first-defect rule.
7. **Split unit is `group_id`;** quotas; manifest hash; test used once; probes split dev/final.
8. **Truncation at turn boundaries only;** no packing.
9. **Entity columns are not used** in training or serialization.
10. **Behavior set and probes keep disjoint families;** changing either re-freezes the gates.
11. **Gates are frozen before the final run** and never loosened.
12. **Inference uses the same template, system prompt and window rule.** No retrieval, external LLM or summarization. The banner stays deterministic.
13. **Environment pinned;** no mid-project upgrades.
14. **Data and weights stay private.**

---

## Open inputs needed before the implementation backlog

1. Which GPU path will be used: Kaggle/Colab, rental, or a lab machine.
2. Whether a clinician is available to rate the 60 safety probes.
