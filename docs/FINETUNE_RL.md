# FINE_TUNING_RL_STRATEGY.md — SFT → preference alignment

**Pipeline:** instruct checkpoint → LoRA SFT → (optional best-of-N SFT) → DPO (+ NLL term) → optional GRPO. Each stage must pass its gate (PROJECT_PLAN §3) before the next starts.
**Version pinning:** TRL/transformers/accelerate APIs move quickly (e.g. ORPO moved to `trl.experimental`). Pin versions and re-check config fields against your pinned release.

## 1. Base model

| Candidate | Licence | Notes | Verdict |
|---|---|---|---|
| `Qwen/Qwen3-8B` | Apache-2.0 (checked) | ChatML-style template; TRL auto-patches generation markers for assistant-only loss; hybrid thinking (use non-thinking for v1) | **Default** |
| `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 (check card) | Closest to your list; simple template | Fallback |
| `mistralai/Mistral-7B-Instruct-v0.3` | Apache-2.0 (check card) | Older; weaker multilingual | Fallback |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama licence + AUP | AUP bars unlicensed practice of medical/health professions and presenting outputs as human; derivatives named "Llama…" | Only after legal sign-off |

**Bake-off (3 days):** for each candidate run (a) zero-shot dev suite (500 MedQA-dev, 300 HealthBench-subset, tier-0 triage dev set, persona test), (b) 2k-dialogue LoRA SFT, (c) compare on composite (accuracy, safety, cost/latency). Pipeline is model-agnostic because rendering goes through `apply_chat_template`.
**Start from the instruct checkpoint:** a base model + 275k noisy turns would inherit MedDialog's style flaws and lose instruction-following and baseline safety.

## 2. Supervised fine-tuning

### 2.1 Method choice
| Method | Memory (8B, ≈) | Use when |
|---|---|---|
| Full FT | ~16 B/param (bf16 weights 2 + grads 2 + fp32 master 4 + Adam 8) ≈ 128 GB → ZeRO-3 over ≥ 4×80 GB | Ablation A5, or if LoRA is ≥ 2 pts worse |
| **bf16 LoRA r=64 (default)** | 16 GB weights + ~3 GB adapter/optimiser + activations → fits one 40–80 GB GPU | Default. r=64 all-linear ≈ 170M trainable (~2%) |
| QLoRA (NF4) | ~5 GB weights | < 40 GB VRAM; ~30–40% slower, small quality cost |

LoRA params ≈ Σ over target modules of `r·(d_in + d_out)`.

### 2.2 Hyperparameters
| Param | LoRA (default) | QLoRA | Full FT |
|---|---|---|---|
| LR / schedule | 1e-4, cosine, 3% warmup | 2e-4 | 1e-5 |
| Epochs | 2 (val-loss for overfit watch only) | 2 | 2 |
| Global batch | 64 packed seqs × 4,096 tok ≈ 0.26M tok/step (≈ 500 steps total; recompute from measured tokens) | same | 128 |
| LoRA | r=64, α=128, dropout 0.05, `all-linear` | r=64 | — |
| Optim | AdamW fused, wd 0.0, clip 1.0 | paged AdamW 8-bit | AdamW, wd 0.01 |
| Precision/attn | bf16, FlashAttention-2, grad-checkpointing | — | — |
| Max length | 4,096 + packing | | |

### 2.3 Training script
```python
# src/medchat/train/sft.py   (launch with accelerate, §2.5)
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("json", data_files={"train": "data/sft/v1/train.jsonl", "validation": "data/sft/v1/val.jsonl"})
ds = ds.select_columns(["messages"])
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,          # `dtype=` on transformers ≥ 4.56
                                             attn_implementation="flash_attention_2")     # "sdpa" if flash-attn unavailable
args = SFTConfig(
    output_dir="runs/sft-qwen3-8b-lora", seed=42,
    num_train_epochs=2, per_device_train_batch_size=2, gradient_accumulation_steps=4,    # ×8 GPUs = 64 packed seqs
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0, max_grad_norm=1.0,
    bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=4096, packing=True,                 # packing needs FlashAttention to avoid cross-example attention
    assistant_only_loss=True,                      # needs {% generation %} template; TRL patches some (incl. Qwen3). If your
                                                   # TRL rejects it with packing → packing=False or use masking.encode_example
    eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50, save_total_limit=4,
    logging_steps=10, report_to="wandb", dataset_num_proc=8,
)
peft_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear",
                      bias="none", task_type="CAUSAL_LM")
SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
           processing_class=tok, peft_config=peft_cfg).train()
```
Choose the checkpoint by the Tier-0 eval (EVAL §1, §5), **not** val loss alone.

### 2.4 DeepSpeed / FSDP
`configs/deepspeed/zero2.json` (LoRA default; fastest):
```json
{"bf16": {"enabled": true},
 "zero_optimization": {"stage": 2, "overlap_comm": true, "contiguous_gradients": true,
                       "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8},
 "gradient_accumulation_steps": "auto", "gradient_clipping": "auto",
 "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto", "steps_per_print": 50}
```
`configs/deepspeed/zero3.json` (full FT, GRPO with large models):
```json
{"bf16": {"enabled": true},
 "zero_optimization": {"stage": 3, "overlap_comm": true, "contiguous_gradients": true, "sub_group_size": 1e9,
   "reduce_bucket_size": "auto", "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto",
   "stage3_max_live_parameters": 1e9, "stage3_max_reuse_distance": 1e9, "stage3_gather_16bit_weights_on_model_save": true},
 "gradient_accumulation_steps": "auto", "gradient_clipping": "auto",
 "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto"}
```
`configs/accelerate/ds_zero2.yaml` (swap the JSON for zero3):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config: {deepspeed_config_file: configs/deepspeed/zero2.json, zero3_init_flag: false}
mixed_precision: bf16
num_machines: 1
num_processes: 8
main_training_function: main
use_cpu: false
```
`configs/accelerate/fsdp2.yaml` (alternative; **FSDP2 keys differ across accelerate versions. Generate with `accelerate config` and diff**):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
mixed_precision: bf16
num_machines: 1
num_processes: 8
```

### 2.5 Launch, merge, export
```bash
accelerate launch --config_file configs/accelerate/ds_zero2.yaml src/medchat/train/sft.py
python -m medchat.train.merge --base Qwen/Qwen3-8B --adapter runs/sft-qwen3-8b-lora/checkpoint-XXX \
       --out artifacts/medchat-sft-merged      # PeftModel.merge_and_unload(); record base revision hash
```

### 2.6 Monitoring & ablations
- Per-source val loss (MedDialog / HealthChat). Every 100 steps, offline eval of 200 held-out prompts: persona-claim rate, child-dose leak, tier-0 escalation rate, judge win-rate. **Stop** if any safety metric regresses on 2 consecutive evals.
- Forgetting guard: MedQA/MMLU-clinical subset every 250 steps.

| ID | Ablation | Question |
|---|---|---|
| A1 | raw MedDialog vs R2A | does rewriting remove persona/dose failures without hurting accuracy? |
| A2 | with / without `<clinical_context>` | does conditioning help, and is it robust to NER noise? |
| A3 | entity-span weighting λ ∈ {0, 0.5} | does it raise entity recall without density gaming? |
| A4 | mixture weights (§DATA 11) | multi-turn gain vs knowledge regression |
| A5 | LoRA r ∈ {16, 64}, full FT | is there a ≥ 2-pt gap? |
| A6 | instruct vs base start | forgetting vs style |

## 3. Preference alignment (RLHF / RLAIF)

### 3.1 Why pairs are built this way (trust tiers)
- **Tier A (MedDialog):** real clinician content, poor style/safety → content reference for entity recall; after R2A it's a SFT target and a *chosen* candidate; raw text is the natural *rejected* for persona/lead-gen/overconfidence.
- **Tier B (HealthChat):** unvalidated LLM text → never "gold". Used as a candidate pool, for conversational structure, and (after verification) SFT.
- **Tier C (synthetic):** teacher-written, gate-verified, clinician-audited. Fills safety/triage gaps.
- HealthChat-11K's authors report incomplete context, affective messages and leading questions that induce sycophancy → dedicated pair types PP5/PP6.

### 3.2 Scoring function
$$R(x,y)=G(x,y)\cdot\sum_k w_k\,s_k(x,y)\;-\;\big(1-G(x,y)\big)\,C_{viol}\;-\;\lambda_{len}\max\!\Big(0,\tfrac{|y|-L^*}{L^*}\Big)$$

| Term | Definition |
|---|---|
| $G\in\{0,1\}$ | Hard gate: all of — no persona claim; no child dose; tier-0 prompt ⇒ escalation present; crisis prompt ⇒ protocol items present; no method info; no PII. $C_{viol}=1$ |
| $s_{acc}$ ($w$=.35) | Judge accuracy, 1–5 → [0,1]; evidence-grounded when RAG passages exist |
| $s_{help}$ ($w$=.25) | Judge mean of context-seeking, empathy, actionability |
| $s_{ent}$ ($w$=.15) | $F_\beta$ of entity overlap with reference, $\beta=2$; no reference ⇒ grounded-entity ratio. $-0.5$ if an ungrounded CHEMICAL appears |
| $s_{dens}$ ($w$=.05) | $\exp\!\big(-\tfrac12((d-\mu_c)/\sigma_c)^2\big)$, $d=100\,|E(y)|/n_{tok}$; $(\mu_c,\sigma_c)$ per intent from top-quartile R2A answers |
| $s_{style}$ ($w$=.20) | No lead-gen / sign-off / persona; structure & length adherence |

$$F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2 P+R},\qquad P=\tfrac{|E_{gen}\cap E_{ref}|}{|E_{gen}|},\;R=\tfrac{|E_{gen}\cap E_{ref}|}{|E_{ref}|}$$
$R$-weighted ($\beta=2$) because the reference is one valid answer among many. Entity metrics are a diagnostic and never the headline. $L^*$ = `target_len` per prompt; $\lambda_{len}=0.5$.

### 3.3 Preference pair generators

| ID | Pair | Chosen | Rejected | Share |
|---|---|---|---|---|
| PP1 | On-policy best vs worst (K=8, T=0.8) | max R among gated samples | min R | 40% |
| PP2 | Safety-gate | gated, highest R | sample violating the gate | 15% |
| PP3 | Style/persona | R2A rewrite | **raw MedDialog reply** (persona/lead-gen/overconfident) | 10% |
| PP4 | Entity-swap hard negative | rewritten/gold answer | same text with 1–2 same-semantic-type entity swaps or dose ×10 (`error_type` tagged) | ≤ 10% |
| PP5 | Clarify vs assume | ≤ 2 targeted questions + safe general info | confident answer on incomplete context | 8% |
| PP6 | Anti-sycophancy | qualifies/corrects a leading premise | agrees with it | 7% |
| PP7 | Over-refusal | helpful, bounded answer | blanket refusal / "see a doctor" only | 7% |
| PP8 | Multi-turn consistency | respects an earlier-disclosed constraint (allergy, pregnancy) | contradicts it | 3% |

**Global rules:**
- ≤ 2 pairs per prompt.
- Margin ΔR ≥ 0.25.
- Length ratio within [1/1.5, 1.5] (except PP7), to avoid length confounds.
- A position-swapped judge must agree with scalar R.
- Prompts of one near-dup group stay in train.
- If no candidate passes the gate, route the prompt to `needs_gold` (teacher-written chosen, clinician-spot-checked).
- Audit 300 pairs (clinician or senior reviewer); require ≥ 85% agreement before training.
- PP4 teaches token-level artefact detection if overused: cap it, mix with natural errors (PP1), and confirm gains on a *natural* hallucination set (EVAL §4).

```python
# src/medchat/train/build_pairs.py
import math
from vllm import LLM, SamplingParams
from medchat.safety.risk import PERSONA_RX
from medchat.eval.entity_metrics import NER, NORM, entity_set, entity_scores

W, C_VIOL = dict(acc=.35, help=.25, ent=.15, dens=.05, style=.20), 1.0

def sample(path, prompts, k=8):
    llm = LLM(path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    sp = SamplingParams(n=k, temperature=0.8, top_p=0.95, max_tokens=768, seed=7)
    return [[o.text for o in r.outputs] for r in llm.chat([p["prompt"] for p in prompts], sp)]

def density_band(n_ent, n_tok, intent, bands):                   # bands[intent] = (mu, sd) per 100 tokens
    mu, sd = bands.get(intent, bands["default"]); d = 100 * n_ent / max(n_tok, 1)
    return math.exp(-0.5 * ((d - mu) / sd) ** 2)

def composite(p, y, judged, bands, gates_ok, leadgen_rx):
    gen = entity_set(y, NER, NORM)
    e = entity_scores(gen, p["ref_set"], p["prompt_set"])
    s = dict(acc=judged["acc"], help=judged["help"],
             ent=(e["F2"] if e["F2"] is not None else e["grounded"]) - 0.5 * bool(e["novel_chem"]),
             dens=density_band(len(gen), len(y) // 4, p["intent"], bands),
             style=float(not (PERSONA_RX.search(y) or leadgen_rx.search(y))))
    gate = gates_ok(p, y)
    S = sum(W[k] * max(0.0, s[k]) for k in W)
    over = max(0.0, (len(y) / 4 - p["target_len"]) / p["target_len"])
    return (S if gate else -C_VIOL) - 0.5 * min(1.0, over), s, gate

def mine_pairs(rows, judge_pair, delta=0.25, ratio=1.5):         # rows[i]["cands"] = [{"text","R","gate"}]
    pairs, needs_gold = [], []
    for r in rows:
        ok = [c for c in r["cands"] if c["gate"]]
        if not ok: needs_gold.append(r["id"]); continue
        best, worst = max(ok, key=lambda c: c["R"]), min(r["cands"], key=lambda c: c["R"])
        lr = len(best["text"]) / max(len(worst["text"]), 1)
        if best["R"] - worst["R"] < delta or not (1 / ratio <= lr <= ratio): continue
        if not judge_pair(r["prompt"], best["text"], worst["text"]): continue        # position-swapped agreement
        pairs.append({"id": r["id"], "type": "PP2" if not worst["gate"] else "PP1", "prompt": r["prompt"],
                      "chosen": [{"role": "assistant", "content": best["text"]}],
                      "rejected": [{"role": "assistant", "content": worst["text"]}],
                      "meta": {"margin": round(best["R"] - worst["R"], 3), "risk_flags": r["risk_flags"], "source": r["source"]}})
    return pairs, needs_gold
```

### 3.4 DPO (primary)
$$\mathcal{L}_{DPO}=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]\;+\;\alpha\,\mathcal{L}_{NLL}(y_w|x)$$
The $\alpha$ term (RPO-style, 0.2) counters likelihood displacement. TRL accepts a list of losses with weights, including `"sft"`.
```python
# src/medchat/train/dpo.py
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
base = "artifacts/medchat-sft-merged"                  # new LoRA on the MERGED SFT model; adapter-disabled = reference
tok, model = AutoTokenizer.from_pretrained(base), AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")
pairs = load_dataset("json", data_files={"train": "data/pref/v1/pairs_train.jsonl", "validation": "data/pref/v1/pairs_val.jsonl"})
pairs = pairs.select_columns(["prompt", "chosen", "rejected"])
args = DPOConfig(
    output_dir="runs/dpo-v1", seed=42, beta=0.1,
    loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2],          # older TRL: loss_type="sigmoid", rpo_alpha=0.2
    learning_rate=5e-6, lr_scheduler_type="cosine", warmup_ratio=0.1, num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,   # ×8 GPUs = 64 pairs
    max_length=3072, bf16=True, gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=50, save_steps=50, logging_steps=10, report_to="wandb")
DPOTrainer(model=model, ref_model=None, args=args, train_dataset=pairs["train"], eval_dataset=pairs["validation"],
           processing_class=tok, peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                                                        task_type="CAUSAL_LM")).train()
```
- **Sweep:** β ∈ {0.05, 0.1, 0.2} · LR ∈ {2e-6, 5e-6, 1e-5} (LoRA; full FT ≈ 5e-7) · `ipo` loss if reward accuracy hits ~1.0 quickly.
- **Watch:** `rewards/margins` ↑, `rewards/accuracies` ~0.7–0.85 (not 1.0), `logps/chosen` must not fall steadily (displacement → raise α, lower LR); length drift ≤ 15%.

### 3.5 ORPO (ablation B: single stage, no reference model)
$$\mathcal{L}_{ORPO}=\mathbb{E}\big[\mathcal{L}_{SFT}(y_w|x)+\lambda\,\mathcal{L}_{OR}\big],\;\; \mathcal{L}_{OR}=-\log\sigma\!\Big(\log\tfrac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\Big),\;\; \text{odds}_\theta(y|x)=\tfrac{P_\theta(y|x)}{1-P_\theta(y|x)}$$
with $P_\theta$ the length-normalised sequence probability. Use from the instruct checkpoint on **PP3 + PP5–PP7** pairs (style/behaviour), not as the sole carrier of safety pairs.
```python
from trl.experimental.orpo import ORPOConfig, ORPOTrainer     # experimental namespace in current TRL
args = ORPOConfig(output_dir="runs/orpo-b", beta=0.1, learning_rate=5e-6, max_length=3072,   # beta = λ
                  per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1, bf16=True)
```

### 3.6 GRPO (secondary: verifiable + rule/judge rewards)
$$\hat A_i=\frac{r_i-\text{mean}(r_{1:G})}{\text{std}(r_{1:G})+\epsilon},\quad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{old}(o_{i,t}|q,o_{i,<t})}$$
$$\mathcal{J}=\mathbb{E}_{q,\{o_i\}}\Big[\tfrac1G\sum_{i}\tfrac1{|o_i|}\sum_t\Big(\min\big(\rho_{i,t}\hat A_i,\ \text{clip}(\rho_{i,t},1\!-\!\epsilon,1\!+\!\epsilon)\hat A_i\big)-\beta D_{KL}[\pi_\theta\|\pi_{ref}]\Big)\Big]$$
TRL's default `loss_type="dapo"` normalises by active tokens in the batch instead of per-sequence length, which removes length bias. **Prompt mix:** 60% conversation (rule + judge reward), 25% risk prompts (safety-dominated), 15% MedQA/MedMCQA **train** MCQs (binary correctness; protects knowledge). Run GRPO only after DPO has passed its gate.
```python
# src/medchat/train/rewards.py   (dataset columns arrive as list kwargs aligned with completions)
import re
from medchat.safety.risk import DOSE_RX, PERSONA_RX, ESCALATE_RX
from medchat.eval.entity_metrics import NER, NORM, entity_set, entity_scores
from medchat.eval.judge import judge_pointwise_batch            # async, cached, 2-judge mean in [0,1]
_t = lambda c: c[-1]["content"] if isinstance(c, list) else c
LEADGEN = re.compile(r"consult me|contact me|my (profile|clinic)|video consultation|hope (this|i have) (helps|answered)", re.I)

def r_judge(prompts, completions, reference=None, **kw):  return judge_pointwise_batch(prompts, [_t(c) for c in completions], reference)
def r_style(prompts, completions, **kw):
    return [0.0 if (PERSONA_RX.search(t := _t(c)) or LEADGEN.search(t)) else 1.0 for c in completions]
def r_safety(prompts, completions, tier0, child, **kw):
    out = []
    for c, t0, ch in zip(completions, tier0, child):
        t, v = _t(c), 0.0
        v -= 1.0 * bool(PERSONA_RX.search(t))
        v -= 2.0 * bool(ch and DOSE_RX.search(t))
        v -= 2.0 * bool(t0 and not ESCALATE_RX.search(t))
        out.append(v)
    return out
def r_entity(prompts, completions, ref_entities, prompt_entities, **kw):
    out = []
    for c, re_, pe in zip(completions, ref_entities, prompt_entities):
        s = entity_scores(entity_set(_t(c), NER, NORM), {tuple(x) for x in re_}, {tuple(x) for x in pe})
        out.append((s["F2"] if s["F2"] is not None else s["grounded"]) - 0.5 * bool(s["novel_chem"]))
    return out
def r_len(prompts, completions, target_len, **kw):
    return [-min(1.0, max(0.0, (len(_t(c)) / 4 - L) / L)) for c, L in zip(completions, target_len)]
def r_mcq(prompts, completions, answer, **kw):                  # prompt demands a final line "Answer: X"
    pick = lambda t: (re.findall(r"answer\s*:\s*\(?([A-E])\)?", t, re.I) or [None])[-1]
    return [0.0 if not a else float((pick(_t(c)) or "").upper() == a) for c, a in zip(completions, answer)]
```
```python
# src/medchat/train/grpo.py
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from medchat.train.rewards import *
args = GRPOConfig(
    output_dir="runs/grpo-v1", seed=42, learning_rate=5e-6, lr_scheduler_type="constant_with_warmup", warmup_steps=20,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,   # global completions must be a multiple of num_generations
    num_generations=8, max_completion_length=768, temperature=0.9,  # filter prompts to ≤1.5k tokens in the dataset
    loss_type="dapo", beta=0.02, epsilon=0.2, epsilon_high=0.28,    # TRL default beta=0.0 (no ref model); small KL anchor on purpose
    mask_truncated_completions=True,
    reward_weights=[1.0, 0.3, 0.3, 1.0, 1.0, 1.0],                  # judge, entity, style, safety, len, mcq
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
    bf16=True, gradient_checkpointing=True, max_steps=1500, save_steps=100, logging_steps=5, report_to="wandb")
GRPOTrainer(model="artifacts/medchat-dpo-merged", reward_funcs=[r_judge, r_entity, r_style, r_safety, r_len, r_mcq],
            args=args, train_dataset=grpo_ds, processing_class=tok,
            peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")).train()
```
**Reward-hacking controls:** two judges from different families + a *held-out third* judge for evaluation only · length penalty + density band · rubric paraphrase rotation · KL anchor · weekly human spot-check of top-reward samples · alert if training reward rises while held-out judge/HealthBench falls. Judge timeouts fall back to rule rewards and are logged.

### 3.7 Gates & failure modes
| Failure | Signal | Fix |
|---|---|---|
| Likelihood displacement | `logps/chosen` ↓ | ↑α, ↓LR, `ipo`, fewer epochs |
| Verbosity creep | length ↑ > 15% | length-matched pairs, `r_len`, SimPO-style length control |
| Refusal creep | over-refusal ↑ | more PP7, lower safety weight on benign prompts |
| Entity stuffing | density ↑ with accuracy flat | density band, claim-level verification |
| Sycophancy | S6 suite ↓ | more PP6 |
| Judge hacking | judge↑, HealthBench/clinician↓ | judge ensemble, held-out judge |
| Forgetting | MedQA/MMLU ↓ > 2 pts | LoRA, lower LR, MCQ share in GRPO, replay |

## 4. Reproducibility
Seed 42 everywhere (sampling seed 7). Log: data manifest hash, prompt-policy version, TRL/transformers/peft/vllm versions, base-model revision SHA, adapter name `medchat-{stage}-{date}-{gitsha}`. Every checkpoint is evaluated by the same `scripts/run_eval.sh`.

## 5. References (checked 2026-10-23)
TRL docs: SFT (`assistant_only_loss`, `packing`, `max_length`), GRPO (`loss_type`, `beta`, `use_vllm`), DPO (`loss_type` list + `loss_weights`), ORPO (`trl.experimental.orpo`) — https://huggingface.co/docs/trl · Llama AUP — https://developer.meta.com/ai/llama3/use-policy/ · Qwen3-8B card — https://huggingface.co/Qwen/Qwen3-8B · HealthChat-11K — https://arxiv.org/abs/2506.21532