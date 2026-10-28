# DATA_PIPELINE.md — from `integrated_conversations_clean.csv` to SFT and preference data

**Input:** `data/processed/integrated_conversations_clean.csv` (nb06 output, preferred). Fallback: `…_ner_validated.csv` (275,639 turns / 123,256 dialogues).
**Outputs:** `data/sft/v1/{train,val,test}.jsonl` · `data/pref/v1/prompts.jsonl` · `data/sft/v1/manifest.json`

## 0. Contract & schema assumptions
Verify with `df.columns` before running. The clean file is only described at stage level in the reference.
- Required: `dialogue_id, turn_id, speaker, utterance, source_dataset, original_id, extracted_entities, entity_labels, entity_count`. Optional: `umls_cuis`.
- Labels: `DISEASE, CHEMICAL, SYMPTOM, ANATOMY, BEHAVIOR`. Entity columns are parallel pipe-delimited lists.
- Always load with `dtype={"original_id": str, "dialogue_id": str}`.

| Stage | Module | Output |
|---|---|---|
| 1 Audit & reconcile | `audit.py` | `reports/audit.json` |
| 2 Tag, tier, recover generator model | `tag.py` | tier/licence/model columns |
| 3 Structural repair & hygiene | `repair.py` | valid dialogues only |
| 4 Anti-collapse rules | tests | — |
| 5 Entities, flags, intent, conditioning | `entities.py` | context blocks |
| 6 Quality scoring + R2A | `quality.py`, `r2a.py` | cleaned assistant text |
| 7 Group-aware split | `split.py` | `split` column |
| 8–10 Render, mask, weight | `render.py`, `masking.py` | JSONL / tensors |
| 11 Mixture + preference prompts | `mixture.py` | final sets |
| 12 QA + manifest | `tests/` | manifest |

## 1. Audit & reconciliation
```python
# src/medchat/data/audit.py
import re, json, pandas as pd
REQUIRED = ["dialogue_id","turn_id","speaker","utterance","source_dataset","original_id",
            "extracted_entities","entity_labels","entity_count"]
LABELS = {"DISEASE","CHEMICAL","SYMPTOM","ANATOMY","BEHAVIOR"}
_S = re.compile(r"\s*\|\s*")
_n = lambda s: len([x for x in _S.split(s) if x])

def load(path):
    df = pd.read_csv(path, low_memory=False, dtype={"original_id": str, "dialogue_id": str})
    miss = [c for c in REQUIRED if c not in df.columns]
    assert not miss, f"missing columns {miss}: fix REQUIRED or the nb06 export"
    df[["extracted_entities","entity_labels"]] = df[["extracted_entities","entity_labels"]].fillna("")
    return df.sort_values(["dialogue_id","turn_id"], kind="stable").reset_index(drop=True)

def audit(df, validated_file=False):
    g, src = df.groupby("dialogue_id"), df.groupby("source_dataset")
    rep = {
      "rows": len(df), "dialogues": g.ngroups,
      "null_utterances": int(df.utterance.isna().sum()),                 # 198 in the validated file
      "dup_turn_keys": int(df.duplicated(["dialogue_id","turn_id"]).sum()),
      "turns_per_dialogue": (src.size() / src.dialogue_id.nunique()).round(2).to_dict(),
      "single_exchange_share": float((g.size() == 2).mean()),            # ≈0.91 (MedDialog)
      "role_counts": df.speaker.value_counts().to_dict(),
      "bad_alternation_dialogues": int(g.speaker.apply(lambda s: s.iloc[0] != "user" or s.iloc[-1] != "assistant"
                                       or any(a == b for a, b in zip(s, s[1:]))).sum()),
      "misaligned_entity_rows": int((df.extracted_entities.map(_n) != df.entity_labels.map(_n)).sum()),
      "bad_labels": sorted({l for s in df.entity_labels for l in _S.split(s) if l and l not in LABELS}),
      "chatdoctori_rows": int(df.extracted_entities.str.contains("chatdoctori", case=False).sum()),
      "entityless_turns": int((df.entity_count == 0).sum()),
      "entity_coverage_by_source": (src.entity_count.apply(lambda x: (x > 0).mean())).round(3).to_dict(),
    }
    assert rep["dup_turn_keys"] == 0 and not rep["bad_labels"], rep
    if validated_file:   # reference: 46,889 entity-less rows vs ≈48.7k implied by per-source coverage → report, don't trust
        print("entity-less turns:", rep["entityless_turns"], "| reference says 46,889; per-source coverage implies ≈48.7k")
    json.dump(rep, open("reports/audit.json", "w"), indent=2, default=str)
    return rep
```
**Must-pass (clean file):** no duplicate `(dialogue_id, turn_id)` · labels ⊆ allowed set · `chatdoctori_rows == 0` · `null_utterances == 0` after stage 3 · report `single_exchange_share` and `bad_alternation_dialogues` (expected > 0 before repair).

## 2. Tagging, trust tiers, generator-model recovery
```python
# src/medchat/data/tag.py
import pandas as pd
SOURCE_META = {
  "MedDialog":           dict(trust_tier="A", license_track="research_only"),
  "HealthChat-LMSYS":    dict(trust_tier="B", license_track="lmsys_agreement"),
  "HealthChat-WildChat": dict(trust_tier="B", license_track="wildchat_verify_variant"),
}
def tag(df):
    for k in ("trust_tier", "license_track"):
        df[k] = df.source_dataset.map(lambda s: SOURCE_META[s][k])
    return df

def recover_gen_model(df):
    """Join generator model/lang from the ORIGINAL corpora (needs accepted LMSYS licence + HF token).
    Pull ONLY id/model/language/redacted. Never import geo/IP columns (non-identification terms)."""
    from datasets import load_dataset
    parts = []
    for src, repo, idc in [("HealthChat-LMSYS", "lmsys/lmsys-chat-1m", "conversation_id"),
                           ("HealthChat-WildChat", "allenai/WildChat-1M", "conversation_hash")]:
        t = (load_dataset(repo, split="train").select_columns([idc, "model", "language", "redacted"])
             .to_pandas().rename(columns={idc: "original_id", "model": "gen_model", "language": "orig_lang"}))
        t["source_dataset"] = src; parts.append(t)
    out = df.merge(pd.concat(parts), on=["source_dataset", "original_id"], how="left")
    hit = out[out.source_dataset.str.startswith("HealthChat")].gen_model.notna().mean()
    assert hit > 0.99, f"join rate {hit:.3f}: check id mapping / WildChat variant; else skip and rely on judge scores"
    return out
```
Your HealthChat counts (6,828 + 4,264 = 11,092 dialogues) match HealthChat-11K, which was filtered from LMSYS-Chat-1M and WildChat-1M, so the `original_id` join should work. Verify provenance.

`configs/data/gen_model_allowlist.yaml` is placeholder regexes. **Calibrate with a 500-sample judge audit, not model reputation.**
```yaml
keep: ["^gpt-4", "^gpt-3\\.5", "^claude", "^llama-2-.*-chat"]
drop: ["^dolly", "^oasst", "^koala", "^alpaca", "^fastchat-t5", "^rwkv", "^stablelm"]
```

## 3. Structural repair & hygiene
```python
# src/medchat/data/repair.py
import re, pandas as pd
_cat = lambda a, b: " | ".join(x for x in (a, b) if x)

def repair(df, max_pairs=8):
    df = df[df.utterance.fillna("").str.strip().ne("")]                      # drops the 198 nulls + blanks
    rows, dropped, merged = [], 0, 0
    for _, g in df.groupby("dialogue_id", sort=False):
        turns = []
        for r in g.to_dict("records"):
            if turns and turns[-1]["speaker"] == r["speaker"]:               # consecutive same-role → merge
                p = turns[-1]; merged += 1
                p["utterance"] += "\n\n" + r["utterance"]
                p["extracted_entities"] = _cat(p["extracted_entities"], r["extracted_entities"])
                p["entity_labels"] = _cat(p["entity_labels"], r["entity_labels"])
            else:
                turns.append(dict(r))
        while turns and turns[0]["speaker"] != "user": turns.pop(0)          # must start with user
        while turns and turns[-1]["speaker"] != "assistant": turns.pop()     # must end with assistant
        turns = turns[: 2 * max_pairs]                                       # truncate at a turn boundary
        if len(turns) < 2: dropped += 1; continue
        for i, t in enumerate(turns): t["turn_id"] = i
        rows += turns
    return pd.DataFrame(rows), {"dialogues_dropped": dropped, "turns_merged": merged}

PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"), (re.compile(r"https?://\S+|www\.\S+"), "[URL]")]
_PH = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
def scrub_pii(t):
    for rx, ph in PII: t = rx.sub(ph, t)
    return _PH.sub(lambda m: "[PHONE]" if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), t)  # spares "120 80 70"
```
Further hygiene (log each step in the funnel):
1. **Language ID** (fastText `lid.176` or `lingua`): keep `en` with p ≥ 0.8. Other languages go to `data/other_lang/` (v2).
2. **Template residue scan** on assistant turns (§6 regexes): flag, don't blindly delete. Residual rate must be < 0.5% after R2A.
3. **Markdown-table stripper for entity features only** (HealthChat): never alter `utterance`.
4. Keep LMSYS `NAME_k` tokens as-is. Apply `scrub_pii` to user and assistant text. Drop assistant turns that still contain placeholders.

## 4. Anti-collapse rules (enforced by tests, §12)
1. One sample = one dialogue (system + all turns), never independent `(question → answer)` rows.
2. Loss on **every** assistant turn; user/system tokens masked.
3. Never concatenate history into one user string or keep only the first exchange.
4. Roles strictly alternate `user, assistant, …`; truncate only at turn boundaries (≤ 8 exchanges).
5. Dedup at **dialogue** level; if a turn must be dropped, drop its whole exchange or the dialogue.
6. Multi-turn eval slice: test/val keep ≥ 40% HealthChat dialogues with ≥ 2 exchanges (§7 quotas).
7. **Synthetic clarify→answer** (≤ 5% of mix, Tier C, never in test): from MedDialog replies containing clarifying questions, a teacher simulates a patient reply constrained to the original facts, then writes the final answer; all outputs pass the R2A gates.

## 5. Entities, intent, risk flags, conditioning
**Policy:** conditioning uses **normalised surface strings + coarse label**. CUIs (`umls_cuis`) are pipeline keys: label-conflict resolution (the 829 dual-label strings), routing, retrieval keys, same-semantic-type swaps for hard negatives, and evaluation matching. Raw CUIs never go into prompt text.
```python
# src/medchat/data/entities.py
import re, random
from medchat.safety.risk import _scan
FIELD = {"SYMPTOM": "symptoms", "DISEASE": "conditions", "CHEMICAL": "medications_substances",
         "ANATOMY": "anatomy", "BEHAVIOR": "behaviors"}
_S = re.compile(r"\s*\|\s*")

def parse(ents, labs):
    e, l = [x for x in _S.split(ents) if x], [x for x in _S.split(labs) if x]
    if len(e) != len(l): return []                    # fail-safe: misaligned (pipe inside a table cell) → no entities
    out, seen = [], set()
    for s, lab in zip(e, l):
        s = s.lower().strip()
        if not 2 <= len(s) <= 40 or "\n" in s or s.isdigit() or lab not in FIELD or (s, lab) in seen: continue
        seen.add((s, lab)); out.append((s, lab))
    return out

def derive_intent(text, ents, flags):
    t, labs = text.lower(), {l for _, l in ents}
    if flags & {"suicidal_ideation", "self_harm", "suicide_topic"} or re.search(r"\b(depress|anxi|panic)", t): return "mental_health"
    if re.search(r"\b(pregnan|trimester|breastfe|baby|infant|toddler|my child)\b", t): return "pregnancy_peds"
    if re.search(r"\b(lab|blood test|results?|reference range|hba1c|creatinine|bilirubin)\b", t): return "lab_interpretation"
    if "CHEMICAL" in labs and re.search(r"\b(take|dose|dosage|side effects?|interact)", t): return "medication_info"
    if labs & {"SYMPTOM", "DISEASE"}: return "symptom_triage"
    return "other"

def context_block(ents, flags=(), intent="other", cap=6):
    by = {}
    for s, lab in ents: by.setdefault(FIELD[lab], []).append(s)
    lines = [f"{k}: {'; '.join(v[:cap])}" for k, v in by.items()]
    if flags: lines.append("risk_flags: " + ", ".join(sorted(flags)))
    if intent != "other": lines.append(f"intent: {intent}")
    return "<clinical_context>\n" + "\n".join(lines) + "\n</clinical_context>" if lines else ""

def noisy_context(ents, flags, intent, rng: random.Random, p_all=.25, p_ent=.15, p_label=.05):
    if rng.random() < p_all: return ""                              # whole block dropped
    ents = [(s, rng.choice(list(FIELD)) if rng.random() < p_label else l)    # simulated NER label errors
            for s, l in ents if rng.random() > p_ent]                # simulated NER misses
    rng.shuffle(ents); return context_block(ents, flags, intent)
```
- **Cumulative, user-side only:** the block prefixed to user turn *t* is built from user turns ≤ *t*. Assistant entities are used only for rewards/evaluation, never as input.
- At **inference**, the serving layer builds the same block with the same NER + normaliser (`entity_norm_map.csv`) and **no dropout**.
- Risk flags come from the shared `medchat.safety.risk` module, so train and serve use identical lexicons.
- **Routing at inference:** intent classifier + flags select the system-prompt variant, RAG index and guard strictness (`medication_info` → strict DoseGuard + label RAG; `mental_health` → crisis-aware mode).

## 6. Quality scoring & Rewrite-to-Assistant (R2A)
**Heuristic filters** (assistant turns):
- 25 ≤ tokens ≤ 1,800, with exceptions for clarifying questions.
- distinct-3-gram ratio ≥ 0.6.
- entity density per 100 tokens within [p5, p99] of its source (kills table dumps).
- ≥ 1 shared normalised entity with the user turn, or embedding cosine ≥ 0.3 (kills off-topic replies).
- no code blocks; no unresolved placeholders.

**Flag regexes** (flag, then R2A or drop):
```python
LEADGEN = [r"chat\s*doctor", r"chatdoctor\w*", r"healthcare\s*magic", r"icliniq",
           r"\bhope (i|this) (have )?(answered|helps?)", r"\bthanks? for (your )?(query|consulting)",
           r"\b(regards|best wishes)\b", r"\bdr\.?\s+[A-Z]\w+", r"\b(consult|contact|message) me\b",
           r"\bi am (a )?(doctor|physician)\b", r"\bas (a|your) (doctor|physician)\b", r"\bwelcome to\b"]
```
**LLM-judge pass** (open-weights teacher, temp 0, JSON): `relevance, accuracy_risk, safety_risk, overconfident_dx, persona, lead_gen, dose_given, completeness, empathy`. Keep composite ≥ τ (set from a 300-sample clinician-labelled calibration set); route flagged Tier A to R2A, Tier B to verification (claim support ≥ 0.9, §EVAL 4) or drop.

**R2A prompt** (`configs/data/r2a_prompt.txt`):
```
SYSTEM: You are a clinical editor. Rewrite a telemedicine doctor's reply so an AI health-information assistant can show it.
1. Preserve every clinically relevant fact, recommendation, red flag and differential from ORIGINAL. Add NO new diagnoses, drugs, tests or numbers.
2. Remove doctor persona (names, "I examined/prescribe"), greetings/sign-offs, platform or lead-gen text, invitations to private contact.
3. Rephrase definitive diagnoses made without examination as "possible causes"; keep ORIGINAL's urgency.
4. Keep medication doses only if ORIGINAL states them AND the patient is an adult; otherwise write "use the dose on the product label or ask a pharmacist".
5. Plain language, ≤1.3× ORIGINAL length. Add when-to-seek-care advice only if ORIGINAL implies it.
6. If ORIGINAL is off-topic, unsafe or unintelligible → {"status":"reject","reason":"..."}.
OUTPUT JSON: {"status":"ok","rewrite":"..."}
USER: PATIENT: <<USER_TURN>>   ORIGINAL: <<DOCTOR_REPLY>>
```
**Verification gates** (all must pass, else fall back to cleaned original if regex-clean, else drop):
```python
PLATFORM = {"chatdoctori", "chat doctor", "healthcare magic", "icliniq"}
def r2a_gates(orig, rew, user_set, ner, norm):
    eo, er = entity_set(orig, ner, norm), entity_set(rew, ner, norm)      # eval/entity_metrics.py
    core = {e for e in eo if e[0] not in PLATFORM}
    recall = len(core & er) / max(len(core), 1)
    new_chem = {e for e in er - eo - user_set if e[1] == "CHEMICAL"}      # blocks invented drugs
    ok = (recall >= .85 and not new_chem and not PERSONA_RX.search(rew)
          and .6 <= len(rew) / max(len(orig), 1) <= 1.6
          and not (DOSE_RX.search(rew) and not DOSE_RX.search(orig)))
    return ok, {"recall": round(recall, 2), "new_chem": sorted(new_chem)}
```
Target R2A acceptance ≥ 85%. Stratified clinician audit of 200 rewrites (by intent and risk flag); iterate the prompt until ≥ 90% "faithful & safe". Keep `orig` and `rew` side by side (`r2a_orig`, `r2a_text`): the raw text becomes DPO *rejected* material (FT §3.3, PP3).

## 7. Dialogue-grouped splits
Group by **near-duplicate clusters across all sources** (canned replies and repeated pastes are the main leakage path), assign each group to exactly one split, hit per-source quotas deterministically.
```python
# src/medchat/data/split.py
import re, hashlib, pandas as pd
from datasketch import MinHash, MinHashLSH
SALT = "medchat-v1-2026-10"
QUOTA = {"MedDialog": dict(val=2000, test=2000), "HealthChat-LMSYS": dict(val=350, test=350),
         "HealthChat-WildChat": dict(val=250, test=250)}           # dialogues; HealthChat ≈ 5.5% each
PRIORITY = ["HealthChat-WildChat", "HealthChat-LMSYS", "MedDialog"]  # rarest source decides a mixed group's stratum

def shingles(t, k=5):
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}

def build_groups(dlg, thr=0.8, perm=128):        # dlg: [dialogue_id, key_text = first user turn (+ first assistant turn)]
    lsh, sig = MinHashLSH(threshold=thr, num_perm=perm), {}
    for d, txt in zip(dlg.dialogue_id, dlg.key_text):
        m = MinHash(num_perm=perm); [m.update(s.encode()) for s in shingles(txt)]
        sig[d] = m; lsh.insert(d, m)
    parent = {d: d for d in sig}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for d, m in sig.items():
        for nb in lsh.query(m):
            a, b = find(d), find(nb)
            if a != b: parent[b] = a
    return {d: find(d) for d in sig}

def assign(dlg, groups):                         # dlg: [dialogue_id, source_dataset]
    dlg = dlg.assign(group=dlg.dialogue_id.map(groups))
    g = dlg.groupby("group").agg(n=("dialogue_id", "size"),
                                 src=("source_dataset", lambda s: min(s, key=PRIORITY.index))).reset_index()
    g["h"] = g.group.map(lambda x: hashlib.md5(f"{SALT}:{x}".encode()).hexdigest())
    split = {}
    for src, sub in g.sort_values("h").groupby("src"):
        need = dict(QUOTA[src])
        for r in sub.itertuples():
            s = "test" if need["test"] > 0 else "val" if need["val"] > 0 else "train"
            if s != "train": need[s] -= r.n
            split[r.group] = s
    return dlg.dialogue_id.map(lambda d: split[groups[d]])
```
- Log realised vs target counts; assert each within ±10% of quota (mixed-source groups shift counts slightly).
- **Gold subset:** a stratified 500 dialogues drawn from `test` only (by intent × risk flag × source) for clinician review; never used for tuning.
- Freeze `test` at v1; changes require a new SALT and a version bump.

## 8. Prompt templates (canonical JSONL → ChatML / ShareGPT)
**Canonical record** (`train.jsonl`; training reads only `messages` via `ds.select_columns(["messages"])`):
```json
{"id":"<md5 dialogue_id>","source":"MedDialog","trust_tier":"A","license_track":"research_only","split":"train",
 "messages":[{"role":"system","content":"<policy_prompt_v1>"},
             {"role":"user","content":"<clinical_context>\nsymptoms: headache; nausea\nintent: symptom_triage\n</clinical_context>\nI've had a pounding headache and nausea since yesterday. What could it be?"},
             {"role":"assistant","content":"…R2A text…"}],
 "meta":{"intent":"symptom_triage","risk_flags":[],"n_exchanges":1,"synthetic":false,"gen_model":null,
         "r2a_orig":"…raw doctor reply…","entities_assistant":[["migraine","DISEASE"]],"cuis":["C0027497"]}}
```
**Policy prompt v1** (`configs/data/policy_prompt_v1.txt`; version it, and train/serve must match):
```
You are MedChat, an AI health-information assistant. You are not a doctor and cannot examine the user or see their records.
1. Be accurate, concise and empathetic. Present likely causes as possibilities, never as a diagnosis.
2. Always name red-flag symptoms and when to seek urgent or emergency care. If life-threatening symptoms are described, tell the user to contact local emergency services now.
3. If key facts (age, duration, pregnancy, medications, allergies) are missing and change the advice, ask at most two targeted questions and still give safe general guidance.
4. Do not change or prescribe medication. Do not state doses for children. For adults, mention a dose only if it appears on a standard product label and suggest confirming with a pharmacist.
5. Never claim to be human or a clinician, never sign off as a person, never invite private contact.
6. If the user may be in crisis (self-harm, suicide, abuse): respond warmly, encourage immediate human help and local crisis resources, give no methods.
7. <clinical_context> is machine-extracted and may contain errors; the user's own words take precedence.
```
- **System-prompt dropout:** 10% of training samples have no system message, and 10% use a paraphrased variant. Safety must not depend on one exact prompt.
- **Render with the model's own template** (`tokenizer.apply_chat_template`). For Qwen3, train and serve in non-thinking mode (`enable_thinking=False`). The parity assertion in `encode_example` (§10) catches the known train/serve mismatch around the empty `<think>` block.
- **Rendered example** (loss only on `▶` segments):
```
<|im_start|>system {policy_prompt_v1}<|im_end|>
<|im_start|>user <clinical_context>…</clinical_context> I've had chest tightness since morning…<|im_end|>
<|im_start|>assistant ▶ "Chest tightness can have many causes… If it spreads to your arm or jaw, comes with sweating or breathlessness, call emergency services now. Two quick questions: …"<|im_end|>
<|im_start|>user <clinical_context>(cumulative: symptoms: chest tightness; risk_flags: cardiac)</clinical_context> It eased after resting. Can I take ibuprofen?<|im_end|>
<|im_start|>assistant ▶ "…"<|im_end|>
```
- **ShareGPT exporter** (Axolotl/LLaMA-Factory): `{"conversations":[{"from":"system|human|gpt","value":…}]}` via `{"user":"human","assistant":"gpt"}`.

```python
# src/medchat/data/render.py
import random
from medchat.data.entities import parse, derive_intent, context_block, noisy_context
from medchat.safety.risk import _scan

def build_messages(turns, system, rng: random.Random, train=True):
    msgs = [] if (train and rng.random() < .10) else [{"role": "system", "content": system}]
    ents, flags = [], set()
    for t in turns:
        if t["speaker"] == "user":
            ents = (ents + [e for e in parse(t["extracted_entities"], t["entity_labels"]) if e not in ents])[:12]
            flags |= _scan(t["utterance"])[0]                               # USER text only
            intent = derive_intent(t["utterance"], ents, flags)
            ctx = (noisy_context if train else context_block)(ents, flags, intent, *( [rng] if train else [] ))
            msgs.append({"role": "user", "content": (ctx + "\n" if ctx else "") + t["utterance"]})
        else:
            msgs.append({"role": "assistant", "content": t.get("r2a_text") or t["utterance"]})
    return msgs
```

## 9. Leakage & decontamination checks
```python
def assert_no_leak(split_of, groups, orig_id_of):
    by = {}
    for d, s in split_of.items(): by.setdefault(groups[d], set()).add(s)
    assert not [g for g, s in by.items() if len(s) > 1], "near-dup group spans splits"
    tr = {orig_id_of[d] for d, s in split_of.items() if s == "train"}
    assert not any(orig_id_of[d] in tr for d, s in split_of.items() if s != "train"), "original_id leak"

def ngrams(t, n=13):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
BENCH = set().union(*(ngrams(q) for q in benchmark_prompts))   # MedQA/MedMCQA/PubMedQA/MMLU-med/HealthBench prompts
drop_ids = {d for d, u in first_user_text.items() if ngrams(u) & BENCH}   # LMSYS/WildChat users paste exam questions
```
Also assert zero exact first-user-turn overlap (hash) between train and val/test, and log how many dialogues decontamination removed.

## 10. Entity-aware masking, weighting & noise
Four mechanisms:
1. **Assistant-only loss** (default): TRL `SFTConfig(assistant_only_loss=True)` (FT §2). `encode_example` below is a template-agnostic fallback and parity test.
2. **Entity-span upweighting** (ablation A3): tokens overlapping assistant-side clinical entities get weight `1+λ` (λ=0.5).
3. **Conditioning noise/dropout** (§5): robustness to NER errors, no over-reliance on the block.
4. **Entity-swap hard negatives** for DPO (FT §3.3, PP4).
```python
# src/medchat/data/masking.py
import re, torch

def encode_example(tok, messages, ent_surfaces=(), lam=0.5, max_len=4096, **tmpl):
    text = tok.apply_chat_template(messages, tokenize=False, **tmpl)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            pre  = tok.apply_chat_template(messages[:i],     tokenize=False, add_generation_prompt=True, **tmpl)
            upto = tok.apply_chat_template(messages[:i + 1], tokenize=False, **tmpl)
            assert text.startswith(upto) and upto.startswith(pre), "chat template is not prefix-stable (train/serve mismatch)"
            spans.append((len(pre), len(upto)))
    low = text.lower()
    ent = [m.span() for e in ent_surfaces for m in re.finditer(re.escape(e.lower()), low)]
    ids, labels, w = enc["input_ids"], [], []
    for tid, (s, e) in zip(ids, enc["offset_mapping"]):
        sup = any(a <= s and e <= b for a, b in spans)                   # includes <|im_end|> so the model learns to stop
        labels.append(tid if sup else -100)
        w.append(0.0 if not sup else 1.0 + (lam if any(s < b and e > a for a, b in ent) else 0.0))
    return {"input_ids": ids, "labels": labels, "token_weights": w}

def weighted_ce(logits, labels, w):
    logits, labels, w = logits[:, :-1], labels[:, 1:], w[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                                           ignore_index=-100, reduction="none").view_as(w)
    return (ce * w).sum() / w.sum().clamp(min=1.0)

def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch); pad = lambda x, v: x + [v] * (L - len(x))
    return {"input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "token_weights": torch.tensor([pad(b["token_weights"], 0.0) for b in batch])}
```
`WeightedTrainer` (plain `transformers.Trainer`, pre-tokenised dataset, `data_collator=collate`):
```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        w, labels = inputs.pop("token_weights"), inputs["labels"]
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        loss = weighted_ce(out.logits, labels, w)
        return (loss, out) if return_outputs else loss
```

## 11. Mixture, packing, preference-prompt bank
**Target token shares** (starting hypothesis; tune via ablation A4). Convert to per-example sampling probabilities `p_i ∝ share_i / mean_tokens_i`.

| Component | Target token share |
|---|---|
| MedDialog (R2A-accepted, quality-filtered) | 55% |
| HealthChat-WildChat (filtered) | 15% |
| HealthChat-LMSYS (generator-model + judge filtered) | 10% |
| Synthetic safety/triage SFT (Tier C; red-flag, crisis, dosing, deferral) | 8% |
| Synthetic clarify→answer multi-turn (Tier C) | 7% |
| General replay (permissively licensed instruction data) | 5% |

```python
from datasets import interleave_datasets
def probs(shares, mean_tok):
    raw = {k: shares[k] / mean_tok[k] for k in shares}; z = sum(raw.values()); return {k: v / z for k, v in raw.items()}
# cap repeats: natural_tokens × repeats ≥ share × total  →  assert repeats ≤ 3 for every small source
mix = interleave_datasets([ds[k] for k in order], probabilities=[p[k] for k in order], seed=42,
                          stopping_strategy="all_exhausted")      # oversamples small sources; check the repeat cap
```
Goal: ≥ 30% of assistant-loss tokens from multi-turn dialogues (raw share is far lower). Max length 4,096; **packing needs FlashAttention** so examples don't attend across packed boundaries. Optional anneal: last 20% of steps use only Tier A-R2A + verified Tier B + safety data.

**Preference-prompt bank** (`data/pref/v1/prompts.jsonl`, **train split only**, ~60k prompts):
```json
{"id":"…","prompt":[{"role":"system","content":"…"},{"role":"user","content":"…"}],
 "reference":"R2A text or null","ref_entities":[["ibuprofen","CHEMICAL"]],"prompt_entities":[["chest pain","SYMPTOM"]],
 "tier0":false,"child":false,"intent":"medication_info","risk_flags":[],"answer":"","target_len":350,"source":"MedDialog"}
```
Strata: 35% routine · 25% medication/dose · 15% mental-health/crisis · 15% red-flag triage · 10% pregnancy/paediatrics. Add ~10% adversarial prompts: leading questions, minimisers ("probably nothing, but…"), incomplete context. Include prefix prompts for each user turn of multi-turn dialogues (≤ 3 per dialogue), plus ≥ 15% MCQ rows (MedQA/MedMCQA **train** splits only, `answer` set) for the GRPO verifiable component.

## 12. QA, tests, manifest
```python
# tests/test_data.py  (run in CI against data/sft/v1)
def test_alternation(rows):       assert all(m["role"] in ("user", "assistant") and m["role"] == ("user", "assistant")[i % 2]
                                             for r in rows for i, m in enumerate(r["messages"] if r["messages"][0]["role"] != "system" else r["messages"][1:]))
def test_no_group_leak():         assert_no_leak(split_of, groups, orig_id_of)
def test_no_assistant_entity_leak(rows):   # context block at turn t only contains entities from user turns ≤ t
    for r in rows: assert not any(e in r["messages"][i]["content"] for i, e in assistant_only_entities(r))
def test_template_parity(tok, rows):       # train render of prefix == serve render + generation prompt
    for r in rows[:500]: encode_example(tok, r["messages"], enable_thinking=False)   # asserts internally
def test_loss_only_on_assistant(tok, rows):
    ex = encode_example(tok, rows[0]["messages"]); assert "<|im_start|>user" not in tok.decode([t for t in ex["labels"] if t != -100])
def test_residue(rows):           assert residue_rate(rows) < 0.005
```
`manifest.json`: counts per split/source/tier · funnel (rows in → after each stage) · SHA256 of each file · git SHA · config hash · licence composition by tokens · NER model + normaliser versions · R2A acceptance and audit results. `docs/DATA_CARD.md` covers provenance (flag the MedDialog and WildChat-variant questions), known biases (HealthCareMagic/iCliniq population, English-only, 2023 LLM generators), PII handling, and intended and prohibited uses.