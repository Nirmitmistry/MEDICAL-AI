# Dataset Reference: `integrated_conversations_ner_validated.csv`

> **File path:** `data/processed/integrated_conversations_ner_validated.csv`
> **Produced by:** `notebooks/05_ner_validation.ipynb`
> **Input file:** `data/processed/integrated_conversations_ner.csv` (output of `04_scispacy_ner.ipynb`)
> **NER model:** `en_ner_bc5cdr_md` (scispaCy — BC5CDR corpus, DISEASE + CHEMICAL only)

For the fully cleaned successor file, see [`integrated_conversations_clean.csv`](#11-successor-file-integrated_conversations_cleancsv) produced by `notebooks/06_cleaning_pipeline.ipynb`.

---

## 1. Overview

This is the validated version of the integrated medical conversation dataset after NER extraction and multi-stage validation. It is the recommended starting point for inspection; for modelling, prefer `integrated_conversations_clean.csv` which applies additional cleaning (see §11).

| Property | Value |
|---|---|
| Total rows (turns) | 275,639 |
| Total unique dialogues | 123,256 |
| Total entity mentions | 778,779 |
| Unique entity strings (raw, case-sensitive) | 69,039 |
| File size / format | CSV, UTF-8 |

---

## 2. Schema

| Column | Type | Nullable | Description |
|---|---|---|---|
| `dialogue_id` | `str` (MD5 hash) | No | Unique identifier for each conversation |
| `turn_id` | `int` | No | Zero-indexed turn position within the dialogue |
| `speaker` | `str` | No | `"user"` or `"assistant"` |
| `utterance` | `str` | Yes (198 nulls — see §8) | Raw text of the turn |
| `source_dataset` | `str` | No | Source dataset name (see §3) |
| `original_id` | `str` | No | ID from the original source dataset |
| `extracted_entities` | `str` | Yes (46,889 empty) | Pipe-delimited entity surface forms, e.g. `"pain \| OCD \| ibuprofen"` |
| `entity_labels` | `str` | Yes (46,889 empty) | Pipe-delimited labels aligned to `extracted_entities`, e.g. `"DISEASE \| DISEASE \| CHEMICAL"` |
| `entity_count` | `int` | No | Number of entities in this turn (0 if none) |

**Note:** `extracted_entities` and `entity_labels` are parallel pipe-separated lists. When `entity_count == 0`, both columns are empty strings (not NaN in the saved file).

---

## 3. Source Dataset Breakdown

| Source Dataset | Turns | Unique Dialogues | Entity Coverage (% turns with ≥1 entity) |
|---|---|---|---|
| MedDialog | 224,328 | 112,164 | 87.7% |
| HealthChat-LMSYS | 28,183 | 6,828 | 56.8% |
| HealthChat-WildChat | 23,128 | 4,264 | 61.2% |
| **Total** | **275,639** | **123,256** | **82.4%** |

MedDialog dominates the dataset (~81% of turns).

### HealthChat Sources — Nature of Data

**HealthChat-LMSYS** and **HealthChat-WildChat** are subsets of real user–LLM conversations drawn from the [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) and [WildChat](https://huggingface.co/datasets/allenai/WildChat-nontoxic) corpora, filtered for health-related topics. These are **real conversations between humans and LLMs** (primarily GPT-3.5/GPT-4 and Vicuna-class models) collected from public web interfaces. They are not simulated doctor–patient dialogues.

Key properties:
- Users sought healthcare information voluntarily from general-purpose LLMs, not from a medical-domain chatbot
- The LLM responses are not clinically validated
- Person names are redacted (replaced by `NAME_1`, `NAME_2`, etc.) in LMSYS-Chat-1M
- Some turns contain LLM-generated markdown tables, code blocks, and structured outputs that produce noisy NER results (table cell content may be extracted as entities)

This is distinct from MedDialog, which contains real doctor–patient Q&A from HealthCareMagic.com and iCliniq.com.

---

## 4. Speaker Distribution

| Speaker | Turns | % |
|---|---|---|
| assistant | 137,880 | 50.0% |
| user | 137,759 | 50.0% |

The dataset is essentially balanced between user and assistant turns.

---

## 5. Entity Statistics

### 5.1 Label Distribution

| Label | Mentions | % |
|---|---|---|
| DISEASE | 591,626 | 75.9% |
| CHEMICAL | 187,153 | 24.1% |
| **Total** | **778,779** | **100%** |

All labels in this file are either `DISEASE` or `CHEMICAL`. For an extended label set (SYMPTOM, ANATOMY, BEHAVIOR) see `integrated_conversations_clean.csv` (§11).

### 5.2 Entity Count per Turn

| Statistic | Value |
|---|---|
| Mean entities / turn | 2.83 |
| Std dev | 2.94 |
| Min | 0 |
| 25th percentile | 1 |
| Median | 2 |
| 75th percentile | 4 |
| Max | 122 |

### 5.3 Unique Entity Counts

| Label | Unique Entity Strings |
|---|---|
| DISEASE | 50,208 |
| CHEMICAL | 19,660 |
| **Total** | **69,039** (raw, case-sensitive — includes `"pain"`, `"Pain"`, `"pains"` as separate entries) |

### 5.4 Top 20 Disease Entities

| Entity | Mentions |
|---|---|
| pain | 51,228 |
| infection | 17,804 |
| fever | 11,848 |
| anxiety | 9,389 |
| swelling | 8,728 |
| cough | 7,405 |
| cancer | 6,913 |
| inflammation | 6,778 |
| bleeding | 6,301 |
| tooth | 5,174 |
| depression | 4,441 |
| diabetes | 4,438 |
| chest pain | 4,168 |
| headache | 4,152 |
| allergy | 4,015 |
| sore | 3,885 |
| throat | 3,851 |
| rash | 3,844 |
| pains | 3,829 |
| infections | 3,438 |

### 5.5 Top 20 Chemical Entities

| Entity | Mentions |
|---|---|
| alcohol | 5,653 |
| ibuprofen | 3,292 |
| smoking | 3,020 |
| calcium | 2,647 |
| iron | 2,564 |
| paracetamol | 2,470 |
| steroid | 2,329 |
| cholesterol | 2,259 |
| steroids | 1,823 |
| vitamins | 1,791 |
| glucose | 1,658 |
| oxygen | 1,640 |
| creatinine | 1,634 |
| smoke | 1,521 |
| vitamin d | 1,414 |
| chatdoctori | 1,399 ⚠ |
| antihistamines | 1,378 |
| progesterone | 1,350 |
| antihistamine | 1,283 |
| bilirubin | 1,268 |

> ⚠ `"chatdoctori"` is a known artefact from ChatDoctor template text embedded in the MedDialog source. Filtered in `integrated_conversations_clean.csv`.

---

## 6. Dialogue Structure

| Statistic | Value |
|---|---|
| Mean `turn_id` per row | 1.35 |
| Max turns in a single dialogue | 99 |
| Avg max turn index per dialogue | 1.24 |

Most dialogues are short (2–4 turns). A small number of dialogues in HealthChat-LMSYS/WildChat have up to 99 turns, reflecting the multi-turn nature of user–LLM sessions.

---

## 7. Validation Pipeline Summary

The validated file was produced through the following steps in `05_ner_validation.ipynb`:

### Step 1 — Schema & Integrity Check
- All 8 expected columns present.
- **198 null utterances** identified (turns with a valid `dialogue_id` but no utterance text).
- 46,889 rows without any extracted entities (entity columns empty, not null).
- 35 rows had **misaligned entity/label counts** — fixed during pipe-split parsing.

### Step 2 — Duplicate Utterance Removal
- Detected **3,254 unique utterance strings** that appeared in more than one row (case-normalised exact match).
- Kept first occurrence per unique utterance text.
- **533 rows removed** (0.19% of total) after deduplication.
- Rows before: 276,172 → Rows after: 275,639.

### Step 3 — Noisy Entity Removal (rule-based)
Entities flagged and removed if they matched any of these rules:

| Rule | Rationale |
|---|---|
| Numeric-only token (e.g. `"2"`, `"10mg"`) | Not a clinical entity |
| Token length < 2 characters | Stop-words / parsing artefacts |
| Common English stop-word (`"hi"`, `"yes"`, `"no"`, etc.) | False positives |
| Pure punctuation / whitespace | Parsing artefacts |
| Pure number or percentage | Not meaningful as entity |

- Entity mentions before cleaning: **1,027,487** (raw from NER pipeline)
- Entity mentions removed by rule-based filter: **18,229** (**1.77%** of raw)
- Entity mentions after rule-based filter: **1,009,258**

### Step 4 — Intra-row Duplicate Entity Deduplication
- Removed duplicate entity mentions *within the same turn* (e.g. `"OCD | OCD"` → `"OCD"`).
- **230,479 intra-row duplicate pairs removed** — this is where the majority of noise is eliminated.
- Final entity count: **778,779**
- Overall reduction from raw (1,027,487) to final (778,779): **24.20%**

### Validation Score Card

```
✓ All entity labels valid (DISEASE / CHEMICAL only)
⚠ 35 rows: misaligned entity/label counts → fixed
⚠ 3,254 exact duplicate utterances detected; 533 rows removed
⚠ 1.77% of raw entity mentions removed by rule-based noise filter
⚠ 24.20% overall reduction (rule-based + intra-row dedup combined)
```

---

## 8. Known Issues & Caveats

| Issue | Detail |
|---|---|
| `chatdoctori` artefact | Appears ~1,400 times as a CHEMICAL entity; sourced from ChatDoctor template text in MedDialog. Filtered in `integrated_conversations_clean.csv`. |
| BC5CDR label scope | `en_ner_bc5cdr_md` only knows DISEASE and CHEMICAL. Symptom terms (`pain`, `cough`, `fever`), anatomy terms (`tooth`, `throat`, `liver`) and behaviours (`smoking`, `alcohol`) are mislabelled or forced into the wrong category. Corrected in `integrated_conversations_clean.csv` via rule-based relabelling + optional UMLS linking. |
| Morphological variants | `"pain"` / `"pains"`, `"infection"` / `"infections"` are counted separately. Normalised in `integrated_conversations_clean.csv`. |
| Case variants | `"Pain"` and `"pain"` are separate entries. Lowercased in `integrated_conversations_clean.csv`. |
| Null utterances | **198 turns** have a null `utterance` but retain a valid `dialogue_id`. Drop these rows before any text-based modelling. |
| Markdown / table artefacts | Some HealthChat-LMSYS/WildChat turns are LLM outputs with markdown tables; NER may extract table cell content as entities (e.g. `"Yes | Yes |\n| Hormonal contraceptives"`). |
| Mixed types in `original_id` | Column has mixed types (DtypeWarning on load). Always load with `dtype={'original_id': str, 'dialogue_id': str}`. |
| Paraphrased template duplicates | Exact dedup (Step 2 above) removed 533 rows but does not catch near-identical paraphrases of ChatDoctor canned responses. MinHash LSH near-dedup is applied in `06_cleaning_pipeline.ipynb`. |

---

## 9. How to Load

```python
import pandas as pd

df = pd.read_csv(
    "data/processed/integrated_conversations_ner_validated.csv",
    low_memory=False,
    dtype={"original_id": str, "dialogue_id": str}
)

# Drop the 198 null utterance rows if text is required
df = df.dropna(subset=["utterance"])

# Parse entity columns into lists
df["ents_list"]   = df["extracted_entities"].fillna("").apply(
    lambda x: [e.strip() for e in x.split("|") if e.strip()]
)
df["labels_list"] = df["entity_labels"].fillna("").apply(
    lambda x: [l.strip() for l in x.split("|") if l.strip()]
)
```

---

## 10. Suggested Next Steps

| Step | Description |
|---|---|
| **Use clean file** | Load `integrated_conversations_clean.csv` (from `06_cleaning_pipeline.ipynb`) instead of this file for modelling. It includes template filtering, near-dedup, normalised entities, and SYMPTOM/ANATOMY/BEHAVIOR labels. |
| **Manual NER review** | Open `data/processed/spotcheck_200.csv` and fill the `_review_label` (CORRECT / WRONG / PARTIAL) and `_review_notes` columns for ≥200 turns to estimate NER precision. |
| **UMLS grounding** | Install `scispacy==0.5.5` and run Stage D2 of notebook 06 to add UMLS CUIs to every entity. Enables MeSH / SNOMED CT lookup. |
| **Embedding / retrieval** | Use `utterance` + entity context for dense retrieval or RAG pipelines. Entity lists serve as structured metadata filters. |
| **Intent classification** | Classify user turns by symptom/condition focus using the DISEASE / SYMPTOM entities. |
| **Dialogue modelling** | Group by `dialogue_id` + sort by `turn_id` to reconstruct full multi-turn conversations. |
| **Train/val/test split** | Always split on `dialogue_id` (not row index) to avoid turn-level leakage across splits. |

---

## 11. Successor File: `integrated_conversations_clean.csv`

> **File path:** `data/processed/integrated_conversations_clean.csv`
> **Produced by:** `notebooks/06_cleaning_pipeline.ipynb`

Applies five additional cleaning stages on top of this file:

| Stage | Description | Output |
|---|---|---|
| **A — Template filtering** | Strips ChatDoctor sign-offs (`"Hope I have answered your query"`, `"Thanks for your query"`, `"chat doctor."` suffixes, MedDialog greetings) inline; drops whole-utterance boilerplate (`▌`, `hi`, canned LLM openers) | Fewer rows, cleaner text |
| **B — Near-duplicate detection** | MinHash LSH (128 perms, Jaccard ≥ 0.85, char 5-grams) clusters paraphrased templates; keeps longest turn per cluster | Removes template variants missed by exact dedup |
| **C — Entity normalisation** | Lowercase + spaCy `en_core_web_sm` lemmatisation collapses `pain/pains/Pain` → `pain`, `steroids` → `steroid`, etc.; saves `entity_norm_map.csv` | Reduced unique entity count; better frequency counts |
| **D — NER label improvement** | Rule-based SYMPTOM / ANATOMY / BEHAVIOR vocabulary applied to BC5CDR mismatches; optional UMLS EntityLinker adds `umls_cuis` column | Extended label set; `smoking`/`alcohol` → BEHAVIOR, `tooth`/`throat` → ANATOMY, `pain`/`fever` → SYMPTOM |
| **E — Spot-check sample** | 200-turn stratified sample (by source, speaker, entity presence) saved to `spotcheck_200.csv` with blank `_review_label` / `_review_notes` columns | Ready for manual NER quality review |

---

## 12. License & Privacy Notes

> ⚠ **Review these before any commercial use or public distribution.**

| Source | License / Terms | Commercial Use | Key Restrictions |
|---|---|---|---|
| **MedDialog** (HealthCareMagic-100k) | No open license declared. Raw dialogues from [healthcaremagic.com](https://healthcaremagic.com) and [icliniq.com](https://icliniq.com). Copyright belongs to those platforms. | **Not permitted** without explicit consent from the copyright holders. | Research-only use implied by the dataset paper. Do not redistribute raw text. |
| **HealthChat-LMSYS** (LMSYS-Chat-1M subset) | [LMSYS-Chat-1M Dataset License Agreement](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) — custom, must be accepted on HuggingFace. | **Permitted** (research and commercial) subject to the agreement. | No redistribution; no re-identification of individuals; LMSYS may require deletion at any time; model-specific terms apply to outputs from individual LLMs. |
| **HealthChat-WildChat** (WildChat subset) | [AI2 ImpACT License — Low Risk](https://allenai.org/licenses/impact-lr) | **Permitted** with attribution. | Harmful use prohibited; must cite the WildChat paper; AI2 may update terms. |

**Privacy:** LMSYS-Chat-1M has person names redacted as `NAME_1`, `NAME_2` etc. WildChat conversations were collected with explicit user opt-in. Neither dataset should be used to attempt re-identification of individuals.

**Medical disclaimer:** None of the content in HealthChat-LMSYS or HealthChat-WildChat constitutes validated clinical advice. LLM responses are not reviewed by medical professionals.
