# Repository and Dataset Forensic Audit

## Working assumption

For this report, **assume `data/processed/integrated_conversations_ner_validated.csv` is the dataset used by the current training pipeline**.

This is a working assumption supplied for analysis, not a fact independently verified from the training script. The repository, CSV itself, notebooks, SFT export, and training configuration were not available in the audit environment. Therefore, all counts attributed to the CSV below are **documented counts from `integrated_conversations_ner_validated.md`**, not fresh measurements of the underlying data.

The assumption also needs a precise interpretation:

- **Direct-consumption scenario:** the training loader reads `integrated_conversations_ner_validated.csv` directly.
- **Export scenario:** the validated CSV is the source from which a separate SFT file is produced, and the training loader consumes that export.

The first scenario is compatible with a 275,639-row CSV as documented. The second can also be compatible with the assumption, but the exported file and conversion code must be inspected. A reported file of approximately 90 physical lines is not established as a complete one-row-per-turn export.

---

## A. Confirmed repository and dataset inventory

### A1. Accessible evidence

The only project-specific artifact available for this audit was the documentation file `integrated_conversations_ner_validated.md`. No repository tree, actual dataset CSV, preprocessing notebook, training script, SFT file, split file, or dependency manifest was accessible. No files were modified and no training was launched.

**Evidence status:** The inventory below distinguishes documentation from verified file presence.

| Artifact | Documented path/name | Documented purpose | Audit status |
|---|---|---|---|
| Raw source data | Exact paths not provided | MedDialog, HealthChat-LMSYS, HealthChat-WildChat sources | Not accessible; filenames and counts not verified |
| Integrated NER data | `data/processed/integrated_conversations_ner.csv` | Output of `notebooks/04_scispacy_ner.ipynb` | Documented; actual file not accessible |
| Validated NER data | `data/processed/integrated_conversations_ner_validated.csv` | Output of `notebooks/05_ner_validation.ipynb` | **Assumed training source**; actual file not accessible |
| Cleaned conversation data | `data/processed/integrated_conversations_clean.csv` | Output of `notebooks/06_cleaning_pipeline.ipynb` | Documented successor; not assumed to be used for training |
| Entity normalization map | `entity_norm_map.csv` | Entity normalization output | Documented; actual file/path not verified |
| NER spot-check file | `spotcheck_200.csv` | Manual NER review sample | Documented; actual file not accessible |
| SFT export | Unknown | Possible transformed training input | Filename, format, and existence unverified |
| Train/validation/test splits | Unknown | Data partitions for training/evaluation | Not accessible or verified |
| Training scripts/configuration | Unknown | Loads dataset and performs training | Not accessible; actual data path unverified |
| Dependency manifest | Unknown | Specifies project packages | Not accessible |

### A2. Dataset statistics documented for the assumed source

The documentation describes the validated CSV as containing:

| Property | Documented value |
|---|---:|
| Turn rows | 275,639 |
| Unique dialogues | 123,256 |
| Assistant turns | 137,880 |
| User turns | 137,759 |
| Null `utterance` values | 198 |
| Rows with empty entity columns / no extracted entities | 46,889 |
| Entity mentions | 778,779 |
| Source datasets | MedDialog, HealthChat-LMSYS, HealthChat-WildChat |

These values are internally consistent in the following basic checks: 137,880 assistant turns + 137,759 user turns = 275,639 rows, and the documented source-dataset turn counts also add up to 275,639. These arithmetic checks do **not** independently verify the CSV.

### A3. Audit environment

The session environment observed for the prior audit was Linux x86-64, Python 3.13.5, approximately 5.8 GiB RAM, three logical CPUs, CPU-only PyTorch, and no accessible NVIDIA GPU. Transformers, Hugging Face Datasets, PEFT, TRL, and Accelerate were not installed in that environment. No project dependency file was available.

**Limitation:** These observations describe the audit session only; they do not establish the hardware or packages on the user's development machine or on the machine where training may run.

---

## B. Data flow from raw data to the assumed training source

### B1. Flow documented by the dataset reference

```text
Raw source datasets
    |
    | Original integration step: details/path not available
    v
Integrated conversation data
    |
    | notebooks/04_scispacy_ner.ipynb
    v
data/processed/integrated_conversations_ner.csv
    |
    | notebooks/05_ner_validation.ipynb
    | - schema/integrity checks
    | - exact duplicate utterance removal
    | - NER entity filtering
    | - duplicate entities removed within a turn
    v
data/processed/integrated_conversations_ner_validated.csv
    |
    | ASSUMED CURRENT TRAINING SOURCE
    |
    +---- Direct-consumption scenario: training loader reads this CSV
    |
    +---- Export scenario: an unverified conversion creates an SFT file
                                      |
                                      v
                              Current training loader
                              (path and code unverified)

Separate documented branch:
data/processed/integrated_conversations_ner_validated.csv
    |
    | notebooks/06_cleaning_pipeline.ipynb
    v
data/processed/integrated_conversations_clean.csv
```

The documentation identifies the validated file's input and producer and describes the cleaned file as its successor. It does not prove which path the live training script loads.

### B2. What the assumption changes

For the purpose of this report, downstream data auditing should treat the validated CSV as the likely source rather than assuming that the more heavily cleaned successor is being used.

That means the documented validation-stage issues remain relevant:

- 198 null utterances remain documented in the validated dataset.
- Exact duplicate utterance deduplication reportedly removed 533 rows from 276,172 to 275,639.
- The documented validation pipeline does not perform the later near-duplicate template cleanup described for the successor file.
- Known `chatdoctori` template artefacts and the limitations of the DISEASE/CHEMICAL-only NER labels remain in the validated dataset according to the documentation.

If an SFT export is created from this CSV, its conversion process may remove or transform records. Therefore, the assumption alone does not establish that all 275,639 rows become training examples.

### B3. The approximately 90-line discrepancy

The approximately 90-line file reported in the task remains unidentified and unavailable. Under the working assumption, there are two unresolved possibilities:

1. **The validated CSV is loaded directly.** In that case, the documented source is a 275,639-row CSV; the approximately 90-line file may be unrelated to the active loader, a preview/sample, or a different artifact.
2. **The validated CSV feeds an SFT export.** In that case, the approximately 90-line file could be the export, but its format and intended unit of representation must be checked before calling it truncated.

A physical line count is not necessarily a logical record count: CSV quoted fields can contain line breaks, and a JSON/JSONL conversion may use one record per dialogue or one record per example. Nevertheless, if the 90-line file is intended to be a complete **one-row-per-turn** export of the documented source, it does not reconcile with 275,639 documented source rows.

**Conclusion:** The small-file discrepancy is not resolved by the assumption. The current loader path and the file's logical record count are required to resolve it.

### B4. Roles, history, identifiers, entities, and special tokens

| Question | What is documented | What remains unknown |
|---|---|---|
| Roles | `speaker` uses `user` or `assistant` | Whether the training converter maps roles correctly |
| Conversation history | `dialogue_id` and `turn_id` are present in the validated schema | Whether the SFT export groups and orders turns into dialogues or trains on isolated rows |
| Identifiers | `dialogue_id` is a string identifier; `turn_id` is a zero-indexed position | Whether both survive downstream preprocessing/export |
| Entity columns | Pipe-delimited `extracted_entities` and `entity_labels`, plus `entity_count` | Whether these columns are ignored, serialized as text, or used as auxiliary targets/features |
| Special tokens | No training-token convention is specified in the supplied documentation | Whether message boundaries and any model-specific tokens match the chosen tokenizer/template |
| Assistant targets | Assistant rows are counted in the documentation | Whether only assistant messages contribute to loss, and whether each row becomes an example |

No claims about the current SFT serialization can be verified without the data converter and training loader.

### B5. Dialogue integrity risk

The documented validation pipeline removes duplicate utterance text globally, keeping the first occurrence. The later cleaning notebook is described as removing whole boilerplate turns and near-duplicate utterance clusters. These operations can be appropriate for text cleaning but may break conversation context if they remove individual turns without dialogue-aware safeguards.

Whether the existing code preserves `dialogue_id`/`turn_id`, removes complete conversations, renumbers turns, or leaves gaps is **not verified**. This must be checked against the actual notebook and output data; it cannot be inferred from the documentation alone.

---

## C. Expected-versus-observed reconciliation table

In this table, “documented” means reported by the reference document, “derived” means simple arithmetic from its figures, and “observed” means measured from actual files in the audit environment. No underlying dataset was accessible, so no dataset count below qualifies as independently observed.

| Measure | Expected / documented | Independently observed | Status / implication |
|---|---:|---:|---|
| Validated CSV rows | 275,639 | Not measurable | Documented only |
| Unique dialogues | 123,256 | Not measurable | Documented only |
| Assistant turns | 137,880 | Not measurable | Documented only |
| User turns | 137,759 | Not measurable | Documented only |
| Null utterances | 198 | Not measurable | Documented only |
| Rows without extracted entities | 46,889 | Not measurable | Documented only; not automatically invalid for text-only SFT |
| Rows before exact-utterance deduplication | 276,172 | Not measurable | Documented only |
| Rows removed by exact-utterance deduplication | 533 | Not measurable | Documented only |
| Rows after documented deduplication | 275,639 | Not measurable | Documented only |
| Assistant turns after removing null-utterance rows | Not reported | Not measurable | Could be between 137,682 and 137,880 based solely on 198 nulls, before any other filters; actual role/null overlap unknown |
| Complete dialogues after cleaning | Not reported | Not measurable | Requires inspection of actual ordered turns |
| Valid assistant targets | Not reported | Not measurable | Depends on example construction, null handling, and target policy |
| Empty SFT records | Not reported | Not measurable | SFT file unavailable |
| Valid SFT examples | Not reported | Not measurable | Cannot assume one record per source row |
| Approx. 90-line file | Reported in task; path unknown | Not verified | Identity, format, and logical record count unknown |
| Train/validation/test split sizes | Not reported | Not measurable | Split files/configuration unavailable |

**Interpretation of the assistant-target bound:** The documented 137,880 assistant-turn count is an upper bound on potential assistant-message targets before removing null-utterance rows or applying further filtering, assuming targets originate from assistant turns. Because the roles of the 198 null rows are unknown, subtracting all 198 gives a conservative lower bound of 137,682 for assistant turns remaining after *only* null-utterance removal. This is not a count of valid SFT examples.

The documented ratio is approximately 2.24 turn rows per dialogue (275,639 / 123,256). This is a derived average, not proof that every dialogue is complete or that the SFT converter preserves the dialogues.

---

## D. Critical defects ranked by severity

### P0 — Training source and small-file mismatch are not proven

**Evidence:** The working assumption identifies the validated CSV as the source, but neither its actual bytes nor the current loader/configuration nor the approximately 90-line file were available.

**Impact:** It remains possible that the active training code loads another file, or that the source is converted into an incomplete/incorrect export. Until the loader path is confirmed, the effective training dataset size is unknown.

### P1 — Conversation integrity may be damaged by row-level deduplication

**Documented operation:** Exact utterance duplicates were removed globally, keeping the first occurrence. Additional template and near-duplicate filtering belongs to the documented successor pipeline, not necessarily the assumed validated-data path.

**Risk:** Repeated utterances can be legitimate in different dialogues. Removing individual rows can make later context incomplete. The actual rate and impact of such damage are unknown until the data and script are inspected.

### P1 — Evaluation leakage has not been ruled out

The documentation recommends splitting on `dialogue_id`, not row index, to avoid turn-level leakage. No split files or split-generation code were available. Consequently, it is unknown whether turns from the same dialogue or duplicate/near-duplicate text appears across train, validation, and test.

### P1 — Licensing, privacy, and medical safety need release review

The dataset documentation states that MedDialog has no open license declared and that commercial use requires permission from the relevant copyright holders. It identifies separate terms for LMSYS-Chat-1M and WildChat and warns against re-identification. The HealthChat assistant responses are not clinically validated.

Before external release, the project must establish the applicable terms for each source and for the intended derivative/model distribution, and assess residual personal/sensitive information and unsafe medical outputs. This audit does not constitute legal advice or a completed privacy review.

### P2 — Known template artefacts and NER-label limitations remain

The reference identifies the `chatdoctori` artefact and describes BC5CDR-based labels as limited to DISEASE and CHEMICAL, with known misclassifications involving symptoms, anatomy, and behaviours. These issues matter directly if entity labels are used in training; they matter less directly if entity columns are excluded and only utterance text is used. The current SFT mapping is unknown.

### P2 — Documentation/schema mismatch

The reference says “all 8 expected columns” while its schema table lists nine named columns. Treat this as a documentation discrepancy to check against the actual CSV header, not proof that the CSV itself is malformed.

### P2 — Project environment is unconfirmed

The audit session is CPU-only and lacks common fine-tuning packages, but no project-specific dependency manifest or target machine information is available. The compute environment used for training remains unknown.

---

## E. What can safely be reused, what needs repair, and what must not be deleted

### E1. Candidate artifacts for reuse after verification

| Artifact | Potential reuse | Required qualification |
|---|---|---|
| Dataset reference markdown | Baseline for checks and lineage | It is documentation, not the underlying data |
| `integrated_conversations_ner.csv` | Intermediate lineage/checkpoint | Verify actual file, schema, and provenance |
| `integrated_conversations_ner_validated.csv` | **Assumed current source** and baseline for data audit | Reproduce counts and inspect nulls, IDs, roles, and duplicate effects |
| `integrated_conversations_clean.csv` | Comparison point for later cleaning | Do not assume training uses it; compare only after confirming its existence and processing code |
| `entity_norm_map.csv` | Review entity normalization decisions | Verify mappings and provenance |
| `spotcheck_200.csv` | Manual NER quality-review workflow | Verify file and whether review was completed |
| Existing SFT/split files | Forensic comparison with source | Preserve first; establish which files the loader actually reads |

No dataset or script is certified as safe to use as-is because the actual artifacts were not accessible.

### E2. Repairs and checks needed

1. Confirm the actual input path used by the training entry point.
2. Inspect and count the validated CSV directly; reproduce its documented schema and counts.
3. Determine whether training consumes rows directly or an SFT export; identify the approximately 90-line file.
4. Trace how user/assistant turns are grouped, ordered, and transformed into assistant targets.
5. Verify that empty/null turns are handled and that `dialogue_id`/`turn_id` survive the process.
6. Measure whether exact duplicate removal has deleted legitimate turns from otherwise valid conversations.
7. Verify dialogue-level train/validation/test separation and assess duplicate/near-duplicate overlap.
8. Inspect role serialization, message boundaries, and tokenizer-specific special-token handling.
9. Confirm the target machine, accelerator availability, and actual project dependencies.
10. Review licenses, privacy, and safety conditions for the intended use and release plan.

### E3. Preserve these artifacts

Do not overwrite or delete the following until the current pipeline has been reconstructed and counts have been recorded:

- Raw source data or controlled provenance references.
- `integrated_conversations_ner.csv`, `integrated_conversations_ner_validated.csv`, and `integrated_conversations_clean.csv`, wherever present.
- The approximately 90-line SFT file and all other candidate SFT exports.
- Existing train/validation/test split files and split manifests.
- Notebooks 04–06, the training script, preprocessing/export scripts, configs, dependency files, and relevant logs.
- `entity_norm_map.csv`, `spotcheck_200.csv`, and other validation outputs.
- Available Git history, checksums, and provenance notes.

**Exception:** Applicable data-provider terms and legal deletion obligations take precedence. In particular, preserve only data that the project is permitted and required to retain.

---

## F. Smallest next steps required before architecture selection

1. **Provide the project files.** Include the repository tree, notebooks 04–06, the current training entry point/configuration, all preprocessing/export scripts, and dependency manifests. Include the actual validated CSV or a verifiable local inspection report if the file cannot be shared.
2. **Identify the small SFT file.** Provide its exact path and format. Determine physical lines, logical records, columns/keys, dialogue count, assistant-target count, and the script that produced it.
3. **Verify source counts.** Recompute turn rows, dialogues, role counts, null/empty utterances, IDs/turn ordering, and duplicate counts from the actual validated CSV. Record counts after every downstream filter.
4. **Trace the active loader.** Confirm whether it reads the validated CSV directly or an export, and confirm the actual paths used by the current training configuration.
5. **Audit evaluation and release constraints.** Verify dialogue-level split integrity and resolve the data licensing, privacy, and intended-distribution requirements.

### Blockers affecting the architecture decision

- The validated CSV is assumed to be the source but has not been independently inspected.
- The active training loader and exact input path are unknown.
- The approximately 90-line file has not been identified or reconciled.
- Complete dialogues and valid assistant-target counts have not been measured.
- Preservation of dialogue/turn IDs and conversation context through preprocessing is unverified.
- Role mappings, history assembly, SFT example structure, and special-token handling are unknown.
- Train/validation/test construction and leakage risk have not been checked.
- The target compute environment and project dependencies are unknown.
- Source-license, privacy, and model-distribution constraints remain to be resolved.

**Conclusion:** Under the stated assumption, the documented validated dataset contains 275,639 turn rows across 123,256 dialogue IDs, but those figures are not independently measured in this audit. If the CSV is loaded directly, the approximately 90-line file is not the complete one-row-per-turn source described by the documentation. If the CSV is converted to SFT first, the export procedure and logical record count must be examined. The effective training dataset and architecture constraints remain unconfirmed until the repository and actual files are accessible.

---

## Documentation basis

This report is based on the supplied file `integrated_conversations_ner_validated.md`, particularly its sections on the schema and dataset statistics, validation pipeline, known issues, successor cleaning pipeline, and license/privacy notes. The training-source assumption is supplied by the user for this report and is not independently verified.
