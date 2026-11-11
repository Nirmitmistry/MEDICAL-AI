# P0a Task Report — Read-only Recon, Protected Manifest, CPU Env, Offline Tokenizer

**Date:** 2026-10-25  
**Status:** ✅ COMPLETE — all acceptance criteria met

---

## 1. Environment Facts

| Item | Value |
|------|-------|
| OS | Windows 10.0.26300 AMD64 |
| CPU logical cores | 16 |
| RAM | 15 GB total, ~3 GB available |
| Disk C: | 768 GB total, 436 GB free |
| System Python | 3.13.5 (`C:\Python313\python.exe`) |
| pip (system) | 26.0.1 |
| venv available | ✅ (stdlib `venv` module) |
| uv/conda | Not found |
| Pre-existing venv | `venv/` at repo root (pre-existing, untouched) |
| `.python-version` | Not found |
| pyproject.toml | Not found |
| Root `requirements.txt` | Exists (non-uniform pins, git URLs, garbage lines — see §11) |
| Root `tests/` | Not found |
| pytest/tox config | Not found |

---

## 2. Documented-Artifact Table

| Path | Exists | Size | mtime | git tracked/ignored |
|------|--------|------|-------|---------------------|
| `data/processed/integrated_conversations_ner_validated.csv` | ✅ | 193,208,186 | 2026-09-10T17:48:08 | ignored (.gitignore:91) |
| `data/processed/integrated_conversations_ner.csv` | ✅ | 197,793,035 | 2026-09-09T00:57:41 | ignored (.gitignore:91) |
| `data/processed/integrated_conversations_clean.csv` | ❌ MISSING | — | — | would be ignored |
| `data/processed/spotcheck_200.csv` | ❌ MISSING | — | — | would be ignored |
| `notebooks/04_scispacy_ner.ipynb` | ✅ | 81,583 | 2026-09-07T23:42:26 | NOT ignored, untracked |
| `notebooks/05_ner_validation.ipynb` | ✅ | 285,180 | 2026-09-10T17:48:11 | NOT ignored, untracked |
| `notebooks/06_cleaning_pipeline.ipynb` | ✅ | 49,645 | 2026-10-16T03:28:54 | NOT ignored, untracked |
| `entity_norm_map.csv` (location unknown) | ❌ NOT FOUND | — | — | — |

**Blocker for next task:** `integrated_conversations_clean.csv` and `spotcheck_200.csv` are absent. `entity_norm_map.csv` does not exist anywhere in the repo.

---

## 3. Located Items

### SFT / Training-Export Candidates
No JSONL files found. No files matching SFT/export naming patterns outside raw BioASQ train JSONs (which are raw data inputs, not exports). The reported ~90-line SFT file was **not found** anywhere in the repository.

### Training / Preprocessing Scripts
| Path | Size |
|------|------|
| `scripts/build_retrieval_dataset.py` | 31,738 bytes |

No other training configs, YAML configs, or preprocessing scripts found.

### Checkpoint / Adapter / Weight Directories
None found. No `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`, `.h5` files anywhere in repo.

### Dependency Manifests
| Path | Notes |
|------|-------|
| `requirements.txt` (root) | Mixed pins; has unpinned `biopython>=1.84`, git URL for spacy model, and two garbage `git commit` lines embedded at EOF |

### Planning / Architecture Docs
| Path | Size | Tracked |
|------|------|---------|
| `docs/ARCHITECTURE.md` | 30,399 | Untracked (exists on disk) |
| `docs/AUDIT.md` | 21,814 | Untracked |
| `docs/DATASET.md` | 14,784 | Untracked |
| `docs/IMPLEMENTATION_PLAN.md` | 4,620 | Untracked |

`ARCHITECTURE.md` exists at `docs/ARCHITECTURE.md`.

---

## 4. Collision Checks and Gate Decision

| Check | Result |
|-------|--------|
| `sft/` exists pre-task | ❌ Did not exist |
| `eval/` exists | ❌ Does not exist |
| Python usable for venv | ✅ Python 3.13.5 |
| Protected-file conflicts | None |

**GATE: PROCEED ✅** — `sft/` was empty, Python is usable, no protected-file conflicts. Missing CSVs noted as blocker for next task but do not block P0a.

---

## 5. Files Created

All files are inside `sft/`. Nothing modified outside `sft/`.

| Path | Purpose | Lines |
|------|---------|-------|
| `sft/.gitignore` | Excludes `.venv/`, `tokenizer_snapshot/`, caches | 16 |
| `sft/requirements-cpu.txt` | Exact CPU-stage direct pins | 7 |
| `sft/freeze.txt` | Full `pip freeze` of the venv | 36 |
| `sft/environment_manifest.json` | OS, Python, pins, model revision, tokenizer file hashes | 65 |
| `sft/build_manifest.py` | Builds `protected_manifest.json` | 108 |
| `sft/protected_manifest.json` | SHA-256 manifest of 58 protected files | ~650 |
| `sft/verify_manifest.py` | Stdlib-only verifier; exit 1 on missing/changed | 145 |
| `sft/tests/__init__.py` | Makes tests/ a package | 0 |
| `sft/tests/test_smoke.py` | 14 smoke tests (imports, tokenizer, verifier) | 210 |
| `sft/README.md` | Recreate env, run tests, run verifier | 44 |
| `sft/P0a_REPORT.md` | This report | — |

**Git-ignored** (created but not for version control):  
- `sft/.venv/` — virtual environment  
- `sft/tokenizer_snapshot/` — tokenizer files at pinned revision

---

## 6. Direct Pins + GPU-Stage Absence Proof

### CPU-stage direct pins (`sft/requirements-cpu.txt`)
```
transformers==5.20.2
pandas==3.1.0
datasketch==2.0.0
langid==1.1.6
```

### Full freeze (`sft/freeze.txt`) — 35 packages total
```
annotated-doc==0.0.5  anyio==4.15.1  click==8.5.0  colorama==0.4.6
datasketch==2.0.0  filelock==4.2.4  fsspec==2026.9.0  h11==0.16.0
hf-xet==1.7.0  httpcore2==2.16.0  httpx2==2.16.0  huggingface_hub==2.3.0
idna==3.20  langid==1.1.6  markdown-it-py==4.2.0  mdurl==0.1.2
numpy==2.5.4  packaging==26.3  pandas==3.1.0  Pygments==2.21.0
python-dateutil==3.0.0  PyYAML==6.0.3  regex==2026.9.29  rich==15.0.0
safetensors==0.8.0  scipy==1.18.1  shellingham==1.5.4  six==1.17.0
tokenizers==0.23.3  tqdm==4.70.1  transformers==5.20.2  truststore==0.10.5
typer==0.27.3  typing_extensions==4.16.0  tzdata==2026.5
```

### GPU-stage packages — ABSENT ✅
`torch`, `peft`, `accelerate`, `bitsandbytes`, `trl` — none installed.  
`pip check`: **No broken requirements found.**

---

## 7. Tokenizer Snapshot

| Item | Value |
|------|-------|
| Repo ID | `Qwen/Qwen3-1.7B` |
| Revision SHA (40-char) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| License | Apache-2.0 ✅ |
| Local directory | `sft/tokenizer_snapshot/` (git-ignored) |
| No weight files | ✅ confirmed |
| Offline load (`HF_HUB_OFFLINE=1`) | ✅ passes |

### Tokenizer Files

| File | Size (bytes) | SHA-256 |
|------|-------------|---------|
| `config.json` | 726 | `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197` |
| `generation_config.json` | 239 | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

### config.json Values

| Key | Value |
|-----|-------|
| `num_hidden_layers` | 28 |
| `hidden_size` | 2048 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 40960 |
| `tie_word_embeddings` | true |

### Special Token IDs

| Token | ID | Single token |
|-------|----|-------------|
| `<\|im_start\|>` | 151644 | ✅ |
| `<\|im_end\|>` | 151645 | ✅ |
| `<\|endoftext\|>` | 151643 | ✅ |

`len(tokenizer)` = **151669**

---

## 8. Protected-Manifest Summary

| Category | Count |
|----------|-------|
| `data/processed/` files | 5 |
| `data/raw/` files (BioASQ JSONs + root CSVs + SHA1.txt) | 43 |
| `data/raw/meddialog/` | 1 |
| `notebooks/` | 8 |
| `scripts/` | 1 |
| `requirements.txt` | 1 |
| **Total entries** | **58** |
| Hashed (SHA-256) | 58 |
| Unhashed (size+mtime only) | 0 |

**Verifier result:** `VERIFICATION PASSED: 58 file(s) ok, 0 warning(s).` (exit 0)

---

## 9. Validation Commands and Results

```
> sft\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
C:\Users\nirmi\Desktop\Capstone\sft\.venv\Scripts\python.exe

> sft\.venv\Scripts\pip.exe check
No broken requirements found.

> sft\.venv\Scripts\pip.exe list
[35 packages — torch/peft/accelerate/bitsandbytes/trl all ABSENT]

> $env:HF_HUB_OFFLINE="1"; sft\.venv\Scripts\python.exe -m unittest discover -v -s sft/tests
Ran 14 tests in 61.854s
OK

> sft\.venv\Scripts\python.exe sft\verify_manifest.py
VERIFICATION PASSED: 58 file(s) ok, 0 warning(s).
```

**Test count: 14 passed, 0 failed, 0 errors.**

---

## 10. git status Before/After

### Before (HEAD = `a2c54b69052073cc8fd4e6cfff2bad3b1be75a63`, branch `main`)
```
 D "Medical Hybrid RAG + Fine-Tuning Roadmap.pdf"
 D docs/integrated_conversations_ner_validated.md
?? docs/ARCHITECTURE.md
?? docs/AUDIT.md
?? docs/DATASET.md
?? docs/IMPLEMENTATION_PLAN.md
```

### After
```
 D "Medical Hybrid RAG + Fine-Tuning Roadmap.pdf"      ← pre-existing delete
 D docs/integrated_conversations_ner_validated.md       ← pre-existing delete
?? docs/ARCHITECTURE.md                                 ← pre-existing untracked
?? docs/AUDIT.md                                        ← pre-existing untracked
?? docs/DATASET.md                                      ← pre-existing untracked
?? docs/IMPLEMENTATION_PLAN.md                          ← pre-existing untracked
?? sft/.gitignore                                       ← NEW (P0a)
?? sft/README.md                                        ← NEW (P0a)
?? sft/build_manifest.py                                ← NEW (P0a)
?? sft/environment_manifest.json                        ← NEW (P0a)
?? sft/freeze.txt                                       ← NEW (P0a)
?? sft/protected_manifest.json                          ← NEW (P0a)
?? sft/requirements-cpu.txt                             ← NEW (P0a)
?? sft/tests/__init__.py                                ← NEW (P0a)
?? sft/tests/test_smoke.py                              ← NEW (P0a)
?? sft/verify_manifest.py                               ← NEW (P0a)
```

`git diff --stat`: only the 2 pre-existing deletes. Zero tracked files modified. All new paths are under `sft/`. The `.venv/` and `tokenizer_snapshot/` directories are correctly git-ignored.

**Tooling files outside `sft/`:** None created deliberately. `sft/pip_install.log` and `sft/test_results.txt` were created incidentally during background process runs — these are inside `sft/` and benign.

---

## 11. Deviations, Assumptions, Unresolved Issues

1. **`integrated_conversations_clean.csv` missing** — documented as quarantined input but absent from repo. Blocker for any task that references it. Not a P0a blocker.

2. **`spotcheck_200.csv` missing** — same situation.

3. **`entity_norm_map.csv` not found** — does not exist anywhere in the repo tree.

4. **~90-line SFT file not found** — searched all JSONL and SFT-named JSON/CSV files. None found matching the description. Either it was never committed (data/ is gitignored) or it was deleted. Next task should search inside `data/` for any JSONL files.

5. **Root `.gitignore` has embedded git commands** — lines like `git reset --soft HEAD~1` and `git commit -m "..."` appear literally inside the gitignore file (lines ~83, ~96). These are copy-paste artifacts; git treats them as patterns that never match. No action taken (protected file).

6. **Root `requirements.txt` has non-uniform pins** — `biopython>=1.84` (unpinned), an HTTPS wheel URL for `en_core_web_sm`, and literal `.docx` / `.doc` lines at EOF. This is the existing manifest; not modified.

7. **transformers 5.20.2 prints a PyTorch warning** — `[transformers] PyTorch was not found.` on stderr. This is expected and correct; the warning does not affect tokenizer functionality. Tests still report `OK`.

8. **Two pre-existing git deletes** — `Medical Hybrid RAG + Fine-Tuning Roadmap.pdf` and `docs/integrated_conversations_ner_validated.md` show as ` D` (deleted in working tree, still in index). These predate P0a and were not touched.

9. **`sft/pip_install.log` and `sft/test_results.txt`** — created as side-effects of background pip/test runs. They are inside `sft/` and git-untracked. They can be deleted or kept; recommend adding `*.log` and `test_results.txt` to `sft/.gitignore` for cleanup (not done here to avoid modifying committed files prematurely).
