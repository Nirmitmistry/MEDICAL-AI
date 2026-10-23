"""
build_retrieval_dataset_v2.py
==============================
Produces v2 of data/processed/bioasq_pubmed_retrieval.csv
(plus .parquet and two derived files).

Goals implemented
-----------------
  G1  source_group + medquad_subsource + fix pubmed_topic (real PubMed joins only)
  G2  Text cleaning: strip HCM/ChatDoctor templates, normalise whitespace/unicode,
       flag questions < 5 or > 512 words in question_length_flag
  G3  PII removal/masking in HCM text (emails, phones, locations via regex);
       log changed rows
  G4  Deduplication: exact normalised-question dedup + MinHash near-dup (Jaccard ≥ 0.95);
       BioASQ > MedQuAD > HCM priority; report counts per source
  G5  Leakage detection vs integrated_conversations_ner_validated.csv;
       add in_conversation_dataset (bool)
  G6  plan_of_action: evidence-grounded only; null for rows with no snippets;
       add has_evidence (bool)
  G7  Stratified train/val/test split (80/10/10) by source_group + answer_type,
       grouped on question_id; all BioASQ golden-test rows forced to test
  G8  Multi-value fields as true list columns; save Parquet + CSV;
       produce retrieval_eval.parquet and qa_bulk.parquet

Outputs
-------
  data/processed/bioasq_pubmed_retrieval_v2.csv
  data/processed/bioasq_pubmed_retrieval_v2.parquet
  data/processed/retrieval_eval.parquet
  data/processed/qa_bulk.parquet
  data/processed/v2_validation_report.txt
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IN_V1          = ROOT / "data" / "processed" / "bioasq_pubmed_retrieval.csv"
CONV_NER       = ROOT / "data" / "processed" / "integrated_conversations_ner_validated.csv"
OUT_DIR        = ROOT / "data" / "processed"
OUT_CSV        = OUT_DIR / "bioasq_pubmed_retrieval_v2.csv"
OUT_PARQUET    = OUT_DIR / "bioasq_pubmed_retrieval_v2.parquet"
OUT_EVAL       = OUT_DIR / "retrieval_eval.parquet"
OUT_BULK       = OUT_DIR / "qa_bulk.parquet"
REPORT_PATH    = OUT_DIR / "v2_validation_report.txt"

# ── constants ─────────────────────────────────────────────────────────────────
SOURCE_PRIORITY = {"BioASQ": 0, "MedQuAD": 1, "HealthCareMagic": 2}

# BioASQ golden source files -> these stay in test split
GOLDEN_SOURCES = {f"{n}B{b}_golden.json" for n in range(4, 8) for b in range(1, 6)}
GOLDEN_SOURCES.add("7B_golden.json")

# MinHash parameters (near-dup detection, G4)
MINHASH_PERMS  = 128
MINHASH_THRESH = 0.95   # Jaccard threshold for near-duplicates
NGRAM_SIZE     = 3      # character n-gram size for shingles

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0  — Load v1 and attach source metadata
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 0 — Load v1 dataset")
log.info("=" * 65)

df = pd.read_csv(IN_V1, low_memory=False, dtype={"question_id": str, "source_file": str})
v1_rows = len(df)
v1_cols = df.columns.tolist()
log.info("v1 loaded: %d rows, %d cols", v1_rows, len(v1_cols))

# ── G1a: source_group ─────────────────────────────────────────────────────────
def assign_source_group(sf: str) -> str:
    if sf == "HealthCareMagic-100k.json":
        return "HealthCareMagic"
    if sf.startswith("medquad"):
        return "MedQuAD"
    return "BioASQ"

df["source_group"] = df["source_file"].apply(assign_source_group)

# ── G1b: medquad_subsource ─────────────────────────────────────────────────
_mq_re = re.compile(r"medquad\.csv\s*\[(.+?)\]")

def extract_mq_subsource(sf: str) -> str | None:
    m = _mq_re.search(sf)
    return m.group(1).strip() if m else None

df["medquad_subsource"] = df["source_file"].apply(extract_mq_subsource)

# ── G1c: fix pubmed_topic ─────────────────────────────────────────────────────
# In v1 the focus_area from medquad.csv was incorrectly stored in pubmed_topic.
# Rule: pubmed_topic must come from an actual PubMed abstract join (has pubmed_abstract).
#       Move mis-stored MedQuAD focus_area to a new column medquad_topic.
df["pubmed_topic"] = df["pubmed_topic"].where(df["source_file"] == "HealthCareMagic-100k.json", other=None)

# Real PubMed join: rows that actually have a matched abstract
has_abstract = df["pubmed_abstract"].notna() & (df["pubmed_abstract"].astype(str).str.strip() != "")
# For BioASQ rows that genuinely matched a PubMed abstract, keep pubmed_topic from v1
real_topic_v1 = pd.read_csv(
    IN_V1, low_memory=False, usecols=["question_id", "pubmed_topic"],
    dtype={"question_id": str}
).set_index("question_id")["pubmed_topic"]

df["pubmed_topic"] = None
df.loc[has_abstract, "pubmed_topic"] = df.loc[has_abstract, "question_id"].map(real_topic_v1)

# medquad_topic: the NIH focus_area for MedQuAD rows (sourced from medquad.csv)
mq_mask = df["source_group"] == "MedQuAD"
df["medquad_topic"] = None
df.loc[mq_mask, "medquad_topic"] = df.loc[mq_mask, "question_id"].map(real_topic_v1)

log.info(
    "G1: source_group counts: %s",
    df["source_group"].value_counts().to_dict()
)
log.info(
    "G1: pubmed_topic non-null (real joins only): %d", df["pubmed_topic"].notna().sum()
)
log.info(
    "G1: medquad_topic non-null: %d", df["medquad_topic"].notna().sum()
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Text cleaning  (G2)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 1 — Text cleaning (G2)")
log.info("=" * 65)


def normalise_unicode(text: str) -> str:
    """NFKC normalisation + collapse non-breaking / zero-width chars."""
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width spaces and similar invisible characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    return text


def normalise_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of spaces/tabs (but preserve single newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # collapse more than two consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# HCM / ChatDoctor greeting / signature patterns
# These are stripped from the START or END of ideal_answer text.
_HCM_GREETING_START = re.compile(
    r"^(?:"
    r"Hi[,\s]+(?:Thank you for (?:posting|consulting|using) (?:your query|(?:in |at )?(?:chat doctor|chatdoctor))[\.,]?|\.+)?"
    r"|Hello[,\s]+(?:and )?(?:I hope I can help you today[\.,]?)?"
    r"|HI[\.,\s]+"
    r"|Thank you for (?:posting your query|using Chat Doctor|consulting (?:in )?Chat Doctor)[\.,]?\s*"
    r")\s*",
    re.IGNORECASE,
)

_HCM_SIGNATURE_END = re.compile(
    r"\s*(?:"
    r"(?:Hope|I hope)(?: I)?(?: have)? answered(?: your)?(?: query)?[\.,!]?\s*"
    r"|Thanks for (?:the )?(?:query|your query)(?:\s+at\s+\S+)?[\.,]?\s*"
    r"|Wish(?:ing)? you(?: a)? (?:speedy )?(?:good )?(?:health|recovery)[\.,!]?\s*"
    r"|(?:Warm regards|Best regards|Regards|Sincerely)[,\.]?\s*(?:\S+\s*){0,4}"
    r"|General [Pp]ediatrics\s*"
    r"|(?:chat doctor|chatdoctor)[\.,]?\s*"
    r")$",
    re.IGNORECASE,
)

# Inline chatdoctori / chat doctor artefacts (mid-text)
_CHATDOCTOR_INLINE = re.compile(
    r"\b(?:chatdoctori|chat\s*doctor(?:\.com)?)\b",
    re.IGNORECASE,
)


def strip_hcm_templates(text: str) -> str:
    """Strip leading greeting and trailing signature from HCM/ChatDoctor answers."""
    if not isinstance(text, str):
        return text
    # iteratively strip leading greeting
    for _ in range(5):
        cleaned = _HCM_GREETING_START.sub("", text)
        if cleaned == text:
            break
        text = cleaned.lstrip()
    # iteratively strip trailing signature
    for _ in range(5):
        cleaned = _HCM_SIGNATURE_END.sub("", text)
        if cleaned == text:
            break
        text = cleaned.rstrip()
    # Remove inline chatdoctor mentions
    text = _CHATDOCTOR_INLINE.sub("[CLINIC]", text)
    return text.strip()


def clean_text_column(series: pd.Series, is_hcm: pd.Series) -> pd.Series:
    """Apply unicode + whitespace normalisation, HCM template stripping."""
    out = series.copy()
    # Unicode + whitespace for all rows
    out = out.apply(lambda x: normalise_whitespace(normalise_unicode(str(x))) if pd.notna(x) else x)
    # HCM template strip for HCM rows only
    out[is_hcm] = out[is_hcm].apply(strip_hcm_templates)
    return out


is_hcm = df["source_group"] == "HealthCareMagic"

log.info("Cleaning question and ideal_answer columns...")
df["question"]     = clean_text_column(df["question"],     is_hcm)
df["ideal_answer"] = clean_text_column(df["ideal_answer"], is_hcm)
df["ground_truth_answer"] = df["ground_truth_answer"].apply(
    lambda x: normalise_whitespace(normalise_unicode(str(x))) if pd.notna(x) else x
)

# ── G2: question_length_flag ──────────────────────────────────────────────────
q_word_count = df["question"].fillna("").apply(lambda x: len(x.split()))
df["question_length_flag"] = "ok"
df.loc[q_word_count < 5,   "question_length_flag"] = "too_short"
df.loc[q_word_count > 512, "question_length_flag"] = "too_long"

flag_counts = df["question_length_flag"].value_counts().to_dict()
log.info("G2: question_length_flag distribution: %s", flag_counts)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PII masking  (G3)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 2 — PII masking (G3, HCM rows only)")
log.info("=" * 65)

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]{1,64}@[a-zA-Z0-9.\-]{1,253}\.[a-zA-Z]{2,6}",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+?1[-.\s]?)?             # optional country code
        (?:\(?\d{3}\)?[-.\s]?)       # area code
        \d{3}[-.\s]?\d{4}            # 7-digit local
    )
    """,
    re.VERBOSE,
)
# Names: "Dr. Firstname Lastname", "My name is Firstname Lastname"
_NAME_RE = re.compile(
    r"\b(?:Dr\.|Mrs?\.|Miss|my name is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}",
    re.IGNORECASE,
)
# City / state / country after "from", "in", "located in", "living in"
_LOCATION_RE = re.compile(
    r"\b(?:from|in|at|located\s+in|living\s+in|based\s+in)\s+[A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){0,2}",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    """Replace emails, phones, names, and location phrases with tokens."""
    if not isinstance(text, str):
        return text
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _NAME_RE.sub(lambda m: re.sub(r"[A-Z][a-z]+", "[NAME]", m.group(), count=2), text)
    text = _LOCATION_RE.sub(
        lambda m: m.group().split()[0] + " [LOCATION]", text
    )
    return text


def apply_pii_masking(df: pd.DataFrame, is_hcm: pd.Series) -> tuple[pd.DataFrame, int]:
    """Mask PII in question + ideal_answer for HCM rows. Return (df, changed_count)."""
    changed = 0
    hcm_idx = df.index[is_hcm]

    for col in ("question", "ideal_answer", "ground_truth_answer"):
        original = df.loc[hcm_idx, col].copy()
        masked   = original.apply(mask_pii)
        changed_mask = masked != original
        df.loc[hcm_idx, col] = masked
        changed += changed_mask.sum()

    # changed = row-level (a row may be counted multiple times if both cols changed)
    # Return unique rows changed instead
    orig_q   = df.loc[hcm_idx, "question"].copy()
    orig_ia  = df.loc[hcm_idx, "ideal_answer"].copy()

    # recompute unique rows changed
    q_changed  = orig_q  != df.loc[hcm_idx, "question"]
    ia_changed = orig_ia != df.loc[hcm_idx, "ideal_answer"]

    # We already applied above; count fresh from scratch
    rows_changed = 0
    for idx in hcm_idx:
        q_orig = df.at[idx, "question"]
        ia_orig = df.at[idx, "ideal_answer"]
        # detect if masking actually changed anything by re-running
        if mask_pii(q_orig) != q_orig or mask_pii(ia_orig) != ia_orig:
            rows_changed += 1
    return df, rows_changed


# We need to do this carefully: apply masking AFTER cleaning but track changes
# Pre-masking copies for change detection
_hcm_q_before  = df.loc[is_hcm, "question"].copy()
_hcm_ia_before = df.loc[is_hcm, "ideal_answer"].copy()
_hcm_gta_before = df.loc[is_hcm, "ground_truth_answer"].copy()

hcm_idx = df.index[is_hcm]
for col in ("question", "ideal_answer", "ground_truth_answer"):
    df.loc[hcm_idx, col] = df.loc[hcm_idx, col].apply(mask_pii)

_pii_q_changed   = (df.loc[is_hcm, "question"]             != _hcm_q_before).sum()
_pii_ia_changed  = (df.loc[is_hcm, "ideal_answer"]         != _hcm_ia_before).sum()
_pii_gta_changed = (df.loc[is_hcm, "ground_truth_answer"]  != _hcm_gta_before).sum()
_pii_rows_changed = (
    (df.loc[is_hcm, "question"]            != _hcm_q_before)  |
    (df.loc[is_hcm, "ideal_answer"]        != _hcm_ia_before) |
    (df.loc[is_hcm, "ground_truth_answer"] != _hcm_gta_before)
).sum()

log.info(
    "G3: PII masking — rows changed: %d  (q: %d, ideal_answer: %d, gta: %d)",
    _pii_rows_changed, _pii_q_changed, _pii_ia_changed, _pii_gta_changed,
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Deduplication  (G4)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 3 — Deduplication (G4)")
log.info("=" * 65)


def normalise_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for dedup key."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Sort by priority: BioASQ=0, MedQuAD=1, HCM=2  ───────────────────────────
df["_priority"] = df["source_group"].map(SOURCE_PRIORITY)
df = df.sort_values("_priority").reset_index(drop=True)

# ── 3a: Exact dedup on normalised question ────────────────────────────────────
pre_exact = len(df)
df["_norm_q"] = df["question"].apply(normalise_for_dedup)

# Keep first (highest priority) occurrence of each normalised question
exact_dup_mask = df.duplicated(subset="_norm_q", keep="first")
exact_removed_by_source = df.loc[exact_dup_mask, "source_group"].value_counts().to_dict()
df = df[~exact_dup_mask].reset_index(drop=True)
post_exact = len(df)
log.info(
    "G4 exact-dedup: %d rows removed (total %d → %d)",
    pre_exact - post_exact, pre_exact, post_exact,
)
log.info("   removed per source: %s", exact_removed_by_source)

# ── 3b: MinHash near-duplicate detection (Jaccard ≥ 0.95)  ───────────────────
log.info("G4: Building MinHash signatures (%d rows, %d permutations)...",
         len(df), MINHASH_PERMS)


def make_shingles(text: str, n: int = NGRAM_SIZE) -> set[str]:
    text = text if isinstance(text, str) else ""
    return {text[i:i+n] for i in range(max(1, len(text) - n + 1))}


def make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_PERMS)
    for shingle in make_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


# Build LSH index — iterate in priority order (already sorted)
lsh = MinHashLSH(threshold=MINHASH_THRESH, num_perm=MINHASH_PERMS)
minhashes: list[MinHash] = []
near_dup_indices: set[int] = set()  # indices to DROP

for i, (idx, row) in enumerate(df.iterrows()):
    mh = make_minhash(row["_norm_q"])
    minhashes.append(mh)
    key = str(idx)
    try:
        candidates = lsh.query(mh)
        if candidates:
            # This row is a near-dup of a higher-priority row already in the index
            near_dup_indices.add(idx)
        else:
            lsh.insert(key, mh)
    except Exception:
        lsh.insert(key, mh)

    if (i + 1) % 20000 == 0:
        log.info("  MinHash progress: %d / %d", i + 1, len(df))

pre_near = len(df)
near_dup_src = df.loc[list(near_dup_indices), "source_group"].value_counts().to_dict()
df = df.drop(index=list(near_dup_indices)).reset_index(drop=True)
post_near = len(df)
log.info(
    "G4 near-dedup: %d rows removed (total %d → %d)",
    pre_near - post_near, pre_near, post_near,
)
log.info("   removed per source: %s", near_dup_src)

# Clean up helper columns
df = df.drop(columns=["_priority", "_norm_q"], errors="ignore")

total_removed_dedup = v1_rows - len(df)
pct_removed = total_removed_dedup / v1_rows * 100
log.info(
    "G4 total dedup removed: %d rows (%.2f%% of v1)",
    total_removed_dedup, pct_removed,
)

# ── Safety check: alert if > 2% of rows would be deleted ─────────────────────
if pct_removed > 2.0:
    log.warning(
        "⚠  Deduplication removed %.2f%% of rows (> 2%% threshold). "
        "Review near_dup_src=%s before proceeding.",
        pct_removed, near_dup_src,
    )

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Leakage detection  (G5)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 4 — Leakage detection (G5)")
log.info("=" * 65)

df_conv = pd.read_csv(
    CONV_NER,
    low_memory=False,
    dtype={"original_id": str, "dialogue_id": str},
    usecols=["utterance", "source_dataset", "speaker"],
)

# Collect normalised MedDialog user utterances
meddialog_user = df_conv[
    (df_conv["source_dataset"] == "MedDialog") &
    (df_conv["speaker"] == "user") &
    df_conv["utterance"].notna()
]["utterance"]

conv_norm_set: set[str] = set(meddialog_user.apply(normalise_for_dedup))
log.info(
    "G5: MedDialog user utterance set size (unique normalised): %d",
    len(conv_norm_set),
)

# Match HCM questions against the MedDialog set
hcm_mask = df["source_group"] == "HealthCareMagic"
hcm_norms = df.loc[hcm_mask, "question"].apply(normalise_for_dedup)
df["in_conversation_dataset"] = False
df.loc[hcm_mask, "in_conversation_dataset"] = hcm_norms.isin(conv_norm_set)

leakage_count = df["in_conversation_dataset"].sum()
log.info(
    "G5: Leakage overlap — %d HCM rows match a MedDialog user turn",
    leakage_count,
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Plan of Action + has_evidence  (G6)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 5 — Plan of Action & has_evidence (G6)")
log.info("=" * 65)


def snippet_count(snippets_text: Any) -> int:
    """Count pipe-delimited snippets. Returns 0 for null/empty."""
    if not isinstance(snippets_text, str) or not snippets_text.strip():
        return 0
    return len([s for s in snippets_text.split("|||") if s.strip()])


def safe_truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate to max_chars without splitting a word; append '...' if truncated."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip() + "..."


def derive_plan_of_action_v2(
    ideal_answer: str,
    answer_type: str,
    exact_answer: str,
    snippets_text: str,
    snippet_pmids: str,
) -> str | None:
    """
    Build a 5-step POA only for rows with real evidence (snippets_text non-empty).
    Returns None for rows without evidence.
    Step 2 reflects the actual snippet count and PMID list.
    """
    n_snips = snippet_count(snippets_text)
    if n_snips == 0:
        return None  # G6: null when no evidence

    # Actual PMIDs for Step 2
    snippet_pmids_str = str(snippet_pmids) if isinstance(snippet_pmids, str) else ""
    pmids_list = [p.strip() for p in snippet_pmids_str.split("|") if p.strip()]
    pmids_str  = ", ".join(pmids_list[:10])
    if len(pmids_list) > 10:
        pmids_str += f" (+{len(pmids_list)-10} more)"

    # Step 4 synthesis: max 250 chars, no mid-word truncation
    ia = str(ideal_answer) if isinstance(ideal_answer, str) else ""
    summary = safe_truncate_at_word(ia, 250)

    exact = str(exact_answer) if isinstance(exact_answer, str) else ""

    if answer_type == "yesno":
        recommendation = (
            f"Clinical recommendation: The answer is '{exact.upper()}'. "
            "Proceed or withhold the intervention accordingly based on clinical context."
        )
    elif answer_type == "factoid":
        recommendation = (
            f"Clinical recommendation: The specific entity/value is '{exact}'. "
            "Use this as the target in downstream queries or treatment decisions."
        )
    elif answer_type == "list":
        items = [i.strip() for i in exact.split(" | ") if i.strip()] if exact else []
        shown = "; ".join(items[:5])
        tail  = f" (and {len(items)-5} more)" if len(items) > 5 else ""
        recommendation = (
            f"Clinical recommendation: Consider each of the {len(items)} identified "
            f"items: {shown}{tail}."
        )
    else:  # summary / open_qa / unknown
        recommendation = (
            "Clinical recommendation: Review the synthesised evidence above "
            "and formulate a management plan with the treating clinician."
        )

    return (
        "[Step 1] Identify the clinical/research question.\n"
        f"[Step 2] Retrieved {n_snips} supporting snippet(s) from PubMed "
        f"(PMID: {pmids_str}).\n"
        f"[Step 3] Ground-truth answer: {exact}\n"
        f"[Step 4] Synthesise: {summary}\n"
        f"[Step 5] {recommendation}"
    )


# has_evidence flag
df["has_evidence"] = df["snippets_text"].apply(
    lambda x: snippet_count(x) > 0
)
log.info("G6: has_evidence=True rows: %d", df["has_evidence"].sum())

# Rebuild plan_of_action
df["plan_of_action"] = df.apply(
    lambda r: derive_plan_of_action_v2(
        ideal_answer  = r["ideal_answer"],
        answer_type   = r["answer_type"],
        exact_answer  = r["ground_truth_answer"],
        snippets_text = r["snippets_text"],
        snippet_pmids = r["snippet_pmids"],
    ),
    axis=1,
)

poa_nonnull = df["plan_of_action"].notna().sum()
log.info(
    "G6: plan_of_action non-null: %d  (null for %d rows without evidence)",
    poa_nonnull, len(df) - poa_nonnull,
)

# Consistency check: has_evidence must align with plan_of_action non-null
misalign = (df["has_evidence"] & df["plan_of_action"].isna()).sum()
if misalign:
    log.error("G6 consistency error: %d rows have has_evidence=True but null POA", misalign)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Train/val/test split  (G7)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 6 — Train/val/test split (G7)")
log.info("=" * 65)

# Force BioASQ golden-test rows to test split
is_golden = df["source_file"].isin(GOLDEN_SOURCES)
df["split"] = None
df.loc[is_golden, "split"] = "test"

# Remaining rows: stratified split by source_group + answer_type
remaining = df[~is_golden].copy()
remaining["_strat_key"] = remaining["source_group"] + "|" + remaining["answer_type"]

# StratifiedGroupKFold with 10 splits → 10% val, 10% test from the 20% held-out
# We use a 10-fold splitter: fold 0 = val, fold 1 = test
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

X      = np.zeros(len(remaining))
y      = remaining["_strat_key"].values
groups = remaining["question_id"].values

# fold 0 → val candidates (10%)
fold0_train_idx, fold0_val_idx = next(iter(sgkf.split(X, y, groups)))
val_qids = set(remaining.iloc[fold0_val_idx]["question_id"])

# fold 1 from remaining non-val → test candidates (10% of total)
remaining_after_val = remaining.iloc[fold0_train_idx]
X2      = np.zeros(len(remaining_after_val))
y2      = remaining_after_val["_strat_key"].values
groups2 = remaining_after_val["question_id"].values

sgkf2 = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
fold1_train_idx, fold1_test_idx = next(iter(sgkf2.split(X2, y2, groups2)))
test_qids = set(remaining_after_val.iloc[fold1_test_idx]["question_id"])

# Assign splits
remaining_idx = remaining.index
df.loc[remaining_idx[remaining["question_id"].isin(val_qids)],  "split"] = "val"
df.loc[remaining_idx[remaining["question_id"].isin(test_qids)], "split"] = "test"
df.loc[df["split"].isna(), "split"] = "train"

split_counts = df["split"].value_counts().to_dict()
log.info("G7: split distribution: %s", split_counts)

# Verify: no question_id overlap across splits
_train_ids = set(df.loc[df["split"] == "train", "question_id"])
_val_ids   = set(df.loc[df["split"] == "val",   "question_id"])
_test_ids  = set(df.loc[df["split"] == "test",  "question_id"])
assert not (_train_ids & _val_ids),  "FAIL: train/val question_id overlap"
assert not (_train_ids & _test_ids), "FAIL: train/test question_id overlap"
assert not (_val_ids   & _test_ids), "FAIL: val/test question_id overlap"
log.info("G7: ✓ No question_id overlap across splits")

# Verify: all BioASQ golden rows are in test
golden_in_test = df.loc[is_golden, "split"].eq("test").all()
assert golden_in_test, "FAIL: some BioASQ golden rows not in test split"
log.info("G7: ✓ All %d BioASQ golden rows are in test split", is_golden.sum())

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — List columns + final schema  (G8)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 7 — List columns and finalise schema (G8)")
log.info("=" * 65)


def split_pipe_field(value: Any, separator: str = "|||") -> list[str]:
    """Convert '|||'-separated or '|'-separated string to list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [s.strip() for s in value.split(separator) if s.strip()]


def split_pmid_field(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return split_pipe_field(value, separator="|")


df["snippets_list"]     = df["snippets_text"].apply(split_pipe_field)
df["snippet_pmids_list"] = df["snippet_pmids"].apply(split_pmid_field)
df["all_doc_pmids_list"] = df["all_doc_pmids"].apply(split_pmid_field)

# ── Final column order ─────────────────────────────────────────────────────────
FINAL_COLS_CSV = [
    "question_id",
    "question",
    "answer_type",
    "source_group",
    "medquad_subsource",
    "source_file",
    "ground_truth_answer",
    "ideal_answer",
    "snippets_text",
    "snippet_pmids",
    "all_doc_pmids",
    "pubmed_title",
    "pubmed_abstract",
    "pubmed_topic",        # real PubMed joins only
    "medquad_topic",       # NIH focus_area for MedQuAD rows
    "plan_of_action",
    "has_evidence",
    "question_length_flag",
    "in_conversation_dataset",
    "split",
]

FINAL_COLS_PARQUET = FINAL_COLS_CSV + [
    "snippets_list",
    "snippet_pmids_list",
    "all_doc_pmids_list",
]

df_csv     = df[FINAL_COLS_CSV].copy()
df_parquet = df[FINAL_COLS_PARQUET].copy()

# Ensure string columns have clean NaN → None (for Parquet null handling)
for col in FINAL_COLS_CSV:
    if df_csv[col].dtype == object:
        df_csv[col]     = df_csv[col].where(df_csv[col].notna(), other=None)
        df_parquet[col] = df_parquet[col].where(df_parquet[col].notna(), other=None)

log.info("Final v2 schema: %d rows, %d cols (CSV) / %d cols (Parquet)",
         len(df_csv), len(FINAL_COLS_CSV), len(FINAL_COLS_PARQUET))

# ── Assertions  ───────────────────────────────────────────────────────────────
log.info("Running assertions...")

assert df_csv["question_id"].nunique() == len(df_csv), \
    f"FAIL: non-unique question_id ({df_csv['question_id'].nunique()} unique, {len(df_csv)} rows)"

assert (df_csv["question"].fillna("").str.strip() != "").all(), \
    "FAIL: empty questions in final dataset"

no_overlap = (
    not (_train_ids & _val_ids) and
    not (_train_ids & _test_ids) and
    not (_val_ids   & _test_ids)
)
assert no_overlap, "FAIL: split leakage detected"

# has_evidence consistent with snippets_text
he_true  = df_csv["has_evidence"] == True
poa_null = df_csv["plan_of_action"].isna()
assert not (he_true & poa_null).any(), \
    "FAIL: has_evidence=True but plan_of_action is null"

he_false    = df_csv["has_evidence"] == False
poa_notnull = df_csv["plan_of_action"].notna()
assert not (he_false & poa_notnull).any(), \
    "FAIL: has_evidence=False but plan_of_action is non-null"

log.info("✓ All assertions passed")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Save outputs  (G8)
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 8 — Saving outputs")
log.info("=" * 65)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main CSV
df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8")
log.info("Saved CSV → %s  (%.1f MB)", OUT_CSV, OUT_CSV.stat().st_size / 1e6)

# Main Parquet
df_parquet.to_parquet(OUT_PARQUET, index=False, engine="pyarrow")
log.info("Saved Parquet → %s  (%.1f MB)", OUT_PARQUET, OUT_PARQUET.stat().st_size / 1e6)

# Derived: retrieval_eval (rows with evidence/snippets)
df_eval = df_parquet[df_parquet["has_evidence"] == True].copy()
df_eval.to_parquet(OUT_EVAL, index=False, engine="pyarrow")
log.info(
    "Saved retrieval_eval.parquet → %s  (%d rows, %.1f MB)",
    OUT_EVAL, len(df_eval), OUT_EVAL.stat().st_size / 1e6,
)

# Derived: qa_bulk (rows without evidence)
df_bulk = df_parquet[df_parquet["has_evidence"] == False].copy()
df_bulk.to_parquet(OUT_BULK, index=False, engine="pyarrow")
log.info(
    "Saved qa_bulk.parquet → %s  (%d rows, %.1f MB)",
    OUT_BULK, len(df_bulk), OUT_BULK.stat().st_size / 1e6,
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Validation report
# ═════════════════════════════════════════════════════════════════════════════

log.info("=" * 65)
log.info("SECTION 9 — Validation report")
log.info("=" * 65)

null_per_col = {col: int(df_csv[col].isna().sum()) for col in FINAL_COLS_CSV}

report_lines = [
    "=" * 70,
    "  V2 DATASET VALIDATION REPORT",
    "=" * 70,
    "",
    "── ROW COUNTS ──────────────────────────────────────────────────────────",
    f"  v1 input rows          : {v1_rows:>10,}",
    f"  v2 output rows         : {len(df_csv):>10,}",
    f"  Δ (removed)            : {v1_rows - len(df_csv):>10,}",
    "",
    "── DEDUPLICATION (G4) ──────────────────────────────────────────────────",
    f"  Exact-dup removed      : {pre_exact - post_exact:>10,}",
    f"    by source_group      : {exact_removed_by_source}",
    f"  Near-dup removed       : {pre_near - post_near:>10,}",
    f"    by source_group      : {near_dup_src}",
    f"  Total removed          : {total_removed_dedup:>10,}  ({pct_removed:.2f}% of v1)",
    "",
    "── PII MASKING (G3) ────────────────────────────────────────────────────",
    f"  HCM rows with PII changed : {_pii_rows_changed:>7,}",
    f"    question col changed    : {_pii_q_changed:>7,}",
    f"    ideal_answer changed    : {_pii_ia_changed:>7,}",
    f"    ground_truth_answer     : {_pii_gta_changed:>7,}",
    "",
    "── QUESTION LENGTH FLAGS (G2) ──────────────────────────────────────────",
]
for flag, cnt in df_csv["question_length_flag"].value_counts().items():
    report_lines.append(f"  {flag:<20s}: {cnt:>10,}")
report_lines += [
    "",
    "── SOURCE GROUP COUNTS (G1) ────────────────────────────────────────────",
]
for sg, cnt in df_csv["source_group"].value_counts().items():
    pct = cnt / len(df_csv) * 100
    report_lines.append(f"  {sg:<20s}: {cnt:>10,}  ({pct:.1f}%)")
report_lines += [
    "",
    "── ANSWER TYPE DISTRIBUTION ────────────────────────────────────────────",
]
for at, cnt in df_csv["answer_type"].value_counts().items():
    report_lines.append(f"  {at:<20s}: {cnt:>10,}")
report_lines += [
    "",
    "── EVIDENCE & POA (G6) ─────────────────────────────────────────────────",
    f"  has_evidence=True      : {int(df_csv['has_evidence'].sum()):>10,}",
    f"  plan_of_action non-null: {int(df_csv['plan_of_action'].notna().sum()):>10,}",
    f"  POA/evidence mismatch  : {int((df_csv['has_evidence'] & df_csv['plan_of_action'].isna()).sum()):>10,}",
    "",
    "── LEAKAGE DETECTION (G5) ──────────────────────────────────────────────",
    f"  in_conversation_dataset=True  : {int(df_csv['in_conversation_dataset'].sum()):>7,}",
    "",
    "── SPLIT DISTRIBUTION (G7) ─────────────────────────────────────────────",
]
for sp, cnt in df_csv["split"].value_counts().items():
    pct = cnt / len(df_csv) * 100
    report_lines.append(f"  {sp:<10s}: {cnt:>10,}  ({pct:.1f}%)")
report_lines += [
    "",
    "── SPLIT × SOURCE_GROUP ────────────────────────────────────────────────",
]
cross = pd.crosstab(df_csv["split"], df_csv["source_group"])
report_lines.append(cross.to_string())
report_lines += [
    "",
    "── NULL COUNTS PER COLUMN ──────────────────────────────────────────────",
]
for col, nulls in null_per_col.items():
    pct = nulls / len(df_csv) * 100
    report_lines.append(f"  {col:<35s}: {nulls:>10,}  ({pct:.1f}% null)")
report_lines += [
    "",
    "── DERIVED FILES ───────────────────────────────────────────────────────",
    f"  retrieval_eval.parquet : {len(df_eval):>10,} rows (has_evidence=True)",
    f"  qa_bulk.parquet        : {len(df_bulk):>10,} rows (has_evidence=False)",
    "",
    "── ASSERTIONS ──────────────────────────────────────────────────────────",
    "  ✓ unique question_id",
    "  ✓ no empty questions",
    "  ✓ split has no ID overlap",
    "  ✓ has_evidence consistent with snippets_text and plan_of_action",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
print(report_text)

REPORT_PATH.write_text(report_text, encoding="utf-8")
log.info("Validation report saved → %s", REPORT_PATH)

log.info("=" * 65)
log.info("BUILD COMPLETE  — v2 rows: %d  columns: %d", len(df_csv), len(FINAL_COLS_CSV))
log.info("=" * 65)
