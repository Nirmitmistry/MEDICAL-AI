"""
================================================================================
  INTEGRATED CONVERSATIONAL MEDICAL DATASET
  End-to-End Audit · Sanitization · Structural Harmonization · Validation
================================================================================
Script   : 03_audit_sanitize_validate.py
Author   : Data Engineering Pipeline
Purpose  : Full A-to-Z audit, cleaning, re-indexing, and automated test suite
           for integrated_conversational_backbone.csv
           Produces : data/processed/medical_conversations_clean.csv

Kaggle paths  : /kaggle/input/<dataset>/integrated_conversational_backbone.csv
                /kaggle/working/medical_conversations_clean.csv
Local paths   : data/raw/  →  data/processed/
================================================================================
"""

import os
import re
import sys
import unicodedata
import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATH RESOLVER  (Kaggle-first, falls back to local)
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_INPUT  = "/kaggle/input"
KAGGLE_OUTPUT = "/kaggle/working"
LOCAL_INPUT   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOCAL_OUTPUT  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def _resolve_paths() -> Tuple[str, str]:
    """Return (input_dir, output_dir) for the current execution environment."""
    if os.path.isdir(KAGGLE_INPUT):
        # Kaggle: find first sub-directory that contains the target file
        for sub in os.listdir(KAGGLE_INPUT):
            candidate = os.path.join(KAGGLE_INPUT, sub,
                                     "integrated_conversational_backbone.csv")
            if os.path.isfile(candidate):
                return os.path.dirname(candidate), KAGGLE_OUTPUT
        # fallback: root of kaggle input
        return KAGGLE_INPUT, KAGGLE_OUTPUT
    else:
        os.makedirs(LOCAL_OUTPUT, exist_ok=True)
        return LOCAL_INPUT, LOCAL_OUTPUT

INPUT_DIR, OUTPUT_DIR = _resolve_paths()
INPUT_FILE  = os.path.join(INPUT_DIR,  "integrated_conversational_backbone.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "medical_conversations_clean.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit")

DIVIDER = "-" * 72


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONSTANTS  – canonical schema & source mappings
# ─────────────────────────────────────────────────────────────────────────────

# Columns present in the raw CSV that we want to keep (before rename)
RAW_KEEP_COLS = ["dialogue_id", "turn_id", "speaker", "utterance", "source_dataset"]

# Canonical output schema
CANONICAL_COLS = ["conversation_id", "turn_id", "role", "utterance", "source_dataset"]

# Raw source_dataset values  →  canonical source label
SOURCE_LABEL_MAP = {
    "MedDialog":          "MedDialog-EN",
    "HealthChat-LMSYS":   "HealthChat-11k",
    "HealthChat-WildChat": "HealthChat-11k",
}

# Raw speaker/role values  →  canonical role
ROLE_NORMALISE_MAP = {
    "user":      "user",
    "patient":   "user",
    "human":     "user",
    "customer":  "user",
    "assistant": "assistant",
    "doctor":    "assistant",
    "physician": "assistant",
    "bot":       "assistant",
    "agent":     "assistant",
}

# conversation_id prefix per original source_dataset value
ID_PREFIX_MAP = {
    "MedDialog":           "meddialog",
    "HealthChat-LMSYS":    "healthchat",
    "HealthChat-WildChat":  "healthchat",
}

# HTML tag pattern
_HTML_RE   = re.compile(r"<[^>]+>")
# Control characters (keep \t \n \r  =  \x09 \x0A \x0D)
_CTRL_RE   = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
# Collapse runs of whitespace (space, tab, newline) to single space
_SPACE_RE  = re.compile(r"[ \t\r\n]+")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUDIT REPORT  – accumulates stats for the final log
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AuditReport:
    raw_rows:                  int = 0
    raw_conversations:         int = 0
    removed_null_terminal:     int = 0   # single assistant turns trimmed
    removed_null_mid_conv:     int = 0   # full conversations dropped
    removed_null_user_turns:   int = 0   # full conversations dropped (null user)
    removed_assistant_first:   int = 0   # convos starting with assistant
    removed_hanging_user:      int = 0   # convos ending with user after trim
    clean_conversations:       int = 0
    clean_rows:                int = 0
    source_breakdown:          dict = field(default_factory=dict)

    def banner(self) -> str:
        lines = [
            DIVIDER,
            "  AUDIT SUMMARY",
            DIVIDER,
            f"  Raw rows loaded          : {self.raw_rows:>10,}",
            f"  Raw conversations        : {self.raw_conversations:>10,}",
            "",
            "  REMOVALS",
            f"  – Null terminal asst turn (rows trimmed)  : {self.removed_null_terminal:>7,}",
            f"  – Null mid-turn (convos dropped)          : {self.removed_null_mid_conv:>7,}",
            f"  – Null user turn (convos dropped)         : {self.removed_null_user_turns:>7,}",
            f"  – Asst-first convo (convos dropped)       : {self.removed_assistant_first:>7,}",
            f"  – Hanging user final turn (convos dropped): {self.removed_hanging_user:>7,}",
            "",
            "  OUTPUT",
            f"  Clean conversations      : {self.clean_conversations:>10,}",
            f"  Clean rows (turns)       : {self.clean_rows:>10,}",
            "",
            "  SOURCE BREAKDOWN",
        ]
        for src, cnt in self.source_breakdown.items():
            lines.append(f"    {src:<20} : {cnt:>8,} conversations")
        lines.append(DIVIDER)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODULE 1 – LOAD & VERIFY
# ─────────────────────────────────────────────────────────────────────────────
def load_and_verify(path: str, report: AuditReport) -> pd.DataFrame:
    """Load the raw CSV and verify the expected columns exist."""
    log.info("Loading raw dataset from: %s", path)
    df = pd.read_csv(path, low_memory=False, dtype={"original_id": str})

    log.info("Raw shape : %s rows × %s columns", *df.shape)
    log.info("Columns   : %s", df.columns.tolist())

    # ── Verify all expected raw columns are present ──────────────────────────
    missing = [c for c in RAW_KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[SCHEMA ERROR] Missing expected columns: {missing}")
    log.info("[PASS] All required source columns present: %s", RAW_KEEP_COLS)

    # ── Verify known source_dataset values ───────────────────────────────────
    found_sources = set(df["source_dataset"].dropna().unique())
    unknown_sources = found_sources - set(SOURCE_LABEL_MAP.keys())
    if unknown_sources:
        log.warning("[WARN] Unrecognised source_dataset values: %s", unknown_sources)
    else:
        log.info("[PASS] All source_dataset values recognised: %s", sorted(found_sources))

    report.raw_rows          = len(df)
    report.raw_conversations = df["dialogue_id"].nunique()
    log.info("Raw conversations : %d", report.raw_conversations)

    # ── Keep only the columns we need; drop everything else ──────────────────
    df = df[RAW_KEEP_COLS].copy()
    log.info("[PASS] Stripped to minimal schema: %s", RAW_KEEP_COLS)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODULE 2 – COLUMN NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Rename legacy column names to canonical names.
    • Build prefixed conversation_id to prevent cross-source ID collision.
    • Normalise role values to user/assistant.
    • Map source_dataset to canonical label.
    """
    log.info("Normalising columns …")

    # ── Build prefixed conversation_id ────────────────────────────────────────
    prefix_series = df["source_dataset"].map(ID_PREFIX_MAP).fillna("unknown")
    df["conversation_id"] = prefix_series + "_" + df["dialogue_id"].astype(str)

    # ── Normalise role ────────────────────────────────────────────────────────
    df["role"] = (
        df["speaker"]
        .str.strip()
        .str.lower()
        .map(ROLE_NORMALISE_MAP)
    )
    unmapped_roles = df["role"].isnull().sum()
    if unmapped_roles > 0:
        log.warning("[WARN] %d speaker values could not be mapped to user/assistant — "
                    "those rows will carry NaN role and be removed later.", unmapped_roles)

    # ── Canonicalise source_dataset ───────────────────────────────────────────
    df["source_dataset"] = df["source_dataset"].map(SOURCE_LABEL_MAP)

    # ── Drop legacy columns, keep only canonical set ─────────────────────────
    df = df[CANONICAL_COLS].copy()

    log.info("[PASS] Column normalisation complete. Schema: %s", CANONICAL_COLS)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODULE 3 – TEXT SANITISATION
# ─────────────────────────────────────────────────────────────────────────────
def sanitise_text(text) -> str:
    """
    Apply NFKC normalisation, strip HTML tags, remove control characters,
    and collapse internal whitespace.  Returns cleaned string or empty string.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)
    text = _CTRL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def apply_text_sanitisation(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised application of sanitise_text to the utterance column."""
    log.info("Sanitising utterance text …")
    df["utterance"] = df["utterance"].apply(sanitise_text)
    # Convert empty strings back to NaN so null-handling logic works cleanly
    df["utterance"] = df["utterance"].replace("", pd.NA)
    log.info("[PASS] Text sanitisation applied.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MODULE 4 – NULL & DIALOGUE INTEGRITY CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_nulls_and_integrity(df: pd.DataFrame, report: AuditReport) -> pd.DataFrame:
    """
    Null resolution rules (applied in order):
      A. Terminal assistant null  → trim that single turn.
      B. Null on user turn        → drop entire conversation.
      C. Null on mid-dialogue assistant turn → drop entire conversation.

    Structural rules:
      D. Conversation starts with assistant → drop.
      E. After all trimming, conversation ends with user → drop.
    """
    log.info("Resolving nulls and dialogue integrity …")

    # ── Sort so turn order is deterministic ───────────────────────────────────
    df = df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True)

    # ─── A.  Terminal assistant null turns (safe to trim) ─────────────────────
    # Identify conversations whose LAST turn has a null utterance AND role == assistant
    last_idx = df.groupby("conversation_id")["turn_id"].transform("max")
    is_last  = df["turn_id"] == last_idx
    terminal_null_asst = is_last & df["utterance"].isna() & (df["role"] == "assistant")

    report.removed_null_terminal = terminal_null_asst.sum()
    log.info("  Trimming %d terminal null assistant turns …",
             report.removed_null_terminal)
    df = df[~terminal_null_asst].copy()

    # ─── B.  Null user turns → drop whole conversation ────────────────────────
    null_user_convos = df.loc[
        df["utterance"].isna() & (df["role"] == "user"), "conversation_id"
    ].unique()
    report.removed_null_user_turns = len(null_user_convos)
    log.info("  Dropping %d conversations with null user turns …",
             report.removed_null_user_turns)
    df = df[~df["conversation_id"].isin(null_user_convos)].copy()

    # ─── C.  Remaining nulls are mid-dialogue assistant turns → drop convo ────
    remaining_null_convos = df.loc[
        df["utterance"].isna(), "conversation_id"
    ].unique()
    report.removed_null_mid_conv = len(remaining_null_convos)
    log.info("  Dropping %d conversations with mid-dialogue null assistant turns …",
             report.removed_null_mid_conv)
    df = df[~df["conversation_id"].isin(remaining_null_convos)].copy()

    # ─── D.  Conversations that start with assistant → drop ───────────────────
    first_roles = (
        df.sort_values(["conversation_id", "turn_id"])
          .groupby("conversation_id")["role"]
          .first()
    )
    asst_first_convos = first_roles[first_roles != "user"].index
    report.removed_assistant_first = len(asst_first_convos)
    log.info("  Dropping %d conversations starting with assistant …",
             report.removed_assistant_first)
    df = df[~df["conversation_id"].isin(asst_first_convos)].copy()

    # ─── E.  Hanging unanswered user turns at end → drop whole conversation ───
    # (After trimming above, recompute last turn for safety)
    last_roles = (
        df.sort_values(["conversation_id", "turn_id"])
          .groupby("conversation_id")["role"]
          .last()
    )
    hanging_user_convos = last_roles[last_roles != "assistant"].index
    report.removed_hanging_user = len(hanging_user_convos)
    log.info("  Dropping %d conversations with hanging final user turn …",
             report.removed_hanging_user)
    df = df[~df["conversation_id"].isin(hanging_user_convos)].copy()

    log.info("[PASS] Null & integrity cleaning complete. Rows remaining: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MODULE 5 – STRUCTURAL RE-INDEXING
# ─────────────────────────────────────────────────────────────────────────────
def reindex_turn_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every conversation, sort by existing turn_id to preserve original
    chronological order, then reassign turn_id as a strictly 0-indexed
    monotonic integer sequence (0, 1, 2, …).
    """
    log.info("Re-indexing turn_ids …")

    df = df.sort_values(["conversation_id", "turn_id"]).copy()

    # cumcount within each group gives 0, 1, 2, … automatically
    df["turn_id"] = df.groupby("conversation_id").cumcount()

    # Enforce integer dtype
    df["turn_id"] = df["turn_id"].astype(int)

    # Restore canonical column order and reset the DataFrame index
    df = df[CANONICAL_COLS].reset_index(drop=True)

    log.info("[PASS] turn_id re-indexed. Sample:\n%s",
             df[df["conversation_id"] == df["conversation_id"].iloc[0]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MODULE 6 – AUTOMATED VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name:    str
    passed:  bool
    detail:  str = ""

    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        base   = f"  {status}  {self.name}"
        return base if not self.detail else f"{base}\n         ↳ {self.detail}"


def run_validation_suite(df: pd.DataFrame, report: AuditReport) -> List[CheckResult]:
    """
    Automated A-to-Z validation suite.

    Checks:
      1. Zero-Null Assertion
      2. Turn Monotonicity Assertion
      3. Role Alternation Check
      4. Conversation Boundary Integrity
      5. Collision & Distribution Audit
    """
    results: List[CheckResult] = []

    # ── 1. Zero-Null Assertion ────────────────────────────────────────────────
    for col in CANONICAL_COLS:
        null_count = df[col].isna().sum()
        # For utterance also check for blank strings
        if col == "utterance":
            blank_count = (df[col].str.strip() == "").sum()
            total = null_count + blank_count
            results.append(CheckResult(
                name   = f"Zero-Null: '{col}' (null={null_count}, blank={blank_count})",
                passed = (total == 0),
                detail = f"{total} problematic values" if total else "0 null/blank values",
            ))
        else:
            results.append(CheckResult(
                name   = f"Zero-Null: '{col}'",
                passed = (null_count == 0),
                detail = f"{null_count} null values" if null_count else "0 null values",
            ))

    # ── 2. Turn Monotonicity Assertion ────────────────────────────────────────
    def _check_monotonicity(grp):
        turns = grp["turn_id"].tolist()
        expected = list(range(len(turns)))
        return turns != expected

    bad_mono = (
        df.groupby("conversation_id", group_keys=False)
          .apply(_check_monotonicity, include_groups=False)
    )
    bad_count = bad_mono.sum()
    results.append(CheckResult(
        name   = "Turn Monotonicity (starts at 0, increments by +1, no gaps/dupes)",
        passed = (bad_count == 0),
        detail = (f"{bad_count} conversations with non-monotonic turn_ids"
                  if bad_count else "All conversations pass"),
    ))

    # ── 3. Role Alternation Check ─────────────────────────────────────────────
    df_sorted = df.sort_values(["conversation_id", "turn_id"])
    prev_role  = df_sorted.groupby("conversation_id")["role"].shift(1)
    consec_mask = (df_sorted["role"] == prev_role) & prev_role.notna()
    consec_count = consec_mask.sum()

    if consec_count > 0:
        sample_convos = df_sorted.loc[consec_mask, "conversation_id"].unique()[:5].tolist()
        detail = (f"{consec_count} consecutive same-role turns found. "
                  f"Sample conversations: {sample_convos}")
    else:
        detail = "No consecutive same-role turns detected"

    results.append(CheckResult(
        name   = "Role Alternation (no consecutive user→user or asst→asst)",
        passed = (consec_count == 0),
        detail = detail,
    ))

    # ── 4. Conversation Boundary Integrity ────────────────────────────────────
    grouped = df_sorted.groupby("conversation_id")["role"]
    first_roles = grouped.first()
    last_roles  = grouped.last()

    bad_start = (first_roles != "user").sum()
    bad_end   = (last_roles  != "assistant").sum()

    results.append(CheckResult(
        name   = "Boundary: 100% conversations start with 'user'",
        passed = (bad_start == 0),
        detail = (f"{bad_start} conversations start with non-user role"
                  if bad_start else "All conversations start with user"),
    ))
    results.append(CheckResult(
        name   = "Boundary: 100% conversations end with 'assistant'",
        passed = (bad_end == 0),
        detail = (f"{bad_end} conversations end with non-assistant role"
                  if bad_end else "All conversations end with assistant"),
    ))

    # ── 5. Collision & Distribution Audit ────────────────────────────────────
    total_convos = df["conversation_id"].nunique()
    total_turns  = len(df)
    by_source    = (
        df.groupby("source_dataset")["conversation_id"]
          .nunique()
          .to_dict()
    )
    report.clean_conversations = total_convos
    report.clean_rows          = total_turns
    report.source_breakdown    = by_source

    # Check uniqueness of conversation_id across the full dataset
    id_collisions = df.groupby("conversation_id")["source_dataset"].nunique()
    cross_source_collisions = (id_collisions > 1).sum()

    dist_detail = (
        f"Total conversations={total_convos:,} | Total turns={total_turns:,} | "
        + " | ".join(f"{s}={n:,}" for s, n in by_source.items())
    )
    results.append(CheckResult(
        name   = "Collision Audit: conversation_id globally unique per source",
        passed = (cross_source_collisions == 0),
        detail = (f"{cross_source_collisions} cross-source ID collisions detected"
                  if cross_source_collisions else "No cross-source collisions"),
    ))
    results.append(CheckResult(
        name   = "Distribution Audit",
        passed = True,
        detail = dist_detail,
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MODULE 7 – PERSIST OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
def save_output(df: pd.DataFrame, path: str) -> None:
    """Write the clean dataset to CSV, ensuring output directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    size_mb = os.path.getsize(path) / (1024 ** 2)
    log.info("[PASS] Output saved → %s  (%.2f MB, %d rows)", path, size_mb, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# 11.  PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline() -> None:
    report = AuditReport()

    log.info(DIVIDER)
    log.info("  STEP 1/7 — Load & Verify")
    log.info(DIVIDER)
    df = load_and_verify(INPUT_FILE, report)

    log.info(DIVIDER)
    log.info("  STEP 2/7 — Column Normalisation")
    log.info(DIVIDER)
    df = normalise_columns(df)

    log.info(DIVIDER)
    log.info("  STEP 3/7 — Text Sanitisation")
    log.info(DIVIDER)
    df = apply_text_sanitisation(df)

    log.info(DIVIDER)
    log.info("  STEP 4/7 — Null & Dialogue Integrity Cleaning")
    log.info(DIVIDER)
    df = clean_nulls_and_integrity(df, report)

    log.info(DIVIDER)
    log.info("  STEP 5/7 — Structural Re-indexing")
    log.info(DIVIDER)
    df = reindex_turn_ids(df)

    log.info(DIVIDER)
    log.info("  STEP 6/7 — Save Output")
    log.info(DIVIDER)
    save_output(df, OUTPUT_FILE)

    log.info(DIVIDER)
    log.info("  STEP 7/7 — Automated Validation Suite")
    log.info(DIVIDER)
    validation_results = run_validation_suite(df, report)

    # ── Print validation results ──────────────────────────────────────────────
    all_passed = True
    print()
    print("=" * 72)
    print("  VALIDATION SUITE RESULTS")
    print("=" * 72)
    for r in validation_results:
        print(r)
        if not r.passed:
            all_passed = False
    print("=" * 72)
    overall = "[ALL CHECKS PASSED]" if all_passed else "[ONE OR MORE CHECKS FAILED]"
    print(f"  Overall : {overall}")
    print("=" * 72)
    print()

    # ── Print audit summary ───────────────────────────────────────────────────
    print(report.banner())

    # ── Exit code reflects validation status ─────────────────────────────────
    if not all_passed:
        log.error("Pipeline completed with validation failures.")
        sys.exit(1)
    else:
        log.info("Pipeline completed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
