"""
build_retrieval_dataset.py
==========================
Produces data/processed/bioasq_pubmed_retrieval.csv

Sources
-------
- data/raw/BioASQ/*_golden.json                  (native BioASQ format, richest)
- data/raw/BioASQ/BioASQ-train-*.json            (SQuAD-style training sets)
- data/raw/pubmed_abstracts.csv                  (wide-format abstract corpus)
- data/raw/meddialog/HealthCareMagic-100k.json   (112k doctor Q&A, non-conversational)
- data/processed/healthchat_11k.csv              (single-turn Q&A pairs only)
- data/raw/medquad.csv                           (16k NIH/MedlinePlus Q&A)

Output columns
--------------
question_id, question, answer_type, ground_truth_answer, ideal_answer,
snippets_text, snippet_pmids, all_doc_pmids,
pubmed_title, pubmed_abstract, pubmed_topic,
plan_of_action, source_file
"""

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parent.parent
RAW_BIOASQ        = ROOT / "data" / "raw" / "BioASQ"
RAW_PUBMED        = ROOT / "data" / "raw" / "pubmed_abstracts.csv"
RAW_HCM           = ROOT / "data" / "raw" / "meddialog" / "HealthCareMagic-100k.json"
RAW_HEALTHCHAT    = ROOT / "data" / "processed" / "healthchat_11k.csv"
RAW_MEDQUAD       = ROOT / "data" / "raw" / "medquad.csv"
OUT_DIR           = ROOT / "data" / "processed"
OUT_PATH          = OUT_DIR / "bioasq_pubmed_retrieval.csv"

TOPIC_COLS = [
    "deep_learning", "covid_19", "human_connectome", "virtual_reality",
    "brain_machine_interfaces", "electroactive_polymers",
    "pedot_electrodes", "neuroprosthetics",
]
LINK_COLS = [f"{t}_links" for t in TOPIC_COLS]

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def pmid_from_url(url: str) -> str:
    """Extract numeric PMID from a PubMed URL."""
    if not isinstance(url, str):
        return ""
    m = re.search(r"pubmed/(\d+)", url)
    return m.group(1) if m else url.strip()


def flatten_exact_answer(exact_answer, answer_type: str) -> str:
    """
    Normalise exact_answer across BioASQ question types:
      factoid / list  -> list-of-lists  e.g. [["ans1","syn"], ["ans2"]]
      yesno           -> plain string   "yes" / "no"
      summary         -> absent / None
    """
    if exact_answer is None:
        return ""
    if isinstance(exact_answer, str):
        return exact_answer.strip()
    if isinstance(exact_answer, list):
        parts = []
        for item in exact_answer:
            if isinstance(item, list):
                parts.append(" / ".join(str(x) for x in item))
            else:
                parts.append(str(item))
        return " | ".join(parts)
    return str(exact_answer)


def first_ideal_answer(ideal_answer) -> str:
    if not ideal_answer:
        return ""
    if isinstance(ideal_answer, list):
        return ideal_answer[0].strip()
    return str(ideal_answer).strip()


def derive_plan_of_action(ideal_answer: str, answer_type: str,
                          exact_answer: str) -> str:
    """
    Build a structured 5-step Plan-of-Action (POA) for each question.
    """
    if not ideal_answer and not exact_answer:
        return ""

    summary = ideal_answer[:250].rstrip() + ("..." if len(ideal_answer) > 250 else "")

    if answer_type == "yesno":
        recommendation = (
            f"Clinical recommendation: The answer is '{exact_answer.upper()}'. "
            "Proceed or withhold the intervention accordingly based on clinical context."
        )
    elif answer_type == "factoid":
        recommendation = (
            f"Clinical recommendation: The specific entity/value is '{exact_answer}'. "
            "Use this as the target in downstream queries or treatment decisions."
        )
    elif answer_type == "list":
        items = [i.strip() for i in exact_answer.split(" | ") if i.strip()] \
                if exact_answer else []
        shown = "; ".join(items[:5])
        tail  = " (and more)" if len(items) > 5 else ""
        recommendation = (
            f"Clinical recommendation: Consider each of the {len(items)} identified "
            f"items: {shown}{tail}."
        )
    elif answer_type == "open_qa":
        recommendation = (
            "Clinical recommendation: Review the detailed answer above. "
            "Discuss findings with the treating clinician before taking action."
        )
    else:  # summary / unknown
        recommendation = (
            "Clinical recommendation: Review the synthesised evidence above "
            "and formulate a management plan with the treating clinician."
        )

    return (
        "[Step 1] Identify the clinical/research question.\n"
        "[Step 2] Retrieve supporting evidence from PubMed snippets.\n"
        f"[Step 3] Ground-truth answer: {exact_answer}\n"
        f"[Step 4] Synthesise: {summary}\n"
        f"[Step 5] {recommendation}"
    )


# ──────────────────────────────────────────────
# Task 2 — Parse BioASQ golden files
# ──────────────────────────────────────────────

def parse_golden_file(path: Path) -> list[dict]:
    """Parse native BioASQ golden JSON (has questions/snippets/ideal_answer)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for q in data.get("questions", []):
        qid          = q.get("id", "")
        body         = q.get("body", "")
        qtype        = q.get("type", "unknown")
        documents    = q.get("documents", [])
        snippets     = q.get("snippets", [])
        ideal_answer = first_ideal_answer(q.get("ideal_answer", ""))
        exact        = flatten_exact_answer(q.get("exact_answer", None), qtype)

        snippet_texts = [s.get("text", "").strip() for s in snippets]
        snippet_pmids = [pmid_from_url(s.get("document", "")) for s in snippets]
        all_pmids     = [pmid_from_url(d) for d in documents]

        records.append({
            "question_id"        : qid,
            "question"           : body,
            "answer_type"        : qtype,
            "ground_truth_answer": exact,
            "ideal_answer"       : ideal_answer,
            "snippets_text"      : " ||| ".join(snippet_texts),
            "snippet_pmids"      : "|".join(dict.fromkeys(snippet_pmids)),  # dedup
            "all_doc_pmids"      : "|".join(dict.fromkeys(all_pmids)),
            "source_file"        : path.name,
        })
    return records


print("=" * 60)
print("TASK 2 — Parsing BioASQ golden files")
print("=" * 60)

golden_files = sorted(RAW_BIOASQ.glob("*_golden.json"))
print(f"Found {len(golden_files)} golden files")

all_golden = []
for gf in golden_files:
    recs = parse_golden_file(gf)
    print(f"  {gf.name:45s} → {len(recs):4d} questions")
    all_golden.extend(recs)

df_golden = pd.DataFrame(all_golden)
print(f"\nTotal golden records : {len(df_golden):,}")
print(df_golden["answer_type"].value_counts().to_string())


# ──────────────────────────────────────────────
# Task 3 — Parse BioASQ train files (SQuAD-style)
# ──────────────────────────────────────────────

def parse_squad_file(path: Path, answer_type: str) -> list[dict]:
    """
    Parse SQuAD-style BioASQ file.
    Multiple paragraphs per question are aggregated.
    answer_type is 'factoid' or 'yesno'.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    q_map: dict[str, dict] = {}

    for group in data.get("data", []):
        for para in group.get("paragraphs", []):
            context = para.get("context", "").strip()
            for qa in para.get("qas", []):
                full_id  = qa.get("id", "")
                base_id  = re.sub(r"_\d{3}$", "", full_id)
                question = qa.get("question", "").strip()

                raw_answers = qa.get("answers", None)
                if answer_type == "yesno":
                    ans_text = raw_answers if isinstance(raw_answers, str) else ""
                else:
                    if isinstance(raw_answers, list) and raw_answers:
                        ans_text = raw_answers[0].get("text", "")
                    else:
                        ans_text = ""

                if base_id not in q_map:
                    q_map[base_id] = {
                        "question_id"        : base_id,
                        "question"           : question,
                        "answer_type"        : answer_type,
                        "ground_truth_answer": ans_text,
                        "ideal_answer"       : ans_text,
                        "snippet_pmids"      : "",
                        "all_doc_pmids"      : "",
                        "source_file"        : path.name,
                        "_ctxs"              : [context],
                    }
                else:
                    q_map[base_id]["_ctxs"].append(context)

    records = []
    for rec in q_map.values():
        ctxs = list(dict.fromkeys(rec.pop("_ctxs")))  # ordered dedup
        rec["snippets_text"] = " ||| ".join(ctxs)
        records.append(rec)
    return records


print("\n" + "=" * 60)
print("TASK 3 — Parsing BioASQ train files")
print("=" * 60)

factoid_train = sorted(RAW_BIOASQ.glob("BioASQ-train-factoid-*.json"))
yesno_train   = sorted(RAW_BIOASQ.glob("BioASQ-train-yesno-*.json"))

all_train = []
for tf in factoid_train:
    recs = parse_squad_file(tf, "factoid")
    print(f"  {tf.name:45s} → {len(recs):4d} questions")
    all_train.extend(recs)

for tf in yesno_train:
    recs = parse_squad_file(tf, "yesno")
    print(f"  {tf.name:45s} → {len(recs):4d} questions")
    all_train.extend(recs)

df_train = pd.DataFrame(all_train)
print(f"\nTotal train records : {len(df_train):,}")
print(df_train["answer_type"].value_counts().to_string())


# ──────────────────────────────────────────────
# Merge golden + train, deduplicate
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("MERGING golden + train")
print("=" * 60)

df_bioasq = pd.concat([df_golden, df_train], ignore_index=True)
before    = len(df_bioasq)
# Golden always takes precedence (it appeared first in concat)
df_bioasq = df_bioasq.drop_duplicates(subset="question_id", keep="first")
print(f"Combined: {before:,}  → after dedup: {len(df_bioasq):,}")
print(df_bioasq["answer_type"].value_counts().to_string())


# ──────────────────────────────────────────────
# Task 4 — Parse PubMed abstracts CSV (wide → long)
# ──────────────────────────────────────────────

def safe_parse_cell(val) -> tuple[str, str]:
    """
    Parse a pubmed_abstracts.csv topic cell.
    Returns (abstract_text, title).
    """
    if not isinstance(val, str) or not val.strip():
        return ("", "")
    try:
        result = ast.literal_eval(val.strip())
        if isinstance(result, tuple) and len(result) == 2:
            abstracts, title = result
            abstract = abstracts[0] if isinstance(abstracts, list) and abstracts else ""
            return (str(abstract).strip(), str(title).strip())
    except Exception:
        pass
    # Fallback: treat the whole string as a title excerpt
    return ("", str(val)[:200])


print("\n" + "=" * 60)
print("TASK 4 — Parsing PubMed abstracts CSV")
print("=" * 60)

df_pub_raw = pd.read_csv(RAW_PUBMED)
print(f"Raw CSV shape: {df_pub_raw.shape}")

pubmed_records = []
for _, row in df_pub_raw.iterrows():
    for topic, link_col in zip(TOPIC_COLS, LINK_COLS):
        cell = row.get(topic, "")
        url  = str(row.get(link_col, "")).strip()
        pmid = pmid_from_url(url)

        abstract, title = safe_parse_cell(cell)

        if not pmid and not abstract and not title:
            continue

        pubmed_records.append({
            "pmid"           : pmid,
            "pubmed_title"   : title,
            "pubmed_abstract": abstract,
            "pubmed_topic"   : topic,
        })

df_pubmed = pd.DataFrame(pubmed_records)
df_pubmed = df_pubmed[df_pubmed["pmid"].str.len() > 0]
df_pubmed_dedup = df_pubmed.drop_duplicates(subset="pmid", keep="first")

print(f"Long-format rows : {len(df_pubmed):,}")
print(f"After PMID dedup : {len(df_pubmed_dedup):,}")
print("Topic breakdown:")
print(df_pubmed["pubmed_topic"].value_counts().to_string())


# ──────────────────────────────────────────────
# Task 5 — Join BioASQ → PubMed via PMID
# ──────────────────────────────────────────────

def first_pmid(pmid_str: str) -> str:
    """Return the first PMID from a pipe-separated string."""
    if not isinstance(pmid_str, str):
        return ""
    parts = [p.strip() for p in pmid_str.split("|") if p.strip()]
    return parts[0] if parts else ""


print("\n" + "=" * 60)
print("TASK 5 — Joining BioASQ → PubMed abstracts")
print("=" * 60)

df_bioasq["_join_pmid"] = df_bioasq["snippet_pmids"].apply(first_pmid)
# Fallback: use first doc PMID when no snippet PMIDs available
no_pmid = df_bioasq["_join_pmid"] == ""
df_bioasq.loc[no_pmid, "_join_pmid"] = (
    df_bioasq.loc[no_pmid, "all_doc_pmids"].apply(first_pmid)
)

print(f"Questions with join PMID  : {(df_bioasq['_join_pmid'] != '').sum():,}")
print(f"Questions without any PMID: {(df_bioasq['_join_pmid'] == '').sum():,}")

df_merged = df_bioasq.merge(
    df_pubmed_dedup[["pmid", "pubmed_title", "pubmed_abstract", "pubmed_topic"]],
    left_on="pmid" if "pmid" in df_bioasq.columns else "_join_pmid",
    right_on="pmid",
    how="left",
)

# Correct column name after merge
if "_join_pmid" in df_merged.columns:
    df_merged = df_merged.merge(
        df_pubmed_dedup[["pmid", "pubmed_title", "pubmed_abstract", "pubmed_topic"]],
        left_on="_join_pmid",
        right_on="pmid",
        how="left",
    ) if "pubmed_title" not in df_merged.columns else df_merged

# Re-do cleanly
df_merged = df_bioasq.merge(
    df_pubmed_dedup[["pmid", "pubmed_title", "pubmed_abstract", "pubmed_topic"]],
    left_on="_join_pmid",
    right_on="pmid",
    how="left",
).drop(columns=["_join_pmid", "pmid"], errors="ignore")

for col in ["pubmed_title", "pubmed_abstract", "pubmed_topic"]:
    df_merged[col] = df_merged[col].fillna("")

matched = (df_merged["pubmed_abstract"] != "").sum()
print(f"Rows with PubMed abstract : {matched:,} / {len(df_merged):,} "
      f"({matched/len(df_merged)*100:.1f}%)")

# ── Secondary enrichment: try ALL snippet PMIDs per question ──────────────────
# For questions that didn't match on the first PMID, try each PMID in order.
print("\nRunning secondary PMID enrichment (all snippet PMIDs)...")

pubmed_lookup = df_pubmed_dedup.set_index("pmid")[
    ["pubmed_title", "pubmed_abstract", "pubmed_topic"]
].to_dict("index")

def enrich_from_all_pmids(row) -> dict:
    """Try every PMID in snippet_pmids + all_doc_pmids until we get a match."""
    if row["pubmed_abstract"]:       # already matched
        return {}
    pmid_str = "|".join(filter(None, [row["snippet_pmids"], row["all_doc_pmids"]]))
    for pmid in pmid_str.split("|"):
        pmid = pmid.strip()
        if pmid in pubmed_lookup:
            return pubmed_lookup[pmid]
    return {}

enriched = df_merged.apply(enrich_from_all_pmids, axis=1)
for col in ["pubmed_title", "pubmed_abstract", "pubmed_topic"]:
    fill_mask = (df_merged["pubmed_abstract"] == "") & enriched.apply(
        lambda d: bool(d.get(col, "")) if isinstance(d, dict) else False
    )
    df_merged.loc[fill_mask, col] = enriched[fill_mask].apply(
        lambda d: d.get(col, "") if isinstance(d, dict) else ""
    )

matched2 = (df_merged["pubmed_abstract"] != "").sum()
print(f"After secondary enrichment: {matched2:,} / {len(df_merged):,} "
      f"({matched2/len(df_merged)*100:.1f}%)")


# ──────────────────────────────────────────────
# Task 6 — Generate POA & finalise
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 6 — Generating Plan-of-Action column")
print("=" * 60)

df_merged["plan_of_action"] = df_merged.apply(
    lambda r: derive_plan_of_action(
        ideal_answer=r["ideal_answer"],
        answer_type=r["answer_type"],
        exact_answer=r["ground_truth_answer"],
    ),
    axis=1,
)

poa_count = (df_merged["plan_of_action"] != "").sum()
print(f"POA generated for {poa_count:,} / {len(df_merged):,} rows.")

FINAL_COLS = [
    "question_id", "question", "answer_type",
    "ground_truth_answer", "ideal_answer",
    "snippets_text", "snippet_pmids", "all_doc_pmids",
    "pubmed_title", "pubmed_abstract", "pubmed_topic",
    "plan_of_action", "source_file",
]

df_final = df_merged[FINAL_COLS].copy()

# Drop rows with no question text
before = len(df_final)
df_final = df_final[df_final["question"].str.strip() != ""]
print(f"Dropped {before - len(df_final)} empty-question rows.")

df_final = df_final.reset_index(drop=True)

print(f"\nFinal dataset shape: {df_final.shape}")
print("\nAnswer type breakdown:")
print(df_final["answer_type"].value_counts().to_string())
print("\nCompleteness:")
print(f"  ground_truth_answer : {(df_final['ground_truth_answer'] != '').sum():,}")
print(f"  ideal_answer        : {(df_final['ideal_answer'] != '').sum():,}")
print(f"  snippets_text       : {(df_final['snippets_text'] != '').sum():,}")
print(f"  pubmed_abstract     : {(df_final['pubmed_abstract'] != '').sum():,}")
print(f"  plan_of_action      : {(df_final['plan_of_action'] != '').sum():,}")


# ──────────────────────────────────────────────
# Task 7 — Save
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 7 — Saving dataset")
print("=" * 60)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False, encoding="utf-8")
size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"Saved  → {OUT_PATH}")
print(f"Size   : {size_mb:.2f} MB")
print(f"Rows   : {len(df_final):,}")
print(f"Columns: {list(df_final.columns)}")

# Print 2 sample records (one factoid, one yesno)
print("\n── SAMPLE FACTOID RECORD ──")
factoid_ex = df_final[df_final["answer_type"] == "factoid"].iloc[0]
print(f"  question            : {factoid_ex['question']}")
print(f"  ground_truth_answer : {factoid_ex['ground_truth_answer']}")
print(f"  ideal_answer        : {factoid_ex['ideal_answer'][:150]}")
print(f"  pubmed_topic        : {factoid_ex['pubmed_topic']}")
print(f"  plan_of_action snippet: {factoid_ex['plan_of_action'][:200]}")

print("\n── SAMPLE YESNO RECORD ──")
yesno_ex = df_final[df_final["answer_type"] == "yesno"].iloc[0]
print(f"  question            : {yesno_ex['question']}")
print(f"  ground_truth_answer : {yesno_ex['ground_truth_answer']}")
print(f"  ideal_answer        : {yesno_ex['ideal_answer'][:150]}")
print(f"  plan_of_action snippet: {yesno_ex['plan_of_action'][:200]}")

print("\nDone.")

FINAL_COLS = [
    "question_id", "question", "answer_type",
    "ground_truth_answer", "ideal_answer",
    "snippets_text", "snippet_pmids", "all_doc_pmids",
    "pubmed_title", "pubmed_abstract", "pubmed_topic",
    "plan_of_action", "source_file",
]

df_bioasq_final = df_merged[FINAL_COLS].copy()
df_bioasq_final = df_bioasq_final[df_bioasq_final["question"].str.strip() != ""]
df_bioasq_final = df_bioasq_final.reset_index(drop=True)

print(f"BioASQ base rows: {len(df_bioasq_final):,}")


# ──────────────────────────────────────────────
# TASK 1 — HealthCareMagic-100k  (non-conversational)
# ──────────────────────────────────────────────
# Format: list of {"instruction": ..., "input": patient question, "output": doctor answer}
# Every record is a single independent Q&A — structurally non-conversational.
# We use `input` as the question and `output` as the ground-truth answer.
# The instruction is a generic prompt prefix, so we drop it.

print("\n" + "=" * 60)
print("TASK 1 — HealthCareMagic-100k (non-conversational Q&A)")
print("=" * 60)

import hashlib

def make_id(prefix: str, text: str, idx: int) -> str:
    """Stable deterministic ID: prefix + first 12 chars of sha1 of text."""
    h = hashlib.sha1(f"{idx}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


with open(RAW_HCM, encoding="utf-8") as f:
    hcm_raw = json.load(f)

hcm_records = []
for i, rec in enumerate(hcm_raw):
    question = str(rec.get("input", "")).strip()
    answer   = str(rec.get("output", "")).strip()

    # Skip if either field is empty or the question is just the generic instruction
    if not question or not answer or len(question) < 10:
        continue

    # Truncate extremely long answers to 4000 chars (they average ~700 chars anyway)
    answer = answer[:4000]

    qid = make_id("hcm", question, i)

    hcm_records.append({
        "question_id"        : qid,
        "question"           : question,
        "answer_type"        : "open_qa",
        "ground_truth_answer": answer,
        "ideal_answer"       : answer,
        "snippets_text"      : "",
        "snippet_pmids"      : "",
        "all_doc_pmids"      : "",
        "pubmed_title"       : "",
        "pubmed_abstract"    : "",
        "pubmed_topic"       : "",
        "plan_of_action"     : derive_plan_of_action(answer, "open_qa", answer[:200]),
        "source_file"        : "HealthCareMagic-100k.json",
    })

df_hcm = pd.DataFrame(hcm_records)
print(f"HealthCareMagic records loaded   : {len(hcm_raw):,}")
print(f"After filtering empty rows       : {len(df_hcm):,}")
print(f"Avg question length (chars)      : {df_hcm['question'].str.len().mean():.0f}")
print(f"Avg answer length (chars)        : {df_hcm['ideal_answer'].str.len().mean():.0f}")
print(f"Sample question: {df_hcm['question'].iloc[0][:120]}")
print(f"Sample answer  : {df_hcm['ideal_answer'].iloc[0][:120]}")


# ──────────────────────────────────────────────
# TASK 2 — HealthChat 11k  (single-turn pairs only)
# ──────────────────────────────────────────────
# Keep only dialogues with exactly 2 turns (1 user + 1 assistant).
# Multi-turn dialogues are conversational and excluded.

print("\n" + "=" * 60)
print("TASK 2 — HealthChat 11k (single-turn Q&A pairs)")
print("=" * 60)

df_hc = pd.read_csv(RAW_HEALTHCHAT)

turns_per = df_hc.groupby("dialogue_id")["turn_id"].count()
qa_ids    = turns_per[turns_per == 2].index

df_hc_qa = df_hc[df_hc["dialogue_id"].isin(qa_ids)].copy()

# Pivot to question / answer columns
hc_users = (
    df_hc_qa[df_hc_qa["speaker"] == "user"][["dialogue_id", "utterance"]]
    .rename(columns={"utterance": "question"})
)
hc_assts = (
    df_hc_qa[df_hc_qa["speaker"] == "assistant"][["dialogue_id", "utterance"]]
    .rename(columns={"utterance": "answer"})
)
df_hc_pairs = hc_users.merge(hc_assts, on="dialogue_id")

# Drop pairs where question or answer is empty / very short
df_hc_pairs = df_hc_pairs[
    (df_hc_pairs["question"].str.strip().str.len() >= 10) &
    (df_hc_pairs["answer"].str.strip().str.len() >= 20)
].copy()

# Truncate answers to 4000 chars
df_hc_pairs["answer"] = df_hc_pairs["answer"].str[:4000]

hc_out_records = []
for _, row in df_hc_pairs.iterrows():
    q   = row["question"].strip()
    a   = row["answer"].strip()
    qid = row["dialogue_id"]       # already a UUID from the original pipeline
    hc_out_records.append({
        "question_id"        : qid,
        "question"           : q,
        "answer_type"        : "open_qa",
        "ground_truth_answer": a,
        "ideal_answer"       : a,
        "snippets_text"      : "",
        "snippet_pmids"      : "",
        "all_doc_pmids"      : "",
        "pubmed_title"       : "",
        "pubmed_abstract"    : "",
        "pubmed_topic"       : "",
        "plan_of_action"     : derive_plan_of_action(a, "open_qa", a[:200]),
        "source_file"        : "healthchat_11k.csv",
    })

df_hc_out = pd.DataFrame(hc_out_records)

total_hc     = len(turns_per)
single_count = len(qa_ids)
multi_count  = (turns_per > 2).sum()
print(f"Total HealthChat dialogues       : {total_hc:,}")
print(f"  Single-turn (Q+A, kept)        : {single_count:,}  ({single_count/total_hc*100:.1f}%)")
print(f"  Multi-turn (conversational, dropped): {multi_count:,}  ({multi_count/total_hc*100:.1f}%)")
print(f"After quality filter             : {len(df_hc_out):,}")
print(f"Sample question: {df_hc_out['question'].iloc[0][:120]}")
print(f"Sample answer  : {df_hc_out['ideal_answer'].iloc[0][:120]}")


# ──────────────────────────────────────────────
# TASK 3 — MedQuAD  (NIH/MedlinePlus Q&A)
# ──────────────────────────────────────────────
# Format: question, answer, source, focus_area
# Pure Q&A from authoritative NIH sources — ideal ground truth.

print("\n" + "=" * 60)
print("TASK 3 — MedQuAD (NIH Q&A pairs)")
print("=" * 60)

df_mq = pd.read_csv(RAW_MEDQUAD)
print(f"MedQuAD raw shape: {df_mq.shape}")
print(f"Columns          : {list(df_mq.columns)}")
print(f"Focus areas (top 10):")
print(df_mq["focus_area"].value_counts().head(10).to_string())

# Drop rows missing question or answer
df_mq = df_mq.dropna(subset=["question", "answer"])
df_mq = df_mq[
    (df_mq["question"].str.strip().str.len() >= 5) &
    (df_mq["answer"].str.strip().str.len() >= 10)
].copy()

mq_records = []
for i, row in enumerate(df_mq.itertuples(index=False)):
    q      = str(row.question).strip()
    a      = str(row.answer).strip()[:4000]
    source = str(row.source).strip() if hasattr(row, "source") else ""
    topic  = str(row.focus_area).strip() if hasattr(row, "focus_area") else ""
    qid    = make_id("mq", q, i)

    mq_records.append({
        "question_id"        : qid,
        "question"           : q,
        "answer_type"        : "open_qa",
        "ground_truth_answer": a,
        "ideal_answer"       : a,
        "snippets_text"      : "",
        "snippet_pmids"      : "",
        "all_doc_pmids"      : "",
        "pubmed_title"       : "",
        "pubmed_abstract"    : "",
        "pubmed_topic"       : topic,    # focus_area maps naturally to topic
        "plan_of_action"     : derive_plan_of_action(a, "open_qa", a[:200]),
        "source_file"        : f"medquad.csv [{source}]",
    })

df_mq_out = pd.DataFrame(mq_records)
print(f"\nAfter quality filter             : {len(df_mq_out):,}")
print(f"Avg question length (chars)      : {df_mq_out['question'].str.len().mean():.0f}")
print(f"Avg answer length (chars)        : {df_mq_out['ideal_answer'].str.len().mean():.0f}")
print(f"Sample question: {df_mq_out['question'].iloc[0][:120]}")
print(f"Sample answer  : {df_mq_out['ideal_answer'].iloc[0][:120]}")


# ──────────────────────────────────────────────
# TASK 4 — Combine all sources
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 4 — Combining all sources")
print("=" * 60)

df_final = pd.concat(
    [df_bioasq_final, df_hcm, df_hc_out, df_mq_out],
    ignore_index=True,
)

# Ensure no duplicate question_ids across sources
before_dedup = len(df_final)
df_final = df_final.drop_duplicates(subset="question_id", keep="first")
print(f"Combined rows (before dedup) : {before_dedup:,}")
print(f"After dedup on question_id   : {len(df_final):,}")

# Fill any remaining NaN with empty string
for col in FINAL_COLS:
    df_final[col] = df_final[col].fillna("").astype(str)

df_final = df_final.reset_index(drop=True)

print("\nRows by source_file:")
src_counts = df_final["source_file"].value_counts()
print(src_counts.to_string())

print("\nRows by answer_type:")
print(df_final["answer_type"].value_counts().to_string())

print("\nCompleteness:")
for col in ["question", "ground_truth_answer", "ideal_answer",
            "snippets_text", "pubmed_abstract", "plan_of_action"]:
    pct = (df_final[col].str.strip() != "").mean() * 100
    print(f"  {col:30s}: {pct:5.1f}%")


# ──────────────────────────────────────────────
# TASK 5 — Save final dataset
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 5 — Saving final dataset")
print("=" * 60)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False, encoding="utf-8")
size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"Saved  → {OUT_PATH}")
print(f"Size   : {size_mb:.2f} MB")
print(f"Rows   : {len(df_final):,}")
print(f"Columns: {list(df_final.columns)}")

print("\n── SAMPLE BioASQ FACTOID ──")
ex = df_final[df_final["answer_type"] == "factoid"].iloc[0]
print(f"  Q  : {ex['question']}")
print(f"  A  : {ex['ground_truth_answer'][:120]}")
print(f"  POA: {ex['plan_of_action'][:180]}")

print("\n── SAMPLE HealthCareMagic ──")
ex = df_final[df_final["source_file"] == "HealthCareMagic-100k.json"].iloc[0]
print(f"  Q  : {ex['question'][:120]}")
print(f"  A  : {ex['ideal_answer'][:120]}")
print(f"  POA: {ex['plan_of_action'][:180]}")

print("\n── SAMPLE HealthChat ──")
ex = df_final[df_final["source_file"] == "healthchat_11k.csv"].iloc[0]
print(f"  Q  : {ex['question'][:120]}")
print(f"  A  : {ex['ideal_answer'][:120]}")
print(f"  POA: {ex['plan_of_action'][:180]}")

print("\n── SAMPLE MedQuAD ──")
ex = df_final[df_final["source_file"].str.startswith("medquad")].iloc[0]
print(f"  Q  : {ex['question'][:120]}")
print(f"  A  : {ex['ideal_answer'][:120]}")
print(f"  POA: {ex['plan_of_action'][:180]}")

print("\nDone.")
