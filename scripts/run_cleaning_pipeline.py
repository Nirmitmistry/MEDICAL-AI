"""
run_cleaning_pipeline.py
Standalone version of notebooks/06_cleaning_pipeline.ipynb

Stages:
  A  — Template / boilerplate filtering
  B  — Near-duplicate detection (MinHash LSH)
  C  — Entity normalisation (lowercase + lemmatise)
  D1 — Rule-based NER label re-assignment
  D2 — scispaCy UMLS linker (skipped gracefully if not installed)
  E  — 200-turn stratified spot-check

Input:  data/processed/integrated_conversations_ner_validated.csv
Output: data/processed/integrated_conversations_clean.csv
        data/processed/spotcheck_200.csv
        data/processed/entity_norm_map.csv

Run from repo root:
    python scripts/run_cleaning_pipeline.py
"""

import re
import json
import hashlib
import warnings
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "processed"
INPUT_CSV    = DATA_DIR / "integrated_conversations_ner_validated.csv"
OUTPUT_CSV   = DATA_DIR / "integrated_conversations_clean.csv"
SPOTCHECK    = DATA_DIR / "spotcheck_200.csv"
NORM_MAP_CSV = DATA_DIR / "entity_norm_map.csv"

# ── Settings ──────────────────────────────────────────────────────────────────
RANDOM_SEED        = 42
MINHASH_NUM_PERM   = 64   # reduced from 128 for speed; still high accuracy at 0.85 threshold
MINHASH_THRESHOLD  = 0.85
MINHASH_SHINGLE_K  = 5
N_SAMPLE           = 200
MIN_UTT_LEN        = 20

# =============================================================================
# LOAD
# =============================================================================
print(f"\n{'='*60}")
print("LOADING DATA")
print(f"{'='*60}")

df = pd.read_csv(INPUT_CSV, low_memory=False,
                 dtype={"original_id": str, "dialogue_id": str})

df["utterance"]          = df["utterance"].fillna("").astype(str).str.strip()
df["extracted_entities"] = df["extracted_entities"].fillna("").astype(str)
df["entity_labels"]      = df["entity_labels"].fillna("").astype(str)
df["source_dataset"]     = df["source_dataset"].fillna("").astype(str)

def split_pipe(val):
    if not val or val.strip() == "":
        return []
    return [t.strip() for t in val.split("|") if t.strip()]

df["ents_list"]   = df["extracted_entities"].apply(split_pipe)
df["labels_list"] = df["entity_labels"].apply(split_pipe)

N_ORIG = len(df)
print(f"Loaded {N_ORIG:,} rows from {INPUT_CSV.name}")
print(f"Columns: {list(df.columns)}")
print(f"\nSource breakdown:\n{df['source_dataset'].value_counts().to_string()}")

# =============================================================================
# STAGE A — Template / Boilerplate Filtering
# =============================================================================
print(f"\n{'='*60}")
print("STAGE A — Template / Boilerplate Filtering")
print(f"{'='*60}")

SIGNOFF_PATTERNS = [
    r"\bfurther clarifications and consultations on chat\s*doctor are welcome\.?\b",
    r"\bif you do not have any clarifications,? you can close the discussion and rate the answer\.?\b",
    r"\bi am happy to help any\s*time\.?\b",
    r"\bhope my answer was helpful for you\.?\b",
    r"\bhope i have answered your query\.?\b",
    r"\bhope this (?:will )?help(?:s)? you(?: in resolving your query)?\.?\b",
    r"\bthanks for your query\.?\b",
    r"\bthank you for (?:consulting|posting on) (?:in )?chat\s*doctor\.?\b",
    r"\bwish your(?:\s+\w+)? good health\.?\b",
    r"\bwish you (?:best|good)(?: of health)?\.?\b",
    r"\bchat\s*doctor\.\s*\(?\s*(?:mbbs|md|ms|mch|dnb)(?:\s*,\s*(?:mbbs|md|ms|mch|dnb))*\s*\)?",
    r"\bchat\s*doctor\.\s*$",
    r"\bkindly spare some time to rate my answer\b",
    r"\bplease do not hesitate to ask if you have any further questions?\.?\b",
    r"\bfeel free to (?:ask|consult)(?: me)?(?: again)? (?:anytime|if (?:you have|there are))\b",
    r"\btake care\.?\s*$",
    r"\bregards\s*[-–]\s*$",
    r"\bregards\.?\s*$",
    r"\bback to me for any follow[\s-]?up queries anytime\.?\b",
]
_SIGNOFF_RE = re.compile(
    "|".join(f"({p})" for p in SIGNOFF_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)

WHOLE_DROP_EXACT = {
    "▌", "hi", "hello", "hello!", "hi!", "hi.", "hello.",
    "hello! how can i assist you today?",
    "hello! how can i help you today?",
    "hi! how can i assist you today?",
    "hi! how can i help you today?",
    "-", "--", "...", ".",
}

WHOLE_DROP_PATTERNS = [
    r"^chat\s*doctor\.?\s*(?:\(?\s*(?:mbbs|md|ms|mch|dnb|dgo)[^)]*\)?)?\s*$",
    r"^[\d\s.,;:%/\(\)\-\+]+$",
    r"^(?:yes|no|ok|okay|sure|thanks|thank\s*you|alright|alright\.|noted|noted\.)$",
]
_WHOLE_DROP_RE = re.compile(
    "|".join(f"({p})" for p in WHOLE_DROP_PATTERNS),
    flags=re.IGNORECASE,
)

def scrub_signoffs(text: str) -> str:
    cleaned = _SIGNOFF_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"[.\s]+$", "", cleaned).strip()
    return cleaned

def is_whole_drop(text: str) -> bool:
    norm = text.strip().lower()
    if norm == "":
        return True
    if norm in WHOLE_DROP_EXACT:
        return True
    if _WHOLE_DROP_RE.fullmatch(norm):
        return True
    return False

df["utterance_clean"] = df["utterance"].progress_apply(scrub_signoffs)
df["_drop_template"]  = df["utterance_clean"].apply(is_whole_drop)

n_scrubbed      = (df["utterance_clean"] != df["utterance"]).sum()
n_template_drop = df["_drop_template"].sum()
print(f"Sign-offs scrubbed : {n_scrubbed:,}")
print(f"Whole-drop rows    : {n_template_drop:,}")

df_a = df[~df["_drop_template"]].copy()
df_a["utterance"] = df_a["utterance_clean"]
df_a = df_a.drop(columns=["utterance_clean", "_drop_template"])

print(f"Rows before A : {N_ORIG:,}")
print(f"Rows after  A : {len(df_a):,}  (-{N_ORIG - len(df_a):,})")

# =============================================================================
# STAGE B — Near-Duplicate Detection (MinHash LSH)
# =============================================================================
print(f"\n{'='*60}")
print("STAGE B — Near-Duplicate Detection (MinHash LSH)")
print(f"{'='*60}")

from datasketch import MinHash, MinHashLSH

def make_shingles(text: str, k: int = MINHASH_SHINGLE_K) -> set:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return {t[i:i+k] for i in range(max(1, len(t) - k + 1))}

def make_minhash(shingles: set, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m

MAX_UTT_LEN = 500   # only near-dup check utterances under this length
                    # long clinical answers are unique; canned phrases are short
print(f"Building MinHash index (utterances 20-{MAX_UTT_LEN} chars) …")
lsh             = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
idx_to_minhash  = {}
inserted_keys   = set()

for row_idx, row in tqdm(df_a.iterrows(), total=len(df_a), desc="MinHash"):
    utt = row["utterance"]
    if len(utt) < MIN_UTT_LEN or len(utt) > MAX_UTT_LEN:
        continue
    shingles = make_shingles(utt)
    if not shingles:
        continue
    mh  = make_minhash(shingles)
    key = str(row_idx)
    idx_to_minhash[row_idx] = mh
    try:
        lsh.insert(key, mh)
        inserted_keys.add(row_idx)
    except ValueError:
        pass

print(f"Inserted {len(inserted_keys):,} utterances into LSH index")

class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[rb] = ra

uf = UnionFind()
for row_idx, mh in tqdm(idx_to_minhash.items(), desc="LSH query"):
    for n_key in lsh.query(mh):
        n_idx = int(n_key)
        if n_idx != row_idx:
            uf.union(row_idx, n_idx)

cluster_map        = defaultdict(list)
for row_idx in inserted_keys:
    cluster_map[uf.find(row_idx)].append(row_idx)

near_dup_clusters  = {k: v for k, v in cluster_map.items() if len(v) > 1}
total_near_dup     = sum(len(v) for v in near_dup_clusters.values())
print(f"Near-dup clusters : {len(near_dup_clusters):,}")
print(f"Rows in clusters  : {total_near_dup:,}")

rows_to_drop = set()
for root, cluster in near_dup_clusters.items():
    sorted_cluster = sorted(cluster,
                            key=lambda i: len(df_a.loc[i, "utterance"]),
                            reverse=True)
    rows_to_drop.update(sorted_cluster[1:])

df_b = df_a.drop(index=list(rows_to_drop)).copy()
print(f"Rows before B : {len(df_a):,}")
print(f"Rows after  B : {len(df_b):,}  (-{len(rows_to_drop):,})")

# =============================================================================
# STAGE C — Entity Normalisation
# =============================================================================
print(f"\n{'='*60}")
print("STAGE C — Entity Normalisation (Lowercase + Lemmatise)")
print(f"{'='*60}")

import spacy
try:
    nlp_lemma = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("en_core_web_sm loaded")
except OSError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp_lemma = spacy.load("en_core_web_sm", disable=["parser", "ner"])

all_raw_entities = list({
    ent
    for ents in df_b["ents_list"]
    for ent in ents
    if ent.strip()
})

print(f"Unique raw entity strings: {len(all_raw_entities):,}")
print("Lemmatising …")

norm_list = []
for doc in tqdm(nlp_lemma.pipe(all_raw_entities, batch_size=512),
                total=len(all_raw_entities), desc="Lemmatise"):
    norm_list.append(" ".join(t.lemma_ for t in doc).strip().lower())

norm_map = dict(zip(all_raw_entities, norm_list))

norm_df = pd.DataFrame({
    "raw_entity":        all_raw_entities,
    "normalised_entity": norm_list,
})
norm_df["changed"] = (norm_df["raw_entity"].str.lower().str.strip()
                      != norm_df["normalised_entity"])
norm_df.to_csv(NORM_MAP_CSV, index=False)
print(f"entity_norm_map.csv saved: {len(norm_df):,} entries, "
      f"{norm_df['changed'].sum():,} changed")

def apply_norm(ents_list):
    return [norm_map.get(e, e.strip().lower()) for e in ents_list]

df_b["ents_norm"] = df_b["ents_list"].apply(apply_norm)
df_b["extracted_entities"] = df_b["ents_norm"].apply(
    lambda lst: " | ".join(lst) if lst else ""
)

# =============================================================================
# STAGE D1 — Rule-Based Label Re-assignment
# =============================================================================
print(f"\n{'='*60}")
print("STAGE D1 — Rule-Based NER Label Re-assignment")
print(f"{'='*60}")

SYMPTOM_VOCAB = {
    "pain","ache","aching","burning","stinging","tingling","numbness",
    "soreness","tenderness","cramp","spasm","pressure",
    "cough","sneeze","sneezing","wheeze","shortness of breath","sob",
    "breathlessness","dyspnea","sputum","phlegm","congestion","runny nose",
    "nausea","vomiting","diarrhea","diarrhoea","constipation","bloating",
    "heartburn","indigestion","belching","flatulence",
    "fever","fatigue","tiredness","weakness","dizziness","vertigo",
    "headache","migraine","insomnia","sweating","chills","itching",
    "itchiness","rash","hives","swelling","edema","oedema",
    "anxiety","depression","mood swing","irritability","confusion",
    "memory loss","brain fog","hallucination",
    "bleeding","bruising","discharge","pus","jaundice","pallor","cyanosis","sore",
}
ANATOMY_VOCAB = {
    "tooth","teeth","gum","tongue","throat","tonsil","tonsils",
    "neck","ear","ears","nose","eye","eyes","scalp","jaw",
    "chest","breast","nipple","abdomen","stomach","belly","navel","groin","pelvis",
    "back","spine","shoulder","arm","elbow","wrist","hand","finger",
    "thumb","hip","knee","ankle","foot","toe","heel","calf","thigh","leg","buttock",
    "liver","kidney","lung","lungs","heart","brain","spleen",
    "pancreas","gallbladder","bladder","uterus","ovary","ovaries",
    "prostate","thyroid","adrenal","colon","rectum","appendix",
    "trachea","larynx","esophagus","oesophagus","duodenum","ileum",
    "jejunum","cecum","sigmoid","anus","vagina","cervix","penis",
    "testis","testes","bone","cartilage","tendon","ligament","muscle",
    "nerve","artery","vein","capillary","lymph node","lymph",
    "skin","nail","nails","hair",
}
BEHAVIOR_VOCAB = {
    "smoking","smoke","tobacco","cigarette","vaping","chewing tobacco",
    "alcohol","drinking","alcoholism","drug use","substance use",
    "recreational drug","heroin","cocaine","methamphetamine","cannabis",
    "marijuana","sedentary","exercise","physical activity",
}

LABEL_OVERRIDE = {}
for e in SYMPTOM_VOCAB:  LABEL_OVERRIDE[e] = "SYMPTOM"
for e in ANATOMY_VOCAB:  LABEL_OVERRIDE[e] = "ANATOMY"
for e in BEHAVIOR_VOCAB: LABEL_OVERRIDE[e] = "BEHAVIOR"

def relabel_entities(ents_norm, labels):
    new_labels = []
    for ent, lbl in zip(ents_norm, labels):
        override = LABEL_OVERRIDE.get(ent.strip().lower())
        if override and lbl in ("DISEASE", "CHEMICAL"):
            new_labels.append(override)
        else:
            new_labels.append(lbl)
    return new_labels

df_b["labels_updated"] = df_b.apply(
    lambda r: relabel_entities(r["ents_norm"], r["labels_list"]), axis=1
)

n_relabelled = sum(
    sum(a != b for a, b in zip(old, new))
    for old, new in zip(df_b["labels_list"], df_b["labels_updated"])
)
print(f"Entities relabelled by rule-based override: {n_relabelled:,}")

df_b["entity_labels"] = df_b["labels_updated"].apply(
    lambda lst: " | ".join(lst) if lst else ""
)

# =============================================================================
# STAGE D2 — scispaCy UMLS (graceful skip)
# =============================================================================
print(f"\n{'='*60}")
print("STAGE D2 — scispaCy UMLS EntityLinker")
print(f"{'='*60}")

UMLS_AVAILABLE = False
try:
    import scispacy  # noqa
    from scispacy.linking import EntityLinker  # noqa
    nlp_sci = spacy.load("en_ner_bc5cdr_md")
    nlp_sci.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.80,
        "filter_for_definitions": False,
    })
    UMLS_AVAILABLE = True
    print("scispaCy UMLS EntityLinker loaded")
except Exception as e:
    print(f"UMLS linker not available ({type(e).__name__}: {e}) — skipping D2")

df_b["umls_cuis"] = ""
if UMLS_AVAILABLE:
    unique_ents = list({e for ents in df_b["ents_norm"] for e in ents if e.strip()})
    cui_map = {}
    for doc in tqdm(nlp_sci.pipe(unique_ents, batch_size=256, disable=["ner"]),
                    total=len(unique_ents), desc="UMLS link"):
        all_cuis = [kb[0] for ent in doc.ents for kb in ent._.kb_ents]
        cui_map[doc.text] = all_cuis[0] if all_cuis else ""
    df_b["umls_cuis"] = df_b["ents_norm"].apply(
        lambda ents: " | ".join(cui_map.get(e, "") for e in ents) if ents else ""
    )
    n_linked = sum(1 for v in cui_map.values() if v)
    print(f"Entities with UMLS CUI: {n_linked:,} / {len(unique_ents):,}")

# =============================================================================
# STAGE E — 200-Turn Stratified Spot-Check
# =============================================================================
print(f"\n{'='*60}")
print("STAGE E — 200-Turn Stratified Spot-Check")
print(f"{'='*60}")

df_b["has_entity"] = df_b["ents_norm"].apply(lambda x: len(x) > 0)
df_b["_stratum"] = (
    df_b["source_dataset"] + "::" +
    df_b["speaker"] + "::" +
    df_b["has_entity"].map({True: "has_ent", False: "no_ent"})
)

strata_counts   = df_b["_stratum"].value_counts()
stratum_weights = strata_counts / strata_counts.sum()
stratum_alloc   = (stratum_weights * N_SAMPLE).round().astype(int)
diff = N_SAMPLE - stratum_alloc.sum()
if diff != 0:
    stratum_alloc[stratum_alloc.idxmax()] += diff

sample_parts = []
for stratum, n_draw in stratum_alloc.items():
    pool = df_b[df_b["_stratum"] == stratum]
    actual = min(n_draw, len(pool))
    if actual > 0:
        sample_parts.append(pool.sample(n=actual, random_state=RANDOM_SEED))

spotcheck_df = pd.concat(sample_parts).sample(frac=1, random_state=RANDOM_SEED).copy()
spotcheck_df["_review_label"] = ""
spotcheck_df["_review_notes"] = ""

review_cols = ["dialogue_id","turn_id","speaker","source_dataset",
               "utterance","extracted_entities","entity_labels","umls_cuis",
               "_review_label","_review_notes"]
review_cols = [c for c in review_cols if c in spotcheck_df.columns]
spotcheck_df[review_cols].to_csv(SPOTCHECK, index=False)
print(f"spotcheck_200.csv saved: {len(spotcheck_df)} rows")

# =============================================================================
# FINAL SAVE
# =============================================================================
print(f"\n{'='*60}")
print("FINAL SAVE")
print(f"{'='*60}")

df_b["entity_count"] = df_b["ents_norm"].apply(len)
drop_cols = ["ents_list","labels_list","ents_norm","labels_updated",
             "has_entity","_stratum"]
df_final = df_b.drop(columns=[c for c in drop_cols if c in df_b.columns])

df_final.to_csv(OUTPUT_CSV, index=False)

print("\n=== Pipeline Complete ===")
print(f"Input rows              : {N_ORIG:,}")
print(f"After Stage A (template): {len(df_a):,}  (-{N_ORIG - len(df_a):,})")
print(f"After Stage B (near-dup): {len(df_b):,}  (-{len(df_a) - len(df_b):,})")
print(f"Final rows              : {len(df_final):,}")
print(f"\nSaved:")
print(f"  {OUTPUT_CSV}")
print(f"  {SPOTCHECK}")
print(f"  {NORM_MAP_CSV}")

label_counts = Counter(
    lbl
    for row in df_final["entity_labels"]
    for lbl in (row.split(" | ") if row else [])
    if lbl.strip()
)
print(f"\nFinal entity label distribution:")
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {lbl:12s} {cnt:>10,}")
print(f"\nTotal entity mentions: {sum(label_counts.values()):,}")
print(f"UMLS linking: {'enabled' if UMLS_AVAILABLE else 'skipped'}")
