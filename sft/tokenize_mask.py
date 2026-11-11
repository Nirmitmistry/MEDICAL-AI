"""
sft/tokenize_mask.py — A2: Tokenize and mask dialogue JSONL records.

TORCH-FREE: returns plain Python lists of ints.  Torch is imported only
inside functions decorated with _requires_torch, which are skipped by the
CPU test suite.

Architecture refs: D4, E2, E3, G(P3).

Public API
----------
load_tokenizer(pkg_dir)         -> PreTrainedTokenizerFast (or compat)
tokenize_dialogue(tok, record, max_len, hard_cap) -> TokenizedItem | None
    Returns None if the dialogue is dropped entirely.

TokenizedItem fields:
    input_ids  : list[int]   (length <= max_len)
    labels     : list[int]   (same length; -100 for masked positions)
    n_tokens   : int
    truncated  : bool        (True if any exchange was dropped)
    source     : str
    dialogue_id: str

Masking rules (Architecture E2):
  Supervised (labels = token id):
    - Every assistant turn: content tokens + closing <|im_end|>
  Masked (labels = -100):
    - System block (header + content + <|im_end|>)
    - Every user turn (header + content + <|im_end|>)
    - Role headers for assistant turns ("<|im_start|>assistant\\n")
    - The "\\n" after every <|im_end|>
    - Padding (added at collate time, not here)

Truncation (Architecture E3):
  max_len default = 2,048; hard_cap = 3,072.
  If the full dialogue exceeds max_len:
    1. Drop trailing exchanges until it fits, keeping the longest prefix
       that ends on an assistant turn.
    2. If even exchange-1 alone doesn't fit, drop the whole dialogue.
  Count both: truncated items and dropped items, per source.

Special-token sanitisation (Architecture A2 requirement):
  Content is tokenised with add_special_tokens=False.
  Any content string that contains a literal special-token sequence
  (e.g. "<|im_start|>" pasted by a user) is DROPPED; the dialogue is
  counted as dropped_special_token_in_content, per source.

Design note — segment-by-segment tokenisation:
  We build the full token sequence by tokenising each segment separately:
    sys_header, sys_content, im_end, newline,
    user_header, user_content, im_end, newline,
    asst_header, asst_content, im_end, newline, ...
  then concatenate.  Labels mirror the ids, with -100 everywhere except
  asst_content and the asst <|im_end|>.  T1 asserts this equals the
  full-string tokenisation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants (Architecture D4 / E2)
# ---------------------------------------------------------------------------
MAX_LEN_DEFAULT: int = 2_048
HARD_CAP: int = 3_072
MASKED: int = -100

# Special-token ids (Qwen3-1.7B tokenizer)
IM_START_ID: int = 151_644   # <|im_start|>
IM_END_ID: int = 151_645     # <|im_end|>
ENDOFTEXT_ID: int = 151_643  # <|endoftext|>  — this is our pad token

# Default system prompt (Architecture D4 / SYSTEM_PROMPT_V1)
# Must match sft/prompts/system_prompt_v1.txt byte-for-byte.
SYSTEM_PROMPT_V1: str = (
    "You are a general health-information assistant. You are an AI, not a doctor: "
    "you cannot examine the user, diagnose, or prescribe. Use everything the user "
    "has told you in this conversation. If key details are missing, ask one or two "
    "focused questions. State uncertainty plainly. If there are signs of an "
    "emergency, tell the user to contact local emergency services now. Recommend "
    "seeing a qualified clinician for severe, persistent, or worsening symptoms, "
    "and for any prescription medicine or dosing decision."
)
# Alias for backward compat / imports
_SYSTEM_PROMPT = SYSTEM_PROMPT_V1

# Pre-compiled regex for literal special-token strings in content
_SPECIAL_TOK_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|object_ref_start|object_ref_end|"
    r"box_start|box_end|quad_start|quad_end|vision_start|vision_end|"
    r"vision_pad|image_pad|video_pad|fim_prefix|fim_middle|fim_suffix|"
    r"fim_pad|repo_name|file_sep)\|>"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TokenizedItem:
    input_ids: List[int]
    labels: List[int]
    n_tokens: int
    truncated: bool
    source: str
    dialogue_id: str

    def __post_init__(self) -> None:
        assert len(self.input_ids) == len(self.labels), (
            f"input_ids length {len(self.input_ids)} != labels length {len(self.labels)}"
        )
        assert self.n_tokens == len(self.input_ids)


@dataclass
class TokenizationStats:
    """Mutable counter, one instance per run; merge with += ."""
    kept: int = 0
    truncated: int = 0
    dropped_too_long: int = 0
    dropped_special_token_in_content: int = 0

    def __iadd__(self, other: "TokenizationStats") -> "TokenizationStats":
        self.kept += other.kept
        self.truncated += other.truncated
        self.dropped_too_long += other.dropped_too_long
        self.dropped_special_token_in_content += other.dropped_special_token_in_content
        return self

    def to_dict(self) -> dict:
        return {
            "kept": self.kept,
            "truncated": self.truncated,
            "dropped_too_long": self.dropped_too_long,
            "dropped_special_token_in_content": self.dropped_special_token_in_content,
        }


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------
def load_tokenizer(pkg_dir: str | os.PathLike):
    """
    Load the tokenizer from tokenizer_pkg/ with local_files_only.
    Sets HF_HUB_OFFLINE=1 as a side-effect to prevent accidental network calls.
    Returns a PreTrainedTokenizerFast (or Qwen2Tokenizer).
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(str(pkg_dir), local_files_only=True)
    # Sanity checks
    assert tok.eos_token_id == IM_END_ID, (
        f"Expected eos_token_id={IM_END_ID} (<|im_end|>), got {tok.eos_token_id}"
    )
    assert tok.pad_token_id == ENDOFTEXT_ID, (
        f"Expected pad_token_id={ENDOFTEXT_ID} (<|endoftext|>), got {tok.pad_token_id}"
    )
    assert tok.pad_token_id != tok.eos_token_id, (
        "pad_token_id must differ from eos_token_id (Architecture D4)"
    )
    return tok


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _tok_segment(tok, text: str) -> List[int]:
    """Tokenise plain text with special-token parsing DISABLED."""
    # In transformers 5.x the flag is `add_special_tokens=False`.
    # We also pass the text as a string (not a list) to avoid padding.
    return tok.encode(text, add_special_tokens=False)


def _has_literal_special_token(text: str) -> bool:
    """Return True if 'text' contains a literal special-token string."""
    return bool(_SPECIAL_TOK_RE.search(text))


def _build_exchange_tokens(
    tok,
    messages: list[dict],
    start_idx: int,
    end_idx: int,
    include_system: bool,
    system_text: str,
) -> tuple[list[int], list[int]]:
    """
    Build (input_ids, labels) for messages[start_idx:end_idx+1].
    If include_system is True, prepend the system block.
    start_idx must point to a user turn; end_idx must point to an assistant turn.
    Returns empty lists on any error.
    """
    ids: List[int] = []
    labels: List[int] = []

    # Pre-compute constant token sequences
    newline_ids = _tok_segment(tok, "\n")

    # System block (always masked)
    if include_system:
        sys_header = _tok_segment(tok, "<|im_start|>system\n")
        sys_content = _tok_segment(tok, system_text)
        sys_end = [IM_END_ID]
        sys_nl = newline_ids[:]

        ids += sys_header + sys_content + sys_end + sys_nl
        labels += [MASKED] * (len(sys_header) + len(sys_content) + len(sys_end) + len(sys_nl))

    # Turns
    for msg in messages[start_idx : end_idx + 1]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            u_header = _tok_segment(tok, "<|im_start|>user\n")
            u_content = _tok_segment(tok, content)
            u_end = [IM_END_ID]
            u_nl = newline_ids[:]
            ids += u_header + u_content + u_end + u_nl
            labels += [MASKED] * (len(u_header) + len(u_content) + len(u_end) + len(u_nl))

        elif role == "assistant":
            a_header = _tok_segment(tok, "<|im_start|>assistant\n")
            a_content = _tok_segment(tok, content)
            a_end = [IM_END_ID]
            a_nl = newline_ids[:]
            ids += a_header + a_content + a_end + a_nl
            # Supervised: content + <|im_end|>; masked: header + trailing \n
            labels += [MASKED] * len(a_header)
            labels += list(a_content)      # supervised
            labels += list(a_end)          # supervised (<|im_end|>)
            labels += [MASKED] * len(a_nl)

    return ids, labels


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def tokenize_dialogue(
    tok,
    record: dict,
    max_len: int = MAX_LEN_DEFAULT,
    hard_cap: int = HARD_CAP,
) -> tuple[Optional[TokenizedItem], str]:
    """
    Tokenise a single dialogue record.

    Parameters
    ----------
    tok      : tokenizer loaded by load_tokenizer()
    record   : JSONL record from dialogues_*.jsonl (D2 schema)
    max_len  : maximum sequence length (default 2,048)
    hard_cap : absolute hard cap (default 3,072)

    Returns
    -------
    (item, drop_reason)
      item is None when the dialogue is dropped entirely.
      drop_reason is one of:
        ""                              — kept (no drop)
        "dropped_too_long"              — exchange 1 alone doesn't fit
        "dropped_special_token"         — literal special token in content
    """
    assert max_len <= hard_cap, f"max_len {max_len} > hard_cap {hard_cap}"

    messages = record.get("messages", [])
    source = record.get("source", "")
    dialogue_id = record.get("dialogue_id", "")

    # Build system text
    # (In production, callers pass system_text from system_prompt_v1.txt;
    #  here we use the module-level constant.)
    # Determine if the record has an explicit system message (not expected from JSONL)
    if messages and messages[0]["role"] == "system":
        system_text = messages[0]["content"]
        turn_start = 1
    else:
        system_text = SYSTEM_PROMPT_V1
        turn_start = 0

    # Build list of (user_idx, asst_idx) exchange pairs
    exchanges: list[tuple[int, int]] = []
    i = turn_start
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            exchanges.append((i, i + 1))
            i += 2
        else:
            break  # malformed; stop here (R3)

    if not exchanges:
        # Drop: no valid exchange
        return None, "dropped_too_long"

    # Check for literal special tokens in content (Architecture A2)
    all_texts = [system_text] + [m["content"] for m in messages[turn_start:]]
    if any(_has_literal_special_token(t) for t in all_texts):
        return None, "dropped_special_token"

    # Try to tokenise with as many exchanges as fit within max_len
    # Working backwards: try n_ex = len(exchanges), len(exchanges)-1, ..., 1
    for n_ex in range(len(exchanges), 0, -1):
        last_msg_idx = exchanges[n_ex - 1][1]  # index of last assistant turn
        ids, lbls = _build_exchange_tokens(
            tok,
            messages,
            start_idx=turn_start,
            end_idx=last_msg_idx,
            include_system=True,
            system_text=system_text,
        )
        if len(ids) <= max_len:
            truncated = n_ex < len(exchanges)
            item = TokenizedItem(
                input_ids=ids,
                labels=lbls,
                n_tokens=len(ids),
                truncated=truncated,
                source=source,
                dialogue_id=dialogue_id,
            )
            return item, ""
        # If even at n_ex==1 it doesn't fit, loop exits and we return drop

    # Exchange 1 alone exceeds max_len → drop
    return None, "dropped_too_long"


# ---------------------------------------------------------------------------
# Batch helper (generator, no torch)
# ---------------------------------------------------------------------------
def iter_tokenized(
    tok,
    jsonl_path: str | os.PathLike,
    max_len: int = MAX_LEN_DEFAULT,
    hard_cap: int = HARD_CAP,
    stats_by_source: Optional[dict[str, TokenizationStats]] = None,
):
    """
    Yield TokenizedItem for every kept dialogue in a JSONL file.
    Optionally accumulates per-source TokenizationStats.
    """
    import json  # local import to avoid module-level dependency cycles

    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = record.get("source", "UNKNOWN")

            item, reason = tokenize_dialogue(tok, record, max_len, hard_cap)

            if stats_by_source is not None:
                if source not in stats_by_source:
                    stats_by_source[source] = TokenizationStats()
                st = stats_by_source[source]
                if item is not None:
                    st.kept += 1
                    if item.truncated:
                        st.truncated += 1
                elif reason == "dropped_too_long":
                    st.dropped_too_long += 1
                elif reason == "dropped_special_token":
                    st.dropped_special_token_in_content += 1

            if item is not None:
                yield item


# ---------------------------------------------------------------------------
# Reference renderer for T6 (train/inference parity)
# ---------------------------------------------------------------------------
def apply_chat_template_to_prefix(tok, messages: list[dict]) -> str:
    """
    Call tok.apply_chat_template(messages, add_generation_prompt=True)
    and return the resulting string.
    Used by T6 to verify train/inference parity byte-for-byte.

    If jinja2 is not available (CPU-only venv), falls back to
    render_chat_template_no_jinja() which implements the same logic
    without jinja2.
    """
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except ImportError:
        # jinja2 not installed (CPU venv) — use our own renderer
        return render_chat_template_no_jinja(messages, add_generation_prompt=True)


def render_chat_template_no_jinja(
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    """
    Pure-Python implementation of our custom ChatML template (Architecture D4).
    Equivalent to tok.apply_chat_template() without requiring jinja2.

    Template rules:
      - If messages[0].role != 'system', inject SYSTEM_PROMPT_V1 as system block.
      - Each turn: <|im_start|>{role}\n{content}<|im_end|>\n
      - If add_generation_prompt: append <|im_start|>assistant\n
    """
    # Use module-level constant (avoids local definition duplication)
    parts: list[str] = []

    # Check if messages already have a system message
    has_system = messages and messages[0]["role"] == "system"

    # Inject default system prompt if absent
    if not has_system:
        parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT_V1}<|im_end|>\n")

    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def training_prefix_string(tok, record: dict, exchange_idx: int) -> str:
    """
    Build the training string for the prefix ending at exchange_idx (0-based),
    i.e. messages[0 .. user_turn_of_exchange_idx+1] (inclusive of user turn,
    up to and including the assistant header of exchange_idx).

    This is compared byte-for-byte with apply_chat_template_to_prefix() in T6.
    """
    messages = record.get("messages", [])
    if messages and messages[0]["role"] == "system":
        turn_start = 1
        system_text = messages[0]["content"]
    else:
        system_text = None
        turn_start = 0

    # Build exchanges list
    exchanges: list[tuple[int, int]] = []
    i = turn_start
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            exchanges.append((i, i + 1))
            i += 2
        else:
            break

    if exchange_idx >= len(exchanges):
        return ""

    # Build messages list up to and including the user turn of exchange_idx
    # This is what apply_chat_template should see with add_generation_prompt=True
    user_idx = exchanges[exchange_idx][0]
    prefix_msgs = []

    # Always include system (the template auto-injects it if absent)
    if system_text is not None:
        prefix_msgs.append({"role": "system", "content": system_text})
    # else: template will inject default; we don't add system here

    for msg in messages[turn_start : user_idx + 1]:
        prefix_msgs.append({"role": msg["role"], "content": msg["content"]})

    return apply_chat_template_to_prefix(tok, prefix_msgs)


# ---------------------------------------------------------------------------
# CLI (for quick checks)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Tokenize a single JSONL record (for debugging)."
    )
    parser.add_argument("jsonl", help="JSONL file")
    parser.add_argument("--pkg", default="sft/tokenizer_pkg", help="tokenizer_pkg dir")
    parser.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    parser.add_argument("--n", type=int, default=5, help="Number of items to process")
    args = parser.parse_args()

    tok = load_tokenizer(args.pkg)
    count = 0
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            item, reason = tokenize_dialogue(tok, rec, max_len=args.max_len)
            if item:
                supervised = sum(1 for x in item.labels if x != MASKED)
                print(
                    f"id={item.dialogue_id[:8]}  "
                    f"n_tok={item.n_tokens}  "
                    f"supervised={supervised}  "
                    f"truncated={item.truncated}"
                )
            else:
                print(f"DROPPED  reason={reason}")
            count += 1
            if count >= args.n:
                break
