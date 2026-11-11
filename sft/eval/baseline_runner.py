"""
sft/eval/baseline_runner.py — baseline evaluation runner (C2)

Architecture ref: ARCHITECTURE.md E6 (V1–V6), G(P2), IMPLEMENTATION_PLAN Stage 2.

IMPORTANT: This script requires torch, transformers with GPU, and the full model
weights. It is designed to run on Kaggle (GPU). Do NOT run in sft/.venv (CPU only).

The script is device-agnostic: it auto-detects CUDA. On CPU it will run but
very slowly (use only for smoke-testing with --max-probes 2).

Usage (Kaggle)
--------------
python sft/eval/baseline_runner.py \\
    --model-dir  /kaggle/input/qwen3-weights/Qwen3-1.7B \\
    --val-jsonl  /kaggle/input/sft-splits/dialogues_val.jsonl \\
    --probes-dev sft/work/probes_dev.jsonl \\
    --out-dir    /kaggle/working/p2_baseline \\
    [--max-probes N]  # limit for smoke-testing

Outputs (written to --out-dir)
-------------------------------
baseline_results.json          aggregates: NLL per source, V2 gap, V6 stats, dev-probe pass rates
baseline_generations_dev.jsonl {id, response, format_a, format_b} per dev probe
v5_rater_sheet.csv             V5 probe ids + generated response text for human rating

The test split is never loaded or touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# Guard: torch and transformers required
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(
        f'ERROR: {e}\n'
        'This script requires torch and transformers with GPU support.\n'
        'Run it on Kaggle, not in sft/.venv (CPU only).',
        file=sys.stderr,
    )
    sys.exit(2)

# ── constants ──────────────────────────────────────────────────────────────────
PINNED_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MAX_LEN         = 2048
MAX_NEW_TOKENS  = 512
PROMPT_BUDGET   = MAX_LEN - MAX_NEW_TOKENS   # 1536 tokens for prompt
DECODING_PARAMS = {
    'temperature': 0.0,   # greedy
    'do_sample':   False,
    'max_new_tokens': MAX_NEW_TOKENS,
    'repetition_penalty': 1.05,
}


# ── ChatML renderer (C0, same as in P2c notebook) ─────────────────────────────

def load_system_prompt(repo_root: Path) -> str:
    p = repo_root / 'sft' / 'prompts' / 'system_prompt_v1.txt'
    if not p.exists():
        p = repo_root / 'sft' / 'config' / 'system_prompt_v1.txt'
    with open(p, encoding='utf-8') as f:
        return f.read().strip()


def render_chatml(messages: list[dict], system: str) -> str:
    """Render a full ChatML prompt string (no generation prompt appended)."""
    parts = [f'<|im_start|>system\n{system}<|im_end|>\n']
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    return ''.join(parts)


def render_chatml_gen(messages: list[dict], system: str) -> str:
    """Render ChatML with generation prompt (ends with assistant header)."""
    return render_chatml(messages, system) + '<|im_start|>assistant\n'


# ── truncation helper (E3) ────────────────────────────────────────────────────

def truncate_to_budget(messages: list[dict], system: str, tokenizer: Any) -> list[dict]:
    """
    Keep the longest suffix of full exchanges that fits in PROMPT_BUDGET tokens.
    System prompt is always kept. Never cuts inside a message.
    Returns the (possibly truncated) messages list.
    """
    # find exchange boundaries (pairs of user+assistant)
    exchanges = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
            exchanges.append((i, i + 1))
            i += 2
        else:
            i += 1

    # try progressively shorter prefixes
    for start_ex in range(len(exchanges)):
        subset = [messages[j] for ex in exchanges[start_ex:] for j in ex]
        prompt = render_chatml_gen(subset, system)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= PROMPT_BUDGET:
            return subset

    # fall back to just the last exchange
    if exchanges:
        last = exchanges[-1]
        return [messages[last[0]], messages[last[1]]]
    return messages


# ── NLL computation ───────────────────────────────────────────────────────────

def compute_nll(
    model: Any,
    tokenizer: Any,
    dialogue: dict,
    system: str,
    device: str,
) -> float | None:
    """
    Compute per-token NLL on assistant tokens only, for one dialogue.
    Returns float or None on error.
    """
    try:
        messages = dialogue['messages']
        # build full sequence with labels
        sys_block = f'<|im_start|>system\n{system}<|im_end|>\n'
        input_ids = []
        labels    = []

        sys_ids = tokenizer.encode(sys_block, add_special_tokens=False)
        input_ids.extend(sys_ids)
        labels.extend([-100] * len(sys_ids))

        for msg in messages:
            header  = f'<|im_start|>{msg["role"]}\n'
            content = msg['content']
            footer  = '<|im_end|>\n'

            h_ids = tokenizer.encode(header,  add_special_tokens=False)
            c_ids = tokenizer.encode(content, add_special_tokens=False)
            f_ids = tokenizer.encode(footer,  add_special_tokens=False)

            input_ids.extend(h_ids)
            labels.extend([-100] * len(h_ids))

            if msg['role'] == 'assistant':
                # supervise content + im_end (not the trailing \n)
                im_end_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)
                nl_id     = tokenizer.encode('\n', add_special_tokens=False)
                input_ids.extend(c_ids + im_end_id + nl_id)
                labels.extend(c_ids + im_end_id + [-100] * len(nl_id))
            else:
                input_ids.extend(c_ids + f_ids)
                labels.extend([-100] * (len(c_ids) + len(f_ids)))

        if len(input_ids) > MAX_LEN:
            return None

        t_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        t_lbl = torch.tensor([labels],    dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids=t_ids, labels=t_lbl)
        return out.loss.item()
    except Exception as e:
        print(f'NLL error on {dialogue.get("dialogue_id", "?")}: {e}', file=sys.stderr)
        return None


# ── V2 history ablation ───────────────────────────────────────────────────────

def compute_v2_gap(
    model: Any,
    tokenizer: Any,
    dialogues: list[dict],
    system: str,
    device: str,
    max_items: int = 200,
) -> dict[str, Any]:
    """
    For multi-exchange HealthChat items: NLL of assistant turn >=2 with
    (a) full history, (b) only first user turn, (c) a different dialogue's context.
    Returns gap statistics.
    """
    items = [d for d in dialogues
             if d.get('source', '').startswith('HealthChat') and d.get('n_exchanges', 0) >= 2]
    items = items[:max_items]

    full_nlls   = []
    trunc_nlls  = []
    swap_nlls   = []

    for idx, dlg in enumerate(items):
        msgs = dlg['messages']
        # take the second exchange (turns 2 and 3)
        if len(msgs) < 4:
            continue

        full_msgs  = msgs  # full history
        trunc_msgs = [msgs[0], msgs[1], msgs[2], msgs[3]]  # only first+second

        # swap context: use previous dialogue's first exchange as history
        swap_src_idx = (idx + 1) % len(items)
        swap_msgs = items[swap_src_idx]['messages'][:2] + [msgs[2], msgs[3]]

        faux_dlg_full  = dict(dlg, messages=full_msgs)
        faux_dlg_trunc = dict(dlg, messages=trunc_msgs)
        faux_dlg_swap  = dict(dlg, messages=swap_msgs)

        # compute NLL only on the last assistant turn
        for faux, store in [(faux_dlg_full, full_nlls),
                            (faux_dlg_trunc, trunc_nlls),
                            (faux_dlg_swap, swap_nlls)]:
            # override messages to just the last exchange for NLL
            target_dlg = dict(faux, messages=[faux['messages'][-2], faux['messages'][-1]])
            # build the prompt prefix (context) then compute NLL over last asst turn
            ctx_msgs = faux['messages'][:-2]
            tgt_msgs = faux['messages'][-2:]

            sys_block = f'<|im_start|>system\n{system}<|im_end|>\n'
            ctx_block = ''.join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in ctx_msgs
            )
            tgt_user  = f"<|im_start|>{tgt_msgs[0]['role']}\n{tgt_msgs[0]['content']}<|im_end|>\n"
            asst_hdr  = f"<|im_start|>{tgt_msgs[1]['role']}\n"
            asst_body = tgt_msgs[1]['content']
            asst_end  = '<|im_end|>\n'

            prefix    = sys_block + ctx_block + tgt_user + asst_hdr
            supervised = asst_body + '<|im_end|>'

            try:
                p_ids = tokenizer.encode(prefix,     add_special_tokens=False)
                s_ids = tokenizer.encode(supervised, add_special_tokens=False)
                nl_id = tokenizer.encode('\n',        add_special_tokens=False)

                if len(p_ids) + len(s_ids) + len(nl_id) > MAX_LEN:
                    continue

                all_ids = p_ids + s_ids + nl_id
                all_lbl = [-100] * len(p_ids) + s_ids + [-100] * len(nl_id)

                t_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
                t_lbl = torch.tensor([all_lbl], dtype=torch.long, device=device)

                with torch.no_grad():
                    out = model(input_ids=t_ids, labels=t_lbl)
                store.append(out.loss.item())
            except Exception as e:
                print(f'V2 error: {e}', file=sys.stderr)
                continue

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    n = len(full_nlls)
    gap_full_trunc = (
        round(_mean(full_nlls) - _mean(trunc_nlls), 4)
        if full_nlls and trunc_nlls else None
    )
    gap_full_swap = (
        round(_mean(full_nlls) - _mean(swap_nlls), 4)
        if full_nlls and swap_nlls else None
    )

    return {
        'n_items': n,
        'mean_nll_full_history': _mean(full_nlls),
        'mean_nll_truncated_history': _mean(trunc_nlls),
        'mean_nll_swapped_history': _mean(swap_nlls),
        'gap_full_minus_truncated': gap_full_trunc,
        'gap_full_minus_swapped': gap_full_swap,
    }


# ── probe generation ──────────────────────────────────────────────────────────

def generate_probe_response(
    model: Any,
    tokenizer: Any,
    probe: dict,
    system: str,
    device: str,
    eos_token_id: int,
) -> dict[str, str]:
    """
    Generate a response for a probe using both prompt formats:
      format_a: our ChatML renderer (C0)
      format_b: Qwen's own template (enable_thinking=False)
    Returns {id, format_a, format_b}.
    """
    turns = probe.get('turns', [])
    # build up to the last user turn
    history = []
    last_user = None
    for t in turns:
        if t['role'] == 'user' and turns.index(t) == len(turns) - 1:
            last_user = t['content']
        elif t['role'] in ('user', 'assistant') and t.get('scripted', True):
            history.append(t)

    if last_user is None and turns:
        # last turn is user
        history = turns[:-1]
        last_user = turns[-1]['content']

    def _gen(prompt_str: str) -> str:
        try:
            ids = tokenizer.encode(prompt_str, return_tensors='pt', add_special_tokens=False)
            if ids.shape[1] > PROMPT_BUDGET:
                ids = ids[:, -PROMPT_BUDGET:]
            ids = ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **DECODING_PARAMS,
                )
            new_ids = out[0][ids.shape[1]:]
            return tokenizer.decode(new_ids, skip_special_tokens=False).strip()
        except Exception as e:
            return f'[ERROR: {e}]'

    # Format A: C0 ChatML renderer
    msgs_a = history + [{'role': 'user', 'content': last_user}]
    prompt_a = render_chatml_gen(msgs_a, system)
    resp_a = _gen(prompt_a)

    # Format B: Qwen tokenizer apply_chat_template
    try:
        chat_b = [{'role': 'system', 'content': system}] + msgs_a
        prompt_b = tokenizer.apply_chat_template(
            chat_b,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_b = _gen(prompt_b)
    except Exception as e:
        resp_b = f'[ERROR: {e}]'

    return {
        'id': probe['id'],
        'format_a': resp_a,
        'format_b': resp_b,
        'response': resp_a,  # default for scorer compatibility
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Baseline runner for P2 evaluation (Kaggle GPU).',
    )
    parser.add_argument('--model-dir',  required=True)
    parser.add_argument('--val-jsonl',  required=True)
    parser.add_argument('--probes-dev', required=True)
    parser.add_argument('--out-dir',    required=True)
    parser.add_argument('--repo-root',  default='.')
    parser.add_argument('--max-probes', type=int, default=None,
                        help='Limit probe count (smoke-test)')
    parser.add_argument('--max-nll-items', type=int, default=400,
                        help='Max dialogues for NLL eval (val_subset_400)')
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_OFFLINE'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cpu':
        print('WARNING: running on CPU — NLL eval will be very slow.', file=sys.stderr)

    # Load system prompt
    system = load_system_prompt(repo_root)
    sys_sha = hashlib.sha256(system.encode('utf-8')).hexdigest()
    print(f'System prompt SHA-256: {sys_sha}')

    # Load model and tokenizer
    print(f'Loading model from {args.model_dir} ...')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
    )
    dtype = torch.bfloat16 if (device == 'cuda' and
                                torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=dtype,
        device_map='auto', trust_remote_code=False,
    )
    model.eval()
    print(f'Loaded in {time.time()-t0:.1f}s  dtype={dtype}')

    im_end_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]
    eos_ids   = [im_end_id]
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # ── V1 NLL per source ─────────────────────────────────────────────────────
    print('Computing V1 NLL ...')
    nll_by_source: dict[str, list[float]] = {}
    val_dialogues: list[dict] = []

    with open(args.val_jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            val_dialogues.append(d)
            if len(val_dialogues) >= args.max_nll_items:
                break

    for dlg in val_dialogues:
        src = dlg.get('source', 'unknown')
        nll = compute_nll(model, tokenizer, dlg, system, device)
        if nll is not None and not math.isnan(nll):
            nll_by_source.setdefault(src, []).append(nll)

    v1_stats = {
        src: {
            'n': len(vs),
            'mean_nll': round(sum(vs) / len(vs), 4) if vs else None,
        }
        for src, vs in nll_by_source.items()
    }
    print('V1 NLL:', v1_stats)

    # ── V2 history ablation ───────────────────────────────────────────────────
    print('Computing V2 history ablation ...')
    v2_stats = compute_v2_gap(model, tokenizer, val_dialogues, system, device)
    print('V2:', v2_stats)

    # ── Probe generation (dev only) ───────────────────────────────────────────
    print('Generating probe responses (dev partition) ...')
    probes_path = Path(args.probes_dev)
    dev_probes: list[dict] = []
    with open(probes_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p.get('partition') == 'dev':
                dev_probes.append(p)

    if args.max_probes:
        dev_probes = dev_probes[:args.max_probes]

    gen_path = out_dir / 'baseline_generations_dev.jsonl'
    v5_rows  = []
    with open(gen_path, 'w', encoding='utf-8', newline='\n') as gf:
        for probe in dev_probes:
            gen = generate_probe_response(model, tokenizer, probe, system, device, im_end_id)
            gf.write(json.dumps(gen, ensure_ascii=False, sort_keys=True) + '\n')
            if probe.get('suite') == 'v5_safety':
                v5_rows.append({
                    'id': probe['id'],
                    'suite': probe.get('suite', ''),
                    'category': probe.get('category', ''),
                    'needs_clinician': probe.get('needs_clinician', False),
                    'response_format_a': gen['format_a'],
                    'response_format_b': gen['format_b'],
                    'pass_fail': '',
                    'notes': '',
                })

    print(f'Generated {len(dev_probes)} dev probe responses.')

    # ── V5 rater sheet ────────────────────────────────────────────────────────
    rater_path = out_dir / 'v5_rater_sheet.csv'
    with open(rater_path, 'w', encoding='utf-8', newline='') as rf:
        writer = csv.DictWriter(rf, fieldnames=[
            'id', 'suite', 'category', 'needs_clinician',
            'response_format_a', 'response_format_b', 'pass_fail', 'notes',
        ])
        writer.writeheader()
        writer.writerows(v5_rows)
    print(f'V5 rater sheet: {rater_path}  ({len(v5_rows)} rows)')

    # ── V6 format stats ───────────────────────────────────────────────────────
    eos_term = 0
    think_count = 0
    total_gen = len(dev_probes)
    with open(gen_path, encoding='utf-8') as gf:
        for line in gf:
            if not line.strip():
                continue
            g = json.loads(line)
            resp = g.get('format_a', '')
            if resp.rstrip().endswith('<|im_end|>'):
                eos_term += 1
            import re as _re
            if _re.search(r'<think>', resp, _re.IGNORECASE):
                think_count += 1

    v6_stats = {
        'total_generations': total_gen,
        'eos_terminated': eos_term,
        'eos_pct': round(eos_term / max(1, total_gen) * 100, 2),
        'think_blocks': think_count,
    }
    print('V6:', v6_stats)

    # ── Write results ─────────────────────────────────────────────────────────
    results = {
        'model': args.model_dir,
        'pinned_revision': PINNED_REVISION,
        'system_prompt_sha256': sys_sha,
        'device': device,
        'val_items': len(val_dialogues),
        'v1_nll_by_source': v1_stats,
        'v2_history_ablation': v2_stats,
        'v6_format_stats': v6_stats,
        'dev_probe_count': len(dev_probes),
        'generations_path': str(gen_path),
        'rater_sheet_path': str(rater_path),
    }
    out_path = out_dir / 'baseline_results.json'
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(results, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'Baseline results written to: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
