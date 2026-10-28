# sft/ — CPU-stage scaffold

Base model: `Qwen/Qwen3-1.7B` (Apache-2.0)
Revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

## Recreate the environment

```bash
# From repo root (Windows PowerShell)
python -m venv sft/.venv
sft/.venv/Scripts/pip install --no-cache-dir -r sft/requirements-cpu.txt
```

## Run the smoke tests

```bash
sft/.venv/Scripts/python -m unittest discover -v -s sft/tests
```

Expected: all tests pass; tokenizer test requires `sft/tokenizer_snapshot/`.

## Run the manifest verifier

```bash
sft/.venv/Scripts/python sft/verify_manifest.py
```

Exit 0 = all protected files match. Exit 1 = something changed or is missing.

## Rebuild the protected manifest (after intentional changes only)

```bash
sft/.venv/Scripts/python sft/build_manifest.py
```

## Tokenizer snapshot (already downloaded)

Stored in `sft/tokenizer_snapshot/` (git-ignored). No weights — config and
tokenizer files only. Re-download at the pinned revision:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    allow_patterns=["tokenizer*","vocab.json","merges.txt","config.json",
                    "generation_config.json","LICENSE"],
    ignore_patterns=["*.safetensors","*.bin","*.gguf"],
    local_dir="sft/tokenizer_snapshot")
```
