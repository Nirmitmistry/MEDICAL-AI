# CAPSTONE amendments

## A1 — Notebook deliverables (2026-11-08, requested by project owner)

**Decision:** All new deliverables are Jupyter notebooks (`.ipynb`) under `sft/notebooks/`.

**Details:**

- Local notebook execution uses `sft/.venv-nb`; the pinned CPU environment `sft/.venv` and `sft/freeze.txt` are unchanged.
- Kaggle already provides Jupyter, so no Kaggle dependency change is required.
- Scope covers new work from Stage 1 onward.
- Existing P0a `.py` files and tests stay as they are; no `.py` modules are renamed or removed.
- Nothing in ARCHITECTURE decisions H1–H14 changes except H13's "environment pinned", which now also covers the notebook venv via `sft/freeze-notebook.txt`.
- The notebook venv is tracked by `sft/requirements-notebook.txt` (direct pins) and `sft/freeze-notebook.txt` (full freeze).
