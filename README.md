# TGC-MPD

Tree-Guided Confidence-Preserved Correction for hard-case malicious npm package detection.

This repository contains the implementation used for the paper:

> TGC-MPD: Tree-Guided Confidence-Preserved Correction for Hard-Case Malicious npm Package Detection

TGC-MPD targets suspicious hard cases where benign and malicious npm packages share overlapping static risk signals. The method keeps high-confidence predictions from a strong HGB teacher and applies a neural TreeCross correction branch only when the teacher is uncertain.

## Repository Contents

- `scripts/build_feature_table.py`: extracts static package-level risk features.
- `scripts/build_eval_settings.py`: builds scoped-balanced and hard-case evaluation settings.
- `scripts/train_baselines.py`: trains LR, RF, HGB, and MLP baselines.
- `scripts/train_treecross.py`: trains TreeCross/TGC-MPD variants.
- `scripts/run_baselines_multiseed.py`: runs baseline experiments over multiple random seeds.
- `scripts/run_treecross_multiseed.py`: runs TGC-MPD experiments over multiple random seeds.
- `scripts/evaluate_tool_baselines.py`: evaluates tool-output baselines when NPMStudy tool outputs are available.
- `scripts/build_paper_assets.py`: builds tables and case-study artifacts for the paper.
- `scripts/make_paper_figures.py`: regenerates paper figures.
- `cscloud2026_latex/`: LaTeX source for the CSCloud 2026 submission draft.

## Data Notice

The original NPMStudy archives, downloaded npm package archives, malicious package artifacts, and generated experiment outputs are not included in this repository. Users should obtain datasets from their original sources and follow the corresponding redistribution and safety policies.

Expected local data paths include:

```text
data/raw/
data/processed/
runs/
paper_assets/
```

These directories are intentionally excluded from version control.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The experiments use Python with pandas, NumPy, scikit-learn, and PyTorch. GPU execution is optional but recommended for TreeCross experiments.

## Typical Workflow

Build feature tables:

```bash
python scripts/build_feature_table.py \
  --benign-manifest data/processed/npmstudy_index/benign_registry_random2000_manifest.csv \
  --out-csv data/processed/features/base_feature_table_random2000.csv
```

Build evaluation settings:

```bash
python scripts/build_eval_settings.py \
  --features data/processed/features/base_feature_table_random2000.csv \
  --out-dir data/processed/eval_settings/random2000 \
  --n-per-class 1800
```

Run baselines:

```bash
python scripts/run_baselines_multiseed.py \
  --seeds 40 41 42 \
  --features data/processed/eval_settings/random2000/scoped_balanced.csv data/processed/eval_settings/random2000/hard_case.csv \
  --out-dir runs/baselines/random2000_multiseed \
  --permutation-repeats 3
```

Run TGC-MPD:

```bash
python scripts/run_treecross_multiseed.py \
  --seeds 40 41 42 \
  --features data/processed/eval_settings/random2000/scoped_balanced.csv data/processed/eval_settings/random2000/hard_case.csv \
  --out-dir runs/treecross/random2000_v3_multiseed \
  --epochs 160 \
  --patience 24 \
  --batch-size 128 \
  --rf-trees 400 \
  --blend-step 0.05 \
  --model-family TreeCross-MPD-v3 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable.

## Safety

This project is intended for defensive research on malicious package detection. Do not upload or redistribute active malicious package payloads, downloaded package archives, private dataset copies, or credentials.

