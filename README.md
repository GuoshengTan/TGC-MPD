# TGC-MPD

Tree-Guided Confidence-Preserved Correction for Hard-Case Malicious npm Package Detection.

This repository contains the paper PDF, research code, and experiment parameter records. Datasets are not distributed.

TGC-MPD combines an HGB teacher with a neural correction branch. Its confidence gate preserves teacher **scores** outside the uncertain region; the validation-selected classification threshold can still change labels. The current implementation evaluates both branches before selecting scores, and is not a conditional-compute implementation.

## Paper

**TGC-MPD: Tree-Guided Confidence-Preserved Correction for Hard-Case Malicious npm Package Detection**

Guosheng Tan, Yuhao Feng, Yuxuan Zhou, Jiahui Li, Yiming Geng, and Ling Lu.

Accepted at the 2026 IEEE 13th International Conference on Cyber Security and Cloud Computing (CSCloud).

[Read the paper (PDF)](paper/TGC-MPD.pdf) | [Download PDF](https://github.com/GuoshengTan/TGC-MPD/raw/refs/heads/main/paper/TGC-MPD.pdf)

The seven-page PDF is the PDF eXpress-validated submission copy, not an IEEE Xplore version of record. The original downloaded filename was `2026380451.pdf`; it is published here without modifying its contents.

### Overview

The TreeCross branch combines static package features, teacher risk priors, and RF leaf embeddings. A validation-selected confidence gate chooses between the HGB teacher score and the correction score.

Across three seeds under the package-version protocol, mean hard-case F1 increases from 0.9458 to 0.9535 and recall from 0.9310 to 0.9487 relative to fixed-threshold HGB. The hard-case F1 gain is not observed under package-name-grouped testing, which evaluates transfer to unseen package names.

## Contents


- `paper/TGC-MPD.pdf`: the seven-page paper submission copy.
- `scripts/`: all 16 project scripts for data indexing, feature construction, baseline and neural training, multi-seed evaluation, tool baselines, analysis, benchmarking, and figure generation.
- `tests/`: package-name grouped split regression tests.
- `configs/run_arguments.json`: arguments recorded by 18 historical runs, including ConfuCross, TreeCross v1/v2/v3, multi-seed, grouped, and smoke runs.
- `configs/validation_selected_parameters.json`: feature lists and validation-selected thresholds, epochs, blend coefficients, and confidence margins from those runs.
- `configs/README.md`: reproduction details and limitations.

## Setup

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Install a PyTorch build appropriate for your hardware. The dependency ranges are not an exact historical environment lock.

## Data Preparation

Obtain NPMStudy data from its original source and reconstruct benign package archives under the paths expected by the scripts. Inspect each data script's `--help` before use. Downloaded package versions may become unavailable. No raw archives or feature tables are distributed here.

```bash
python scripts/build_feature_table.py --benign-manifest data/processed/npmstudy_index/benign_registry_random2000_manifest.csv --out-csv data/processed/features/base_feature_table_random2000.csv
python scripts/build_eval_settings.py --features data/processed/features/base_feature_table_random2000.csv --out-dir data/processed/eval_settings/random2000 --n-per-class 1800
```

## Main Experiments

The manuscript's main multi-seed configuration corresponds to `random2000_v2_multiseed`, not the later v3 experiment directory. Historical labels are run labels, not switches selecting different source-code versions.

```bash
python scripts/run_baselines_multiseed.py --seeds 40 41 42 --out-dir runs/baselines/random2000_multiseed --permutation-repeats 3
python scripts/run_treecross_multiseed.py --seeds 40 41 42 --split-mode package_version --out-dir runs/treecross/random2000_v2_multiseed --epochs 160 --patience 24 --batch-size 128 --rf-trees 400 --blend-step 0.05 --model-family "TreeCross-MPD v2" --device cuda
python scripts/run_treecross_multiseed.py --seeds 40 41 42 --split-mode package_name --out-dir runs/treecross/package_name_grouped_20261012 --epochs 160 --patience 24 --batch-size 128 --rf-trees 400 --blend-step 0.05 --model-family TGC-MPD-package-name-grouped --device cuda
```

Both runners default to the scoped-balanced and hard-case feature tables generated above. Use `--device cpu` for neural training without CUDA. Keep the two split protocols separate when interpreting results.

## Auxiliary Scripts

`benchmark_treecross_cost.py` uses a frozen v3 seed-42 hard-case split. Run its `prepare` mode before `hgb_cpu`, `full_cpu`, or `full_cuda`. It requires local training outputs; its output directory is currently hard-coded under `cscloud2026_latex/revision_20261012/performance`.

Analysis and figure scripts also expect locally generated experiment outputs, and some use historical hard-coded paths. They are included as research code, not standalone tools that can run without those inputs.

## Safety and Scope

Use isolated environments when handling malicious-package archives; do not install or execute their payloads. This upload excludes datasets, package archives, trained binary checkpoints, LaTeX sources, administrative submission records, private personal materials, and credentials. The paper PDF is the sole explicitly included manuscript artifact.
