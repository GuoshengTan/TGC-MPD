# Experiment Parameters

These JSON files are extracted from existing local run summaries, not reconstructed guesses.

- `run_arguments.json`: each record's `run` identifies its original relative output directory; `args` contains recorded CLI arguments.
- `validation_selected_parameters.json`: corresponding validation-selected model parameters and feature order. Test examples, predictions, and dataset rows are excluded.
- A missing field means it was not recorded. In older runs, missing `split_mode` predates the grouped-split implementation; those runs used package-version splits.
- Smoke runs are retained for completeness, not as publication results.
- Historical source snapshots and exact package locks are not available for every run. Replaying old arguments with current code is not a guarantee of bit-identical results.
- These are hyperparameter and selection records, not trained model weights. Current training code does not persist all fitted models.

## Main TreeCross Settings

Seeds: 40, 41, 42. Train/validation/test fractions: 0.7/0.1/0.2.
RF: 400 trees, minimum leaf size 2, balanced_subsample weights.
HGB: 300 iterations, learning rate 0.05, L2 regularization 0.01.
Neural branch: hidden dimension 128, leaf embedding dimension 8, dropout 0.2.
AdamW: learning rate 0.001, weight decay 0.0001.
Batch size: 128; maximum epochs: 160; early-stopping patience: 24.
Blend grid step: 0.05. Confidence-margin grid: 0 through 0.50 in increments of 0.05.
Thresholds and gate parameters are selected on validation data, not test data.
Use the training source for fixed baseline estimators and default library settings.

## Package-Name Grouping

Grouping uses the complete package name, including scope. StratifiedGroupKFold uses 10 folds, shuffle enabled, and the run seed. Folds 0 and 1 are test, fold 2 is validation, and folds 3 through 9 are training. Grouped split tests check disjoint names, row preservation, reproducibility, and invalid inputs.

## Replaying a Recorded Run

Translate `args` keys to CLI flags by replacing underscores with hyphens and invoke `scripts/train_treecross.py` (or `scripts/train_confucross.py` for ConfuCross records). Expand the features list as separate arguments. Choose a fresh output directory to avoid replacing previous outputs. Paths in historical records may use Windows separators; normalize them for other platforms.
