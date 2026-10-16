import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEAKY_OR_UNFAIR_COLUMNS = {
    "package_key",
    "package_name",
    "version",
    "label",
    # These are only available for malicious samples in the current feature table.
    "malware_tgz_size",
    "snippet_count",
    "behavior_label_count",
    "evasion_label_count",
    "has_behavior_label",
    "has_evasion_label",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/base_feature_table_benign1000.csv")
    parser.add_argument("--out-dir", default="runs/baselines/benign1000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-per-class", type=int, default=982)
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--permutation-repeats", type=int, default=10)
    return parser.parse_args()


def build_fair_subset(df, random_state, max_per_class):
    fair = df[df["has_package_json"].eq(1)].copy()
    fair["label"] = fair["label"].astype(int)
    parts = []
    for label, group in fair.groupby("label"):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=random_state))
    fair = pd.concat(parts, ignore_index=True)
    fair = fair.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return fair


def feature_columns(df):
    cols = []
    for col in df.columns:
        if col in LEAKY_OR_UNFAIR_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_data(df, random_state, test_size, val_size):
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label"],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def models(random_state):
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=random_state,
        ),
        "mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def predict_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(x)


def evaluate(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def top_feature_importance(name, model, x_val, y_val, columns, out_dir, repeats, random_state):
    result = permutation_importance(
        model,
        x_val,
        y_val,
        scoring="f1",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(out_dir / f"{name}_permutation_importance.csv", index=False)
    return imp.head(20).to_dict(orient="records")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    if args.no_resample:
        fair = df[df["has_package_json"].eq(1)].copy()
        fair["label"] = fair["label"].astype(int)
        fair = fair.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    else:
        fair = build_fair_subset(df, args.random_state, args.max_per_class)
    columns = feature_columns(fair)
    fair.to_csv(out_dir / "fair_subset.csv", index=False)

    train, val, test = split_data(fair, args.random_state, args.test_size, args.val_size)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)

    x_train = train[columns].fillna(0)
    y_train = train["label"].astype(int)
    x_val = val[columns].fillna(0)
    y_val = val["label"].astype(int)
    x_test = test[columns].fillna(0)
    y_test = test["label"].astype(int)

    all_metrics = []
    all_predictions = []
    importance_summary = {}

    for name, model in models(args.random_state).items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        val_score = predict_scores(model, x_val)
        test_score = predict_scores(model, x_test)
        val_metrics = evaluate(y_val, val_score)
        test_metrics = evaluate(y_test, test_score)

        row = {"model": name, "split": "val", **val_metrics}
        all_metrics.append(row)
        row = {"model": name, "split": "test", **test_metrics}
        all_metrics.append(row)

        pred_df = test[["package_key", "package_name", "version", "label"]].copy()
        pred_df["model"] = name
        pred_df["score"] = test_score
        pred_df["prediction"] = (test_score >= 0.5).astype(int)
        pred_df["error_type"] = np.where(
            (pred_df["label"].eq(0) & pred_df["prediction"].eq(1)),
            "false_positive",
            np.where(
                (pred_df["label"].eq(1) & pred_df["prediction"].eq(0)),
                "false_negative",
                "correct",
            ),
        )
        all_predictions.append(pred_df)

        importance_summary[name] = top_feature_importance(
            name,
            model,
            x_val,
            y_val,
            columns,
            out_dir,
            args.permutation_repeats,
            args.random_state,
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "feature_source": args.features,
        "fair_subset_rows": int(len(fair)),
        "class_counts": {str(k): int(v) for k, v in fair["label"].value_counts().sort_index().items()},
        "feature_count": int(len(columns)),
        "features": columns,
        "metrics": metrics_df.to_dict(orient="records"),
        "top_permutation_importance": importance_summary,
        "outputs": {
            "fair_subset": str(out_dir / "fair_subset.csv"),
            "metrics_csv": str(out_dir / "metrics.csv"),
            "predictions_csv": str(out_dir / "predictions.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

