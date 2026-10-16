import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "data/processed/eval_settings/random2000/scoped_balanced.csv",
            "data/processed/eval_settings/random2000/hard_case.csv",
        ],
    )
    parser.add_argument("--out-dir", default="runs/baselines/random2000_multiseed")
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def setting_name(feature_path):
    return Path(feature_path).stem


def run_one(seed, feature_path, args, out_dir):
    setting = setting_name(feature_path)
    run_dir = out_dir / f"seed_{seed}" / setting
    metrics_path = run_dir / "metrics.csv"
    if args.skip_existing and metrics_path.exists():
        print(f"[seed={seed}][{setting}] existing metrics found, skipping.")
        return

    cmd = [
        sys.executable,
        "scripts/train_baselines.py",
        "--features",
        feature_path,
        "--out-dir",
        str(run_dir),
        "--random-state",
        str(seed),
        "--no-resample",
        "--permutation-repeats",
        str(args.permutation_repeats),
    ]
    print(f"[seed={seed}][{setting}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def aggregate(args, out_dir):
    frames = []
    for seed in args.seeds:
        for feature_path in args.features:
            setting = setting_name(feature_path)
            metrics_path = out_dir / f"seed_{seed}" / setting / "metrics.csv"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            df = pd.read_csv(metrics_path)
            df.insert(0, "setting", setting)
            df.insert(0, "seed", seed)
            frames.append(df)

    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(out_dir / "all_metrics.csv", index=False)

    test_metrics = all_metrics[all_metrics["split"].eq("test")].copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"]
    mean_df = (
        test_metrics.groupby(["setting", "model"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"{col}_mean" for col in metric_cols})
    )
    std_df = (
        test_metrics.groupby(["setting", "model"], as_index=False)[metric_cols]
        .std()
        .rename(columns={col: f"{col}_std" for col in metric_cols})
    )
    summary = mean_df.merge(std_df, on=["setting", "model"], how="left")
    summary["seed_count"] = len(args.seeds)
    summary = summary.sort_values(["setting", "f1_mean"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)

    payload = {
        "seeds": args.seeds,
        "features": args.features,
        "out_dir": str(out_dir),
        "all_metrics": str(out_dir / "all_metrics.csv"),
        "summary": str(out_dir / "summary.csv"),
        "best_by_setting": summary.groupby("setting").head(4).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["best_by_setting"], indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for feature_path in args.features:
            run_one(seed, feature_path, args, out_dir)
    aggregate(args, out_dir)


if __name__ == "__main__":
    main()

