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
    parser.add_argument("--out-dir", default="runs/treecross/random2000_v2_multiseed")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--model-family", default="TreeCross-MPD v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_seed(seed, args, seed_dir):
    metrics_path = seed_dir / "metrics.csv"
    if args.skip_existing and metrics_path.exists():
        print(f"[seed={seed}] existing metrics found, skipping training.")
        return

    cmd = [
        sys.executable,
        "scripts/train_treecross.py",
        "--features",
        *args.features,
        "--out-dir",
        str(seed_dir),
        "--random-state",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--rf-trees",
        str(args.rf_trees),
        "--blend-step",
        str(args.blend_step),
        "--model-family",
        args.model_family,
        "--device",
        args.device,
    ]
    print(f"[seed={seed}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def aggregate(args, out_dir):
    metric_frames = []
    for seed in args.seeds:
        metrics_path = out_dir / f"seed_{seed}" / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file for seed {seed}: {metrics_path}")
        df = pd.read_csv(metrics_path)
        df.insert(0, "seed", seed)
        metric_frames.append(df)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(out_dir / "all_metrics.csv", index=False)

    test_metrics = all_metrics[all_metrics["split"].eq("test")].copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"]
    summary_parts = []
    for stat_name, func in [("mean", "mean"), ("std", "std")]:
        part = (
            test_metrics.groupby(["setting", "model"], as_index=False)[metric_cols]
            .agg(func)
            .rename(columns={col: f"{col}_{stat_name}" for col in metric_cols})
        )
        summary_parts.append(part)

    summary = summary_parts[0].merge(summary_parts[1], on=["setting", "model"], how="left")
    summary["seed_count"] = len(args.seeds)
    summary = summary.sort_values(["setting", "f1_mean"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)

    payload = {
        "seeds": args.seeds,
        "features": args.features,
        "out_dir": str(out_dir),
        "all_metrics": str(out_dir / "all_metrics.csv"),
        "summary": str(out_dir / "summary.csv"),
        "best_by_setting": summary.groupby("setting").head(5).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["best_by_setting"], indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_seed(seed, args, out_dir / f"seed_{seed}")
    aggregate(args, out_dir)


if __name__ == "__main__":
    main()
