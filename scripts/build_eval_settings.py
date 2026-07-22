import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/base_feature_table_benign1000.csv")
    parser.add_argument("--out-dir", default="data/processed/eval_settings/benign1000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=900)
    return parser.parse_args()


def sample_balanced(df, n_per_class, random_state):
    parts = []
    for label, group in df.groupby("label"):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_random_setting(df, n_per_class, random_state):
    base = df[df["has_package_json"].eq(1)].copy()
    return sample_balanced(base, n_per_class, random_state)


def build_scoped_balanced_setting(df, random_state):
    base = df[df["has_package_json"].eq(1)].copy()
    groups = []
    sizes = []
    for label in [0, 1]:
        for scoped in [0, 1]:
            g = base[(base["label"].eq(label)) & (base["is_scoped"].eq(scoped))]
            sizes.append(len(g))
    n = min(sizes)
    for label in [0, 1]:
        for scoped in [0, 1]:
            g = base[(base["label"].eq(label)) & (base["is_scoped"].eq(scoped))]
            groups.append(g.sample(n=n, random_state=random_state))
    return pd.concat(groups).sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_hard_case_setting(df, random_state):
    base = df[df["has_package_json"].eq(1)].copy()

    benign_hard = base[
        base["label"].eq(0)
        & (
            base["has_install_lifecycle_script"].eq(1)
            | base["has_risky_script_token"].eq(1)
            | base["has_confusion_hint"].eq(1)
            | base["script_count"].ge(5)
        )
    ].copy()
    malicious_hard = base[
        base["label"].eq(1)
        & (
            base["has_install_lifecycle_script"].eq(0)
            | base["has_risky_install_script_token"].eq(0)
            | base["script_count"].le(1)
            | base["confusion_hint_count"].ge(1)
        )
    ].copy()

    n = min(len(benign_hard), len(malicious_hard))
    if n == 0:
        return pd.DataFrame(columns=base.columns)
    out = pd.concat(
        [
            benign_hard.sample(n=n, random_state=random_state),
            malicious_hard.sample(n=n, random_state=random_state),
        ]
    )
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


def describe(df):
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "is_scoped_by_class": {
            f"{int(label)}:{int(scoped)}": int(count)
            for (label, scoped), count in df.groupby(["label", "is_scoped"]).size().items()
        },
        "install_lifecycle_rate_by_class": {
            str(k): float(v)
            for k, v in df.groupby("label")["has_install_lifecycle_script"].mean().items()
        },
        "risky_install_rate_by_class": {
            str(k): float(v)
            for k, v in df.groupby("label")["has_risky_install_script_token"].mean().items()
        },
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    df["label"] = df["label"].astype(int)

    settings = {
        "random_balanced": build_random_setting(df, args.n_per_class, args.random_state),
        "scoped_balanced": build_scoped_balanced_setting(df, args.random_state),
        "hard_case": build_hard_case_setting(df, args.random_state),
    }

    summary = {}
    for name, setting_df in settings.items():
        path = out_dir / f"{name}.csv"
        setting_df.to_csv(path, index=False)
        summary[name] = describe(setting_df)
        summary[name]["path"] = str(path)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
