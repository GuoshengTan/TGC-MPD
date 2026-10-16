import argparse
import json
from pathlib import Path

import pandas as pd


SETTINGS = ["scoped_balanced", "hard_case"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="paper_assets")
    parser.add_argument("--baseline-summary", default="runs/baselines/random2000_multiseed/summary.csv")
    parser.add_argument("--treecross-summary", default="runs/treecross/random2000_v3_multiseed/summary.csv")
    parser.add_argument("--treecross-all", default="runs/treecross/random2000_v3_multiseed/all_metrics.csv")
    parser.add_argument("--tool-metrics", default="runs/tool_baselines/random2000_corrected/metrics.csv")
    parser.add_argument("--seed-for-cases", type=int, default=42)
    parser.add_argument("--case-limit", type=int, default=8)
    return parser.parse_args()


def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def write_markdown_table(df, path):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_tree_summary(path):
    df = pd.read_csv(path)
    for col in df.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dataset_table(out_dir):
    rows = []
    for setting in SETTINGS:
        path = Path("data/processed/eval_settings/random2000") / f"{setting}.csv"
        df = pd.read_csv(path)
        rows.append(
            {
                "Setting": setting,
                "Rows": len(df),
                "Benign": int((df["label"] == 0).sum()),
                "Malicious": int((df["label"] == 1).sum()),
                "Scoped benign": int(((df["label"] == 0) & (df["is_scoped"] == 1)).sum()),
                "Scoped malicious": int(((df["label"] == 1) & (df["is_scoped"] == 1)).sum()),
                "Install-script benign": f"{df[df['label'] == 0]['has_install_lifecycle_script'].mean():.3f}",
                "Install-script malicious": f"{df[df['label'] == 1]['has_install_lifecycle_script'].mean():.3f}",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table1_dataset_settings.csv", index=False)
    write_markdown_table(table, out_dir / "table1_dataset_settings.md")
    return table


def baseline_rows(setting):
    run_name = "random2000_scoped_balanced" if setting == "scoped_balanced" else "random2000_hard_case"
    path = Path("runs/baselines") / run_name / "metrics.csv"
    df = pd.read_csv(path)
    df = df[df["split"].eq("test")].copy()
    name_map = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "hist_gradient_boosting": "HGB",
        "mlp": "MLP",
    }
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Setting": setting,
                "Model": name_map.get(r["model"], r["model"]),
                "Source": "reimplemented baseline",
                **{m: f"{float(r[m]):.4f}" for m in METRICS},
            }
        )
    return rows


def baseline_summary_rows(setting, baseline_summary):
    if baseline_summary is None:
        return baseline_rows(setting)
    name_map = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "hist_gradient_boosting": "HGB",
        "mlp": "MLP",
    }
    rows = []
    sub = baseline_summary[baseline_summary["setting"].eq(setting)].copy()
    for model, label in name_map.items():
        hit = sub[sub["model"].eq(model)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        row = {"Setting": setting, "Model": label, "Source": "reimplemented baseline, multi-seed"}
        for metric in METRICS:
            row[metric] = fmt_mean_std(r[f"{metric}_mean"], r[f"{metric}_std"])
        rows.append(row)
    return rows


def tool_rows(setting, tool_metrics):
    keep = {
        "guarddog": "GuardDog",
        "sap_XGB": "SAP-XGB",
        "packj_static": "Packj-static",
        "ossgadget": "OSSGadget",
    }
    rows = []
    sub = tool_metrics[tool_metrics["setting"].eq(setting)].copy()
    for tool, label in keep.items():
        hit = sub[sub["tool"].eq(tool)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "Setting": setting,
                "Model": label,
                "Source": "NPMStudy tool output",
                "accuracy": f"{float(r['accuracy']):.4f}",
                "precision": f"{float(r['precision']):.4f}",
                "recall": f"{float(r['recall']):.4f}",
                "f1": f"{float(r['f1']):.4f}",
                "roc_auc": "",
                "pr_auc": "",
            }
        )
    return rows


def tree_model_row(setting, summary, model, label, source):
    hit = summary[(summary["setting"].eq(setting)) & (summary["model"].eq(model))]
    if hit.empty:
        return None
    r = hit.iloc[0]
    row = {"Setting": setting, "Model": label, "Source": source}
    for metric in METRICS:
        row[metric] = fmt_mean_std(r[f"{metric}_mean"], r[f"{metric}_std"])
    return row


def main_results_table(out_dir, tree_summary, tool_metrics, baseline_summary):
    rows = []
    for setting in SETTINGS:
        rows.extend(baseline_summary_rows(setting, baseline_summary))
        rows.extend(tool_rows(setting, tool_metrics))
        for model, label in [
            ("hgb_teacher", "HGB teacher"),
            ("tree_teacher_avg_fixed05", "Tree teacher avg"),
            ("gate_hgb_treecross_full", "TGC-MPD"),
        ]:
            row = tree_model_row(setting, tree_summary, model, label, "multi-seed")
            if row:
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table2_main_results.csv", index=False)
    write_markdown_table(table, out_dir / "table2_main_results.md")
    return table


def ablation_table(out_dir, tree_summary):
    rows = []
    ablations = [
        ("hgb_teacher", "HGB teacher"),
        ("tree_teacher_avg", "RF/HGB teacher avg"),
        ("treecross_prior", "TreeCross prior"),
        ("treecross_full", "TreeCross full"),
        ("blend_hgb_treecross_full", "HGB + TreeCross blend"),
        ("gate_hgb_treecross_full", "TGC-MPD gate"),
    ]
    for setting in SETTINGS:
        for model, label in ablations:
            hit = tree_summary[(tree_summary["setting"].eq(setting)) & (tree_summary["model"].eq(model))]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "Setting": setting,
                    "Variant": label,
                    "F1": fmt_mean_std(r["f1_mean"], r["f1_std"]),
                    "Precision": fmt_mean_std(r["precision_mean"], r["precision_std"]),
                    "Recall": fmt_mean_std(r["recall_mean"], r["recall_std"]),
                    "PR-AUC": fmt_mean_std(r["pr_auc_mean"], r["pr_auc_std"]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table3_ablation.csv", index=False)
    write_markdown_table(table, out_dir / "table3_ablation.md")
    return table


def enrich_cases(cases, test_df):
    cols = [
        "package_key",
        "package_name",
        "version",
        "label",
        "is_scoped",
        "has_confusion_hint",
        "has_install_lifecycle_script",
        "has_risky_script_token",
        "has_risky_install_script_token",
        "dependency_count",
        "dev_dependency_count",
        "total_dependency_count",
        "script_count",
        "risky_script_token_count",
        "files_count",
        "js_file_count",
        "total_source_bytes",
    ]
    keep = [c for c in cols if c in test_df.columns]
    return cases.merge(test_df[keep], on=["package_key", "package_name", "version", "label"], how="left")


def case_study_table(out_dir, seed, case_limit):
    case_dir = Path("runs/treecross/random2000_v3_multiseed") / f"seed_{seed}" / "hard_case"
    pred_path = case_dir / "predictions.csv"
    test_path = case_dir / "test.csv"
    pred = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    hgb = pred[pred["model"].eq("hgb_teacher")][
        ["package_key", "package_name", "version", "label", "score", "prediction", "error_type"]
    ].rename(columns={"score": "hgb_score", "prediction": "hgb_prediction", "error_type": "hgb_error"})
    tgc = pred[pred["model"].eq("gate_hgb_treecross_full")][
        ["package_key", "package_name", "version", "label", "score", "prediction", "error_type"]
    ].rename(columns={"score": "tgc_score", "prediction": "tgc_prediction", "error_type": "tgc_error"})
    merged = hgb.merge(tgc, on=["package_key", "package_name", "version", "label"])

    fixed = merged[
        merged["label"].eq(1) & merged["hgb_prediction"].eq(0) & merged["tgc_prediction"].eq(1)
    ].copy()
    regressions = merged[
        merged["label"].eq(0) & merged["hgb_prediction"].eq(0) & merged["tgc_prediction"].eq(1)
    ].copy()
    both_wrong = merged[
        (merged["label"].eq(1) & merged["hgb_prediction"].eq(0) & merged["tgc_prediction"].eq(0))
        | (merged["label"].eq(0) & merged["hgb_prediction"].eq(1) & merged["tgc_prediction"].eq(1))
    ].copy()

    fixed["Case type"] = "TGC fixes HGB false negative"
    regressions["Case type"] = "TGC false positive"
    both_wrong["Case type"] = "Both models wrong"

    cases = pd.concat(
        [
            fixed.sort_values("tgc_score", ascending=False).head(case_limit),
            regressions.sort_values("tgc_score", ascending=False).head(case_limit),
            both_wrong.head(case_limit),
        ],
        ignore_index=True,
    )
    cases = enrich_cases(cases, test)
    display_cols = [
        "Case type",
        "package_key",
        "label",
        "hgb_score",
        "tgc_score",
        "hgb_prediction",
        "tgc_prediction",
        "is_scoped",
        "has_confusion_hint",
        "has_install_lifecycle_script",
        "has_risky_script_token",
        "has_risky_install_script_token",
        "total_dependency_count",
        "script_count",
        "risky_script_token_count",
        "js_file_count",
        "total_source_bytes",
    ]
    display_cols = [c for c in display_cols if c in cases.columns]
    table = cases[display_cols].copy()
    for col in ["hgb_score", "tgc_score"]:
        if col in table:
            table[col] = table[col].map(lambda x: f"{float(x):.4f}")
    table.to_csv(out_dir / "table4_case_study.csv", index=False)
    write_markdown_table(table, out_dir / "table4_case_study.md")
    return table


def pair_predictions(pred, left_model, right_model):
    left = pred[pred["model"].eq(left_model)][
        ["package_key", "package_name", "version", "label", "score", "prediction", "error_type"]
    ].rename(columns={"score": "left_score", "prediction": "left_prediction", "error_type": "left_error"})
    right = pred[pred["model"].eq(right_model)][
        ["package_key", "package_name", "version", "label", "score", "prediction", "error_type"]
    ].rename(columns={"score": "right_score", "prediction": "right_prediction", "error_type": "right_error"})
    return left.merge(right, on=["package_key", "package_name", "version", "label"])


def error_transfer_tables(out_dir, case_limit):
    rows = []
    case_rows = []
    base = Path("runs/treecross/random2000_v3_multiseed")
    for seed_dir in sorted(base.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        for setting in SETTINGS:
            pred_path = seed_dir / setting / "predictions.csv"
            test_path = seed_dir / setting / "test.csv"
            if not pred_path.exists():
                continue
            pred = pd.read_csv(pred_path)
            paired = pair_predictions(pred, "hgb_teacher", "gate_hgb_treecross_full")
            left_correct = paired["left_prediction"].eq(paired["label"])
            right_correct = paired["right_prediction"].eq(paired["label"])
            both_correct = int((left_correct & right_correct).sum())
            both_wrong = int((~left_correct & ~right_correct).sum())
            hgb_wrong_tgc_correct = paired[~left_correct & right_correct].copy()
            hgb_correct_tgc_wrong = paired[left_correct & ~right_correct].copy()

            fixed_fn = int(
                (
                    paired["label"].eq(1)
                    & paired["left_prediction"].eq(0)
                    & paired["right_prediction"].eq(1)
                ).sum()
            )
            introduced_fp = int(
                (
                    paired["label"].eq(0)
                    & paired["left_prediction"].eq(0)
                    & paired["right_prediction"].eq(1)
                ).sum()
            )
            rows.append(
                {
                    "Seed": seed,
                    "Setting": setting,
                    "Both correct": both_correct,
                    "Both wrong": both_wrong,
                    "HGB wrong, TGC correct": int(len(hgb_wrong_tgc_correct)),
                    "HGB correct, TGC wrong": int(len(hgb_correct_tgc_wrong)),
                    "Net corrected": int(len(hgb_wrong_tgc_correct) - len(hgb_correct_tgc_wrong)),
                    "Fixed HGB false negatives": fixed_fn,
                    "Introduced false positives": introduced_fp,
                }
            )

            test_df = pd.read_csv(test_path)
            fixed = hgb_wrong_tgc_correct.copy()
            fixed["Seed"] = seed
            fixed["Setting"] = setting
            fixed["Case type"] = "HGB wrong, TGC correct"
            regressed = hgb_correct_tgc_wrong.copy()
            regressed["Seed"] = seed
            regressed["Setting"] = setting
            regressed["Case type"] = "HGB correct, TGC wrong"
            examples = pd.concat(
                [
                    fixed.sort_values("right_score", ascending=False).head(case_limit),
                    regressed.sort_values("right_score", ascending=False).head(case_limit),
                ],
                ignore_index=True,
            )
            if not examples.empty:
                examples = enrich_cases(
                    examples.rename(
                        columns={
                            "left_score": "hgb_score",
                            "right_score": "tgc_score",
                            "left_prediction": "hgb_prediction",
                            "right_prediction": "tgc_prediction",
                        }
                    ),
                    test_df,
                )
                case_rows.append(examples)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table6_error_transfer.csv", index=False)
    write_markdown_table(table, out_dir / "table6_error_transfer.md")

    if case_rows:
        cases = pd.concat(case_rows, ignore_index=True)
        display_cols = [
            "Seed",
            "Setting",
            "Case type",
            "package_key",
            "label",
            "hgb_score",
            "tgc_score",
            "hgb_prediction",
            "tgc_prediction",
            "is_scoped",
            "has_confusion_hint",
            "has_install_lifecycle_script",
            "has_risky_script_token",
            "total_dependency_count",
            "script_count",
            "js_file_count",
            "total_source_bytes",
        ]
        display_cols = [c for c in display_cols if c in cases.columns]
        cases = cases[display_cols].copy()
        for col in ["hgb_score", "tgc_score"]:
            if col in cases:
                cases[col] = cases[col].map(lambda x: f"{float(x):.4f}")
        cases.to_csv(out_dir / "table7_case_candidates_all_seeds.csv", index=False)
        write_markdown_table(cases, out_dir / "table7_case_candidates_all_seeds.md")
    else:
        cases = pd.DataFrame()
    return table, cases


def feature_explanation_table(out_dir):
    rows = []
    for setting, run_name in [
        ("scoped_balanced", "random2000_scoped_balanced"),
        ("hard_case", "random2000_hard_case"),
    ]:
        path = Path("runs/baselines") / run_name / "hist_gradient_boosting_permutation_importance.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).head(12)
        for _, r in df.iterrows():
            rows.append(
                {
                    "Setting": setting,
                    "Feature": r["feature"],
                    "Importance mean": f"{float(r['importance_mean']):.4f}",
                    "Importance std": f"{float(r['importance_std']):.4f}",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table5_feature_importance.csv", index=False)
    write_markdown_table(table, out_dir / "table5_feature_importance.md")
    return table


def write_summary_doc(out_dir, tables):
    text = [
        "# Paper Experiment Assets",
        "",
        "This folder contains conference-sized experiment assets generated from the current runs.",
        "",
        "## Files",
        "",
        "- `table1_dataset_settings.*`: dataset/evaluation setting statistics.",
    "- `table2_main_results.*`: compact main comparison table.",
        "- `table3_ablation.*`: compact ablation table for the TGC-MPD design.",
        "- `table4_case_study.*`: hard-case examples comparing HGB and TGC-MPD.",
        "- `table5_feature_importance.*`: lightweight mechanism explanation via HGB permutation importance.",
        "- `table6_error_transfer.*`: cross-seed error transition counts between HGB and TGC-MPD.",
        "- `table7_case_candidates_all_seeds.*`: examples for optional case selection.",
        "",
        "## Suggested Conference Usage",
        "",
        "- Main paper: use Tables 1-4.",
        "- If space is tight, move Table 5 to appendix/supplement.",
        "- Present TGC-MPD as `gate_hgb_treecross_full`.",
        "- Prefer the multi-seed baseline rows when `runs/baselines/random2000_multiseed/summary.csv` is available.",
        "- Avoid claiming universal dominance; emphasize hard-case robustness.",
        "",
    ]
    for name, table in tables.items():
        tmp_path = out_dir / "_tmp_table.md"
        write_markdown_table(table, tmp_path)
        text.extend([f"## {name}", "", tmp_path.read_text(encoding="utf-8"), ""])
        tmp_path.unlink(missing_ok=True)
    (out_dir / "experiment_assets_summary.md").write_text("\n".join(text), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_summary = load_tree_summary(args.treecross_summary)
    baseline_summary = load_tree_summary(args.baseline_summary) if Path(args.baseline_summary).exists() else None
    tool_metrics = pd.read_csv(args.tool_metrics)
    tables = {
        "Table 1 Dataset/Evaluation Settings": dataset_table(out_dir),
        "Table 2 Main Results": main_results_table(out_dir, tree_summary, tool_metrics, baseline_summary),
        "Table 3 Ablation": ablation_table(out_dir, tree_summary),
        "Table 4 Case Study": case_study_table(out_dir, args.seed_for_cases, args.case_limit),
    }
    feature_table = feature_explanation_table(out_dir)
    error_table, case_candidates = error_transfer_tables(out_dir, args.case_limit)
    tables["Table 5 Feature Importance"] = feature_table
    tables["Table 6 Error Transfer"] = error_table
    if not case_candidates.empty:
        tables["Table 7 Case Candidates"] = case_candidates.head(12)
    write_summary_doc(out_dir, tables)
    manifest = {
        "out_dir": str(out_dir),
        "baseline_summary": args.baseline_summary if baseline_summary is not None else None,
        "treecross_summary": args.treecross_summary,
        "tool_metrics": args.tool_metrics,
        "seed_for_cases": args.seed_for_cases,
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

