import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


TOOL_RESULT_PREFIXES = [
    "NPMStudy/Core/ToolDetection/DetectionResults/",
    "NPMStudy/ToolDetection/DetectionResults/",
]

REPORT_KINDS = [
    "benign_reports.json",
    "malicious_reports.json",
    "false_positives.json",
    "false_negatives.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-dir", default="data/processed/eval_settings/random2000")
    parser.add_argument("--package-index", default="data/processed/npmstudy_index/package_index.csv")
    parser.add_argument("--tool-zip", default="data/raw/npmstudy/NPMStudy_large_65055888.zip")
    parser.add_argument("--out-dir", default="runs/tool_baselines/random2000_corrected")
    return parser.parse_args()


def normalize_package_key(value):
    return str(value).strip().replace("##", "/")


def normalize_prediction(value):
    text = str(value).strip().lower()
    if text in {"malware", "malicious", "1", "true", "yes"}:
        return 1
    if text in {"benign", "clean", "0", "false", "no"}:
        return 0
    return None


def find_tool_result_prefix(names):
    for prefix in TOOL_RESULT_PREFIXES:
        if any(n.startswith(prefix) for n in names):
            return prefix
    return None


def report_priority(report_kind):
    if report_kind == "false_positives.json":
        return 3
    if report_kind == "false_negatives.json":
        return 3
    if report_kind == "malicious_reports.json":
        return 2
    if report_kind == "benign_reports.json":
        return 1
    return 0


def load_tool_predictions_from_zip(zip_path):
    predictions = defaultdict(dict)
    stats = defaultdict(Counter)

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        prefix = find_tool_result_prefix(names)
        if not prefix:
            raise RuntimeError(f"No NPMStudy tool results found in {zip_path}")

        report_files = [
            n
            for n in names
            if n.startswith(prefix) and any(n.endswith(kind) for kind in REPORT_KINDS)
        ]

        for report in report_files:
            rest = report[len(prefix) :]
            parts = rest.split("/")
            if len(parts) < 2:
                continue
            tool = parts[0]
            report_kind = parts[1]
            priority = report_priority(report_kind)
            data = json.loads(zf.read(report).decode("utf-8", errors="replace"))
            if not isinstance(data, dict):
                continue

            for raw_key, item in data.items():
                if not isinstance(item, dict):
                    continue
                package_key = normalize_package_key(raw_key)
                prediction = normalize_prediction(item.get("prediction"))
                actual = normalize_prediction(item.get("actual"))
                if prediction is None:
                    continue

                previous = predictions[tool].get(package_key)
                if previous and previous["priority"] > priority:
                    continue
                predictions[tool][package_key] = {
                    "prediction": prediction,
                    "actual": actual,
                    "report_kind": report_kind.replace(".json", ""),
                    "priority": priority,
                }
                stats[tool][report_kind.replace(".json", "")] += 1

    return predictions, stats


def evaluate_one(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_setting(setting_path, tool_predictions, out_dir):
    setting_name = setting_path.stem
    setting = pd.read_csv(setting_path)
    merged = setting[["package_key", "package_name", "version", "label"]].copy()
    merged["package_key"] = merged["package_key"].map(normalize_package_key)
    rows = []
    preds = []
    for tool in sorted(tool_predictions):
        temp = merged[["package_key", "package_name", "version", "label"]].copy()
        temp["tool"] = tool
        temp["prediction"] = temp["package_key"].map(
            lambda key: tool_predictions[tool].get(key, {}).get("prediction")
        )
        temp["tool_report_kind"] = temp["package_key"].map(
            lambda key: tool_predictions[tool].get(key, {}).get("report_kind")
        )
        covered = temp[temp["prediction"].notna()].copy()
        coverage = len(covered) / max(1, len(temp))
        if covered.empty or covered["label"].nunique() < 2:
            rows.append(
                {
                    "setting": setting_name,
                    "tool": tool,
                    "coverage": coverage,
                    "covered_rows": int(len(covered)),
                    "total_rows": int(len(temp)),
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "tp": None,
                    "fp": None,
                    "tn": None,
                    "fn": None,
                }
            )
            continue
        metrics = evaluate_one(covered["label"].astype(int), covered["prediction"].astype(int))
        rows.append(
            {
                "setting": setting_name,
                "tool": tool,
                "coverage": coverage,
                "covered_rows": int(len(covered)),
                "total_rows": int(len(temp)),
                **metrics,
            }
        )
        covered["error_type"] = [
            "false_positive"
            if y == 0 and p == 1
            else "false_negative"
            if y == 1 and p == 0
            else "correct"
            for y, p in zip(covered["label"].astype(int), covered["prediction"].astype(int))
        ]
        preds.append(covered)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / f"{setting_name}_metrics.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out_dir / f"{setting_name}_predictions.csv", index=False)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = Path(args.settings_dir)
    package_index = pd.read_csv(args.package_index)
    tool_predictions, tool_stats = load_tool_predictions_from_zip(Path(args.tool_zip))

    all_rows = []
    for setting_path in sorted(settings_dir.glob("*.csv")):
        all_rows.extend(evaluate_setting(setting_path, tool_predictions, out_dir))

    all_metrics = pd.DataFrame(all_rows)
    all_metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "settings_dir": args.settings_dir,
        "package_index": args.package_index,
        "tool_zip": args.tool_zip,
        "tools": sorted(tool_predictions),
        "tool_report_counts": {tool: dict(counter) for tool, counter in tool_stats.items()},
        "package_index_rows": int(len(package_index)),
        "metrics": all_metrics.to_dict(orient="records"),
        "output": str(out_dir / "metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

