import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


TOOL_RESULT_PREFIXES = [
    "NPMStudy/Core/ToolDetection/DetectionResults/",
    "NPMStudy/ToolDetection/DetectionResults/",
]


def normalize_package_name(name: str) -> str:
    return name.replace("##", "/").replace("@", "@", 1)


def package_key(package_name: str, version: str) -> str:
    return f"{normalize_package_name(package_name)}/{version}"


def parse_benign_list_path(line: str):
    parts = line.strip().split("/")
    if len(parts) < 3:
        return None
    version = parts[-2]
    raw_name = "/".join(parts[:-2])
    if raw_name.startswith("@") and "@" in raw_name[1:]:
        scope, name = raw_name[1:].split("@", 1)
        pkg = f"@{scope}/{name}"
    else:
        pkg = raw_name
    return pkg, version


def parse_malware_tgz_path(name: str):
    prefix = "NPMStudy/Dataset/zip_malware/"
    if not name.startswith(prefix) or not name.endswith(".tgz"):
        return None
    rest = name[len(prefix) :]
    parts = rest.split("/")
    if len(parts) < 3:
        return None
    raw_pkg = parts[0]
    version = parts[1]
    return normalize_package_name(raw_pkg), version


def split_jsonish_labels(value):
    if not value:
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, dict):
        values = list(value.values())
    else:
        values = [value]

    labels = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, list):
            labels.extend(str(x).strip() for x in item if str(x).strip())
        elif isinstance(item, dict):
            labels.extend(str(x).strip() for x in item.values() if str(x).strip())
        else:
            text = str(item)
            for chunk in re.split(r"[,;/|]\s*", text):
                chunk = chunk.strip()
                if chunk:
                    labels.append(chunk)
    return sorted(set(labels))


def load_large_dataset_index(zip_path: Path):
    packages = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]

        benign_lists = [n for n in names if n.endswith("Dataset/zip_benign/zip_benign_list.txt")]
        if not benign_lists:
            raise RuntimeError(f"No zip_benign_list.txt found in {zip_path}")
        benign_lines = zf.read(benign_lists[0]).decode("utf-8", errors="replace").splitlines()
        for line in benign_lines:
            parsed = parse_benign_list_path(line)
            if not parsed:
                continue
            pkg, version = parsed
            key = package_key(pkg, version)
            packages.setdefault(
                key,
                {
                    "package_key": key,
                    "package_name": pkg,
                    "version": version,
                    "label": 0,
                    "benign_list_path": line.strip(),
                    "malware_tgz_path": "",
                    "malware_tgz_size": 0,
                },
            )

        for name in names:
            parsed = parse_malware_tgz_path(name)
            if not parsed:
                continue
            pkg, version = parsed
            info = zf.getinfo(name)
            key = package_key(pkg, version)
            packages[key] = {
                "package_key": key,
                "package_name": pkg,
                "version": version,
                "label": 1,
                "benign_list_path": "",
                "malware_tgz_path": name,
                "malware_tgz_size": info.file_size,
            }

    return packages


def find_tool_result_prefix(names):
    for prefix in TOOL_RESULT_PREFIXES:
        if any(n.startswith(prefix) for n in names):
            return prefix
    return None


def load_tool_predictions(zip_path: Path):
    tool_predictions = defaultdict(dict)
    tool_stats = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        prefix = find_tool_result_prefix(names)
        if not prefix:
            return tool_predictions, tool_stats

        report_files = [
            n
            for n in names
            if n.startswith(prefix)
            and n.endswith("_reports.json")
            and not n.endswith("false_positives.json")
            and not n.endswith("false_negatives.json")
        ]
        for report in report_files:
            rest = report[len(prefix) :]
            parts = rest.split("/")
            if len(parts) < 2:
                continue
            tool = parts[0]
            kind = parts[1].replace("_reports.json", "")
            data = json.loads(zf.read(report).decode("utf-8", errors="replace"))
            pred_counts = Counter()
            for key, item in data.items():
                if not isinstance(item, dict):
                    continue
                norm_key = normalize_package_name(key)
                pred = item.get("prediction", "")
                actual = item.get("actual", "")
                pred_counts[pred] += 1
                tool_predictions[norm_key][f"{tool}_prediction"] = pred
                tool_predictions[norm_key][f"{tool}_actual"] = actual
                tool_predictions[norm_key][f"{tool}_reported_in"] = kind
            tool_stats.setdefault(tool, Counter()).update(pred_counts)
    return tool_predictions, tool_stats


def load_small_labels(small_zip_path: Path, snippets_jsonl: Path):
    label_features = {}
    with zipfile.ZipFile(small_zip_path) as zf:
        label_files = [
            n
            for n in zf.namelist()
            if not n.startswith("__MACOSX/")
            and n.startswith("NPMStudy/Data/cleaning/package_label/")
            and n.endswith("-analysis.json")
        ]
        for name in label_files:
            try:
                obj = json.loads(zf.read(name).decode("utf-8", errors="replace"))
            except Exception:
                continue
            pkg = obj.get("package_name")
            version = obj.get("version")
            if not pkg or not version:
                continue
            key = package_key(pkg, version)
            malicious_files = obj.get("malicious_files") or []
            malicious_code = obj.get("malicious_code") or {}
            behavior_summaries = obj.get("behavior_summaries") or {}
            attack_types = obj.get("attack_types") or {}
            label_features[key] = {
                "has_package_label": 1,
                "malicious_file_count": len(malicious_files) if isinstance(malicious_files, list) else 0,
                "malicious_code_file_count": len(malicious_code) if isinstance(malicious_code, dict) else 0,
                "behavior_summary_count": len(behavior_summaries) if isinstance(behavior_summaries, dict) else 0,
                "attack_type_count": len(attack_types) if isinstance(attack_types, dict) else 0,
            }

    snippet_counts = Counter()
    detection_types = defaultdict(Counter)
    behavior_labels = defaultdict(Counter)
    evasion_labels = defaultdict(Counter)
    if snippets_jsonl.exists():
        with snippets_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = package_key(obj.get("package_name", ""), obj.get("version", ""))
                if key == "/":
                    continue
                snippet_counts[key] += 1
                detection_type = obj.get("detection_type") or "untagged"
                detection_types[key][detection_type] += 1
                for label in split_jsonish_labels(obj.get("behavior_formal")):
                    behavior_labels[key][label] += 1
                for label in split_jsonish_labels(obj.get("evasion_formal")):
                    evasion_labels[key][label] += 1

    for key, count in snippet_counts.items():
        item = label_features.setdefault(key, {})
        item["snippet_count"] = count
        item["detection_types"] = "|".join(sorted(detection_types[key]))
        item["behavior_labels"] = "|".join(sorted(behavior_labels[key]))
        item["evasion_labels"] = "|".join(sorted(evasion_labels[key]))
        item["behavior_label_count"] = len(behavior_labels[key])
        item["evasion_label_count"] = len(evasion_labels[key])

    return label_features


def write_outputs(packages, tool_predictions, label_features, out_dir: Path, tool_stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tool_cols = sorted({col for preds in tool_predictions.values() for col in preds})
    extra_cols = [
        "has_package_label",
        "malicious_file_count",
        "malicious_code_file_count",
        "behavior_summary_count",
        "attack_type_count",
        "snippet_count",
        "detection_types",
        "behavior_labels",
        "evasion_labels",
        "behavior_label_count",
        "evasion_label_count",
    ]
    base_cols = [
        "package_key",
        "package_name",
        "version",
        "label",
        "benign_list_path",
        "malware_tgz_path",
        "malware_tgz_size",
    ]
    fieldnames = base_cols + extra_cols + all_tool_cols

    rows = []
    for key in sorted(packages):
        row = dict(packages[key])
        row.update({c: "" for c in extra_cols})
        row.update(label_features.get(key, {}))
        row.update(tool_predictions.get(key, {}))
        rows.append(row)

    csv_path = out_dir / "package_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = out_dir / "package_index.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_packages": len(rows),
        "malicious_packages": sum(1 for r in rows if int(r["label"]) == 1),
        "benign_packages": sum(1 for r in rows if int(r["label"]) == 0),
        "packages_with_tool_predictions": sum(1 for r in rows if r["package_key"] in tool_predictions),
        "packages_with_small_labels_or_snippets": sum(1 for r in rows if r["package_key"] in label_features),
        "tool_prediction_columns": all_tool_cols,
        "tool_prediction_stats": {tool: dict(counter) for tool, counter in tool_stats.items()},
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        },
    }
    summary_path = out_dir / "package_index.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-zip", default="data/raw/npmstudy/NPMStudy_large_65055888.zip")
    parser.add_argument("--tool-zip", default="data/raw/npmstudy/NPMStudy_large_65055888.zip")
    parser.add_argument("--small-zip", default="data/raw/npmstudy/NPMStudy_small_63179326.zip")
    parser.add_argument("--snippets-jsonl", default="data/processed/npmstudy_small/malicious_snippets.jsonl")
    parser.add_argument("--out-dir", default="data/processed/npmstudy_index")
    args = parser.parse_args()

    packages = load_large_dataset_index(Path(args.large_zip))
    tool_predictions, tool_stats = load_tool_predictions(Path(args.tool_zip))
    label_features = load_small_labels(Path(args.small_zip), Path(args.snippets_jsonl))
    summary = write_outputs(packages, tool_predictions, label_features, Path(args.out_dir), tool_stats)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

