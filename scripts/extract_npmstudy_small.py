"""Extract structured artifacts from the NPMStudy small figshare zip.

The zip contains many small files. This script reads it directly and writes
compact CSV/JSONL artifacts for modeling and data inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any


def safe_json_loads(raw: bytes) -> Any | None:
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_selected_benign(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    path = "NPMStudy/Data/cleaning/selected_benign_packages.txt"
    rows = []
    if path not in zf.namelist():
        return rows
    for line in zf.read(path).decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or "/" not in line:
            continue
        package_name, version = line.rsplit("/", 1)
        rows.append({"package_name": package_name, "version": version, "label": "benign"})
    return rows


def read_analysis_summary(zf: zipfile.ZipFile) -> dict[str, Any]:
    path = "NPMStudy/Data/cleaning/package_label/analysis_summary.json"
    if path not in zf.namelist():
        return {}
    raw = zf.read(path)
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        return parsed
    text = raw.decode("utf-8", errors="replace")
    summary: dict[str, Any] = {}
    for key in [
        "total_packages",
        "packages_with_detected_malicious_code",
        "packages_with_no_detected_malicious_code",
    ]:
        match = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if match:
            summary[key] = int(match.group(1))
    match = re.search(r'"progress"\s*:\s*"([^"]+)"', text)
    if match:
        summary["progress"] = match.group(1)
    return summary


def iter_package_label_rows(zf: zipfile.ZipFile):
    prefix = "NPMStudy/Data/cleaning/package_label/"
    for name in zf.namelist():
        if name.startswith("__MACOSX") or not name.startswith(prefix):
            continue
        if not name.endswith(".json") or name.endswith("analysis_summary.json"):
            continue
        obj = safe_json_loads(zf.read(name))
        if not isinstance(obj, dict):
            yield {
                "source_path": name,
                "parse_status": "json_error",
                "package_name": "",
                "version": "",
                "malicious_file_count": 0,
                "snippet_count": 0,
                "attack_types": "",
                "behavior_summaries": "",
            }
            continue
        malicious_code = obj.get("malicious_code") or {}
        attack_types = obj.get("attack_types") or {}
        behavior_summaries = obj.get("behavior_summaries") or {}
        yield {
            "source_path": name,
            "parse_status": "ok",
            "package_name": obj.get("package_name", ""),
            "version": obj.get("version", ""),
            "malicious_file_count": len(obj.get("malicious_files") or []),
            "snippet_count": len(malicious_code),
            "attack_types": "|".join(sorted({compact_text(str(v)) for v in attack_types.values() if v})),
            "behavior_summaries": " || ".join(compact_text(str(v)) for v in behavior_summaries.values() if v),
        }


def iter_malicious_snippets(zf: zipfile.ZipFile):
    prefix = "NPMStudy/Analysis/code_snipptes/malware_snippets/"
    for name in zf.namelist():
        if name.startswith("__MACOSX") or not name.startswith(prefix):
            continue
        if not name.endswith("/result.json"):
            continue
        obj = safe_json_loads(zf.read(name))
        if not isinstance(obj, dict):
            continue
        meta = obj.get("metadata") or {}
        package_name = meta.get("package_name", "")
        version = meta.get("version", "")
        for idx, snippet in enumerate(obj.get("malicious_snippets") or []):
            behavior_formal = snippet.get("behavior_formal") or []
            evasion_formal = snippet.get("evasion_formal") or []
            yield {
                "source_path": name,
                "package_name": package_name,
                "version": version,
                "snippet_index": idx,
                "file": snippet.get("file", ""),
                "line_number": snippet.get("line_number", ""),
                "detection_type": snippet.get("type", ""),
                "malicious_code": snippet.get("malicious_code", ""),
                "behavior_summary": snippet.get("behavior_summary", ""),
                "evasion_techniques": snippet.get("evasion_techniques", ""),
                "behavior_formal": "|".join(behavior_formal),
                "evasion_formal": "|".join(evasion_formal),
                "validate_behavior_formal": "|".join(snippet.get("validate_behavior_formal") or []),
                "validate_evasion_formal": "|".join(snippet.get("validate_evasion_formal") or []),
                "is_validated": bool(snippet.get("is_validated")),
                "hash_value": snippet.get("hash_value", ""),
                "label": "malicious",
            }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/raw/npmstudy/NPMStudy_small_63179326.zip")
    parser.add_argument("--out-dir", default="data/processed/npmstudy_small")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out_dir)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        benign_rows = read_selected_benign(zf)
        package_rows = list(iter_package_label_rows(zf))
        snippet_rows = list(iter_malicious_snippets(zf))
        summary = read_analysis_summary(zf)

    write_csv(out_dir / "selected_benign_packages.csv", benign_rows)
    write_csv(out_dir / "package_labels.csv", package_rows)
    write_csv(out_dir / "malicious_snippets.csv", snippet_rows)
    write_jsonl(out_dir / "malicious_snippets.jsonl", snippet_rows)

    metadata = {
        "source_zip": str(zip_path),
        "selected_benign_packages": len(benign_rows),
        "package_label_files": len(package_rows),
        "package_label_parse_errors": sum(1 for row in package_rows if row["parse_status"] != "ok"),
        "malicious_snippets": len(snippet_rows),
        "malicious_package_versions_from_snippets": len(
            {(row["package_name"], row["version"]) for row in snippet_rows}
        ),
        "analysis_summary": {
            "total_packages": summary.get("total_packages"),
            "packages_with_detected_malicious_code": summary.get("packages_with_detected_malicious_code"),
            "packages_with_no_detected_malicious_code": summary.get("packages_with_no_detected_malicious_code"),
            "progress": summary.get("progress"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
