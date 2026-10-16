import argparse
import csv
import io
import json
import math
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path


SCRIPT_KEYS = [
    "preinstall",
    "install",
    "postinstall",
    "prepublish",
    "prepublishOnly",
    "prepare",
    "preversion",
    "postversion",
]

RISKY_TOKENS = [
    "child_process",
    "exec",
    "execSync",
    "spawn",
    "curl",
    "wget",
    "powershell",
    "cmd.exe",
    "bash",
    "sh -c",
    "eval",
    "Function(",
    "Buffer.from",
    "base64",
    "http://",
    "https://",
    "net.connect",
    "process.env",
    "os.hostname",
    "/dev/tcp",
    "fs.readFile",
    "fs.writeFile",
]

PACKAGE_NAME_CONFUSION_HINTS = [
    "angular",
    "babel",
    "commander",
    "discord",
    "eslint",
    "express",
    "jquery",
    "lodash",
    "next",
    "react",
    "request",
    "rimraf",
    "typescript",
    "vue",
    "webpack",
]


def safe_float(value):
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_package_key(key):
    parts = key.rsplit("/", 1)
    if len(parts) != 2:
        return key, ""
    return parts[0], parts[1]


def name_features(package_name):
    scoped = package_name.startswith("@")
    if scoped and "/" in package_name:
        scope, bare = package_name.split("/", 1)
    else:
        scope, bare = "", package_name
    tokens = re.split(r"[-_.@/]+", package_name.lower())
    tokens = [t for t in tokens if t]
    digits = sum(ch.isdigit() for ch in package_name)
    letters = sum(ch.isalpha() for ch in package_name)
    separators = sum(ch in "-_./@" for ch in package_name)
    hint_hits = sum(1 for hint in PACKAGE_NAME_CONFUSION_HINTS if hint in package_name.lower())
    return {
        "name_len": len(package_name),
        "bare_name_len": len(bare),
        "is_scoped": int(scoped),
        "scope_len": len(scope),
        "name_token_count": len(tokens),
        "name_digit_count": digits,
        "name_letter_count": letters,
        "name_separator_count": separators,
        "name_digit_ratio": digits / max(1, len(package_name)),
        "name_separator_ratio": separators / max(1, len(package_name)),
        "name_entropy": entropy(package_name.lower()),
        "confusion_hint_count": hint_hits,
        "has_confusion_hint": int(hint_hits > 0),
    }


def entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def package_json_features(package_json):
    scripts = package_json.get("scripts") if isinstance(package_json, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}
    deps = package_json.get("dependencies") if isinstance(package_json, dict) else {}
    dev_deps = package_json.get("devDependencies") if isinstance(package_json, dict) else {}
    optional_deps = package_json.get("optionalDependencies") if isinstance(package_json, dict) else {}
    peer_deps = package_json.get("peerDependencies") if isinstance(package_json, dict) else {}
    keywords = package_json.get("keywords") if isinstance(package_json, dict) else []
    files = package_json.get("files") if isinstance(package_json, dict) else []

    dep_count = len(deps) if isinstance(deps, dict) else 0
    dev_dep_count = len(dev_deps) if isinstance(dev_deps, dict) else 0
    optional_dep_count = len(optional_deps) if isinstance(optional_deps, dict) else 0
    peer_dep_count = len(peer_deps) if isinstance(peer_deps, dict) else 0
    keyword_count = len(keywords) if isinstance(keywords, list) else 0
    files_count = len(files) if isinstance(files, list) else 0

    script_values = "\n".join(normalize_text(v) for v in scripts.values())
    risky_count = sum(script_values.count(token) for token in RISKY_TOKENS)
    install_script_text = "\n".join(normalize_text(scripts.get(k, "")) for k in SCRIPT_KEYS)
    has_install_script = int(any(k in scripts for k in SCRIPT_KEYS))
    risky_install_count = sum(install_script_text.count(token) for token in RISKY_TOKENS)

    return {
        "has_package_json": 1,
        "description_len": len(normalize_text(package_json.get("description", ""))),
        "has_repository": int(bool(package_json.get("repository"))),
        "has_homepage": int(bool(package_json.get("homepage"))),
        "has_author": int(bool(package_json.get("author"))),
        "has_license": int(bool(package_json.get("license"))),
        "has_main": int(bool(package_json.get("main"))),
        "dependency_count": dep_count,
        "dev_dependency_count": dev_dep_count,
        "optional_dependency_count": optional_dep_count,
        "peer_dependency_count": peer_dep_count,
        "total_dependency_count": dep_count + dev_dep_count + optional_dep_count + peer_dep_count,
        "keyword_count": keyword_count,
        "files_count": files_count,
        "script_count": len(scripts),
        "has_install_lifecycle_script": has_install_script,
        "script_text_len": len(script_values),
        "install_script_text_len": len(install_script_text),
        "risky_script_token_count": risky_count,
        "risky_install_script_token_count": risky_install_count,
        "has_risky_script_token": int(risky_count > 0),
        "has_risky_install_script_token": int(risky_install_count > 0),
    }


def source_file_features(tar_bytes):
    features = {
        "package_file_count": 0,
        "js_file_count": 0,
        "json_file_count": 0,
        "minified_js_count": 0,
        "total_source_bytes": 0,
    }
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                features["package_file_count"] += 1
                name = member.name.lower()
                features["total_source_bytes"] += max(0, member.size)
                if name.endswith(".js"):
                    features["js_file_count"] += 1
                    if ".min.js" in name or member.size > 200_000:
                        features["minified_js_count"] += 1
                elif name.endswith(".json"):
                    features["json_file_count"] += 1
    except Exception:
        pass
    return features


def read_package_json_from_tgz(tar_bytes):
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            candidates = ["package/package.json", "./package/package.json"]
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            target = None
            for candidate in candidates:
                if candidate in members:
                    target = members[candidate]
                    break
            if target is None:
                for member in members.values():
                    if member.name.endswith("/package.json"):
                        target = member
                        break
            if target is None:
                return None
            fh = tf.extractfile(target)
            if not fh:
                return None
            return json.loads(fh.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def load_malware_tgz_bytes(zf, tgz_path):
    if not tgz_path:
        return None
    if tgz_path not in zf.NameToInfo:
        return None
    return zf.read(tgz_path)


def build_features(index_csv, malware_zip, out_csv, limit=None):
    rows = []
    with open(index_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            rows.append(row)

    output_rows = []
    empty_package_json_features = package_json_features({})
    benign_tgz_map = getattr(build_features, "benign_tgz_map", {})
    with zipfile.ZipFile(malware_zip) as zf:
        for row in rows:
            package_name = row["package_name"]
            feats = {
                "package_key": row["package_key"],
                "package_name": package_name,
                "version": row["version"],
                "label": row["label"],
            }
            feats.update(name_features(package_name))
            feats.update(
                {
                    "malware_tgz_size": safe_float(row.get("malware_tgz_size")),
                    "snippet_count": safe_float(row.get("snippet_count")),
                    "behavior_label_count": safe_float(row.get("behavior_label_count")),
                    "evasion_label_count": safe_float(row.get("evasion_label_count")),
                    "has_behavior_label": int(bool(row.get("behavior_labels"))),
                    "has_evasion_label": int(bool(row.get("evasion_labels"))),
                }
            )

            package_json = None
            tar_bytes = None
            if row.get("malware_tgz_path"):
                tar_bytes = load_malware_tgz_bytes(zf, row["malware_tgz_path"])
                if tar_bytes:
                    package_json = read_package_json_from_tgz(tar_bytes)
            elif row["package_key"] in benign_tgz_map:
                tgz_path = benign_tgz_map[row["package_key"]]
                try:
                    tar_bytes = Path(tgz_path).read_bytes()
                    package_json = read_package_json_from_tgz(tar_bytes)
                except Exception:
                    tar_bytes = None
                    package_json = None

            if package_json:
                feats.update(package_json_features(package_json))
            else:
                feats.update({"has_package_json": 0})
                for key in empty_package_json_features:
                    feats.setdefault(key, 0)

            if tar_bytes:
                feats.update(source_file_features(tar_bytes))
            else:
                feats.update(
                    {
                        "package_file_count": 0,
                        "js_file_count": 0,
                        "json_file_count": 0,
                        "minified_js_count": 0,
                        "total_source_bytes": 0,
                    }
                )

            output_rows.append(feats)

    fieldnames = sorted({k for row in output_rows for k in row})
    preferred = ["package_key", "package_name", "version", "label"]
    fieldnames = preferred + [f for f in fieldnames if f not in preferred]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "rows": len(output_rows),
        "labels": Counter(row["label"] for row in output_rows),
        "features": len(fieldnames) - len(preferred),
        "packages_with_package_json": sum(int(row.get("has_package_json", 0)) for row in output_rows),
        "output": str(out_csv),
    }
    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_benign_manifest(manifest_path):
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("tgz_path"):
                mapping[row["package_key"]] = row["tgz_path"]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="data/processed/npmstudy_index/package_index.csv")
    parser.add_argument("--malware-zip", default="data/raw/npmstudy/NPMStudy_large_65055888.zip")
    parser.add_argument("--out-csv", default="data/processed/features/base_feature_table.csv")
    parser.add_argument("--benign-manifest", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_features.benign_tgz_map = load_benign_manifest(args.benign_manifest)
    build_features(Path(args.index_csv), Path(args.malware_zip), Path(args.out_csv), args.limit)


if __name__ == "__main__":
    main()

