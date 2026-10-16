import argparse
import csv
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def registry_name(package_name):
    if package_name.startswith("@") and "/" in package_name:
        scope, name = package_name.split("/", 1)
        return f"{scope}%2F{name}"
    return urllib.parse.quote(package_name, safe="")


def safe_name(package_name):
    return package_name.replace("/", "##")


def request_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "ConfuCross-MPD/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def download_file(url, out_path, timeout=120):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ConfuCross-MPD/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(out_path)


def retry_call(fn, retries, sleep_sec):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(max(sleep_sec, 0.5) * attempt)
    raise last_exc


def has_package_json(tgz_path):
    try:
        with tarfile.open(tgz_path, "r:gz") as tf:
            return any(m.isfile() and m.name.endswith("/package.json") for m in tf.getmembers())
    except Exception:
        return False


def fetch_one(package_name, version, out_root, sleep_sec, retries):
    encoded = registry_name(package_name)
    metadata_url = f"https://registry.npmjs.org/{encoded}"
    package_dir = out_root / safe_name(package_name) / version
    metadata_path = package_dir / "metadata.json"
    tgz_path = package_dir / f"{safe_name(package_name)}-{version}.tgz"

    result = {
        "package_name": package_name,
        "version": version,
        "status": "unknown",
        "metadata_path": str(metadata_path),
        "tgz_path": str(tgz_path),
        "tarball_url": "",
        "error": "",
        "has_package_json": 0,
    }

    try:
        package_dir.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = retry_call(lambda: request_json(metadata_url), retries, sleep_sec)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(sleep_sec)

        versions = metadata.get("versions", {})
        version_meta = versions.get(version)
        if not version_meta:
            result["status"] = "missing_version"
            return result

        tarball_url = version_meta.get("dist", {}).get("tarball", "")
        result["tarball_url"] = tarball_url
        if not tarball_url:
            result["status"] = "missing_tarball"
            return result

        if not tgz_path.exists() or tgz_path.stat().st_size == 0:
            retry_call(lambda: download_file(tarball_url, tgz_path), retries, sleep_sec)
            time.sleep(sleep_sec)

        result["has_package_json"] = int(has_package_json(tgz_path))
        result["status"] = "ok" if result["has_package_json"] else "invalid_tgz"
        return result
    except urllib.error.HTTPError as exc:
        result["status"] = "http_error"
        result["error"] = f"{exc.code} {exc.reason}"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
        return result


def load_benign_rows(index_csv, limit):
    rows = []
    with open(index_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] != "0":
                continue
            rows.append({"package_name": row["package_name"], "version": row["version"], "package_key": row["package_key"]})
            if limit and len(rows) >= limit:
                break
    return rows


def load_benign_rows_random(index_csv, limit, random_state):
    import random

    rows = load_benign_rows(index_csv, None)
    rng = random.Random(random_state)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="data/processed/npmstudy_index/package_index.csv")
    parser.add_argument("--out-dir", default="data/raw/npmstudy/benign_registry_pilot")
    parser.add_argument("--manifest", default="data/processed/npmstudy_index/benign_registry_pilot_manifest.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_mode == "random":
        rows = load_benign_rows_random(args.index_csv, args.limit, args.random_state)
    else:
        rows = load_benign_rows(args.index_csv, args.limit)
    order = {row["package_key"]: i for i, row in enumerate(rows)}

    results = []
    if args.workers <= 1:
        for i, row in enumerate(rows, start=1):
            print(f"[{i}/{len(rows)}] {row['package_key']}", flush=True)
            result = fetch_one(row["package_name"], row["version"], out_root, args.sleep, args.retries)
            result["package_key"] = row["package_key"]
            results.append(result)
            print(f"  -> {result['status']}", flush=True)
    else:
        print(f"Fetching {len(rows)} packages with {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_one,
                    row["package_name"],
                    row["version"],
                    out_root,
                    args.sleep,
                    args.retries,
                ): row
                for row in rows
            }
            completed = 0
            for future in as_completed(futures):
                row = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "package_name": row["package_name"],
                        "version": row["version"],
                        "status": "error",
                        "metadata_path": "",
                        "tgz_path": "",
                        "tarball_url": "",
                        "error": repr(exc),
                        "has_package_json": 0,
                    }
                result["package_key"] = row["package_key"]
                results.append(result)
                print(f"[{completed}/{len(rows)}] {row['package_key']} -> {result['status']}", flush=True)

        results.sort(key=lambda item: order.get(item["package_key"], 10**12))

    fieldnames = [
        "package_key",
        "package_name",
        "version",
        "status",
        "has_package_json",
        "metadata_path",
        "tgz_path",
        "tarball_url",
        "error",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "requested": len(rows),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] != "ok"),
        "manifest": str(manifest_path),
        "out_dir": str(out_root),
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

