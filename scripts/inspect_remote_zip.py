import argparse
import json
import struct
import sys
import urllib.request


EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"


def request_range(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None or end is not None:
        left = "" if start is None else str(start)
        right = "" if end is None else str(end)
        req.add_header("Range", f"bytes={left}-{right}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, dict(resp.headers), resp.read()


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return dict(resp.headers)


def find_eocd(tail):
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD signature not found in downloaded tail")
    eocd = tail[idx : idx + 22]
    if len(eocd) < 22:
        raise RuntimeError("Incomplete EOCD record")
    fields = struct.unpack("<4s4H2LH", eocd)
    return {
        "disk_entries": fields[3],
        "total_entries": fields[4],
        "central_dir_size": fields[5],
        "central_dir_offset": fields[6],
        "comment_len": fields[7],
    }


def parse_central_directory(data, max_entries=None):
    names = []
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != CD_SIG:
            break
        header = data[pos : pos + 46]
        vals = struct.unpack("<4s6H3L5H2L", header)
        compressed_size = vals[8]
        uncompressed_size = vals[9]
        name_len = vals[10]
        extra_len = vals[11]
        comment_len = vals[12]
        local_header_offset = vals[16]
        name_start = pos + 46
        name_end = name_start + name_len
        raw_name = data[name_start:name_end]
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            name = raw_name.decode("cp437", errors="replace")
        names.append(
            {
                "name": name,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
            }
        )
        pos = name_end + extra_len + comment_len
        if max_entries and len(names) >= max_entries:
            break
    return names


def inspect(url, tail_bytes, max_entries):
    headers = head(url)
    size = int(headers.get("Content-Length", "0"))
    if size <= 0:
        raise RuntimeError("Could not determine remote file size")

    start = max(0, size - tail_bytes)
    status, range_headers, tail = request_range(url, start, size - 1)
    if status not in (200, 206):
        raise RuntimeError(f"Unexpected range status: {status}")
    eocd = find_eocd(tail)

    cd_start = eocd["central_dir_offset"]
    cd_end = cd_start + eocd["central_dir_size"] - 1
    status, _, cd = request_range(url, cd_start, cd_end)
    if status not in (200, 206):
        raise RuntimeError(f"Unexpected central-directory range status: {status}")

    entries = parse_central_directory(cd, max_entries=max_entries)
    return {
        "url": url,
        "remote_size": size,
        "accept_ranges": headers.get("Accept-Ranges", ""),
        "range_status": status,
        "eocd": eocd,
        "parsed_entries": len(entries),
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--tail-bytes", type=int, default=262144)
    parser.add_argument("--max-entries", type=int, default=200)
    args = parser.parse_args()
    result = inspect(args.url, args.tail_bytes, args.max_entries)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
