#!/usr/bin/env python3
"""Audit EOSC multimodal bearing archives without downloading full files."""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "external_candidates" / "datarepo_eosc.html"
RESULT_PATH = ROOT / "results" / "multimodal54_acquisition_manifest.json"
BASE_URL = "https://datarepo.eosc.cz/api/records/hjbxn-0yc34/files"
USER_AGENT = "RNED-preregistration-audit/1.0 (Range-only ZIP metadata audit)"


def fetch_range(url: str, start: int, end: int, attempts: int = 6) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"invalid range {start}-{end}")
    last_error = None
    for attempt in range(attempts):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--max-time",
                    "180",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "--user-agent",
                    USER_AGENT,
                    "--range",
                    f"{start}-{end}",
                    url,
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            data = completed.stdout
            expected = end - start + 1
            if len(data) != expected:
                raise RuntimeError(f"got {len(data)} bytes, expected {expected}")
            return data
        except Exception as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"failed range {start}-{end}: {last_error}")


def unpack_eocd(tail: bytes, url: str | None = None) -> dict:
    signature = b"PK\x05\x06"
    positions = [index for index in range(len(tail) - 21) if tail[index : index + 4] == signature]
    if not positions:
        raise ValueError("ZIP EOCD not found in tail")
    eocd_offset = positions[-1]
    fields = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    result = {
        "central_directory_size": fields[5],
        "central_directory_offset": fields[6],
    }

    locator_signature = b"PK\x06\x07"
    locator_positions = [
        index
        for index in range(max(0, eocd_offset - 40), min(len(tail), eocd_offset) - 15)
        if tail[index : index + 4] == locator_signature
    ]
    if result["central_directory_size"] == 0xFFFFFFFF or locator_positions:
        if not locator_positions:
            raise ValueError("ZIP64 central directory detected but locator is missing")
        locator = struct.unpack_from("<4sLQL", tail, locator_positions[-1])
        eocd64_file_offset = locator[2]
        if url is not None:
            eocd64 = fetch_range(url, eocd64_file_offset, eocd64_file_offset + 55)
        else:
            raise ValueError("ZIP64 EOCD requires archive URL")
        fields64 = struct.unpack_from("<4sQ2H2L4Q", eocd64, 0)
        result.update(
            {
                "central_directory_size": fields64[8],
                "central_directory_offset": fields64[9],
            }
        )
    return result


def parse_central_directory(data: bytes) -> list[dict]:
    entries = []
    offset = 0
    central_signature = b"PK\x01\x02"
    while offset + 46 <= len(data) and data[offset : offset + 4] == central_signature:
        fields = struct.unpack_from("<4s6H3L5H2L", data, offset)
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        local_header_offset = fields[16]
        extra_start = offset + 46 + filename_length
        filename = data[offset + 46 : extra_start].decode("cp437")
        extra = data[extra_start : extra_start + extra_length]

        if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
            extra_offset = 0
            while extra_offset + 4 <= len(extra):
                header = struct.unpack_from("<HH", extra, extra_offset)
                if header[0] == 0x0001 and header[1] >= 24:
                    values = struct.unpack_from("<QQQ", extra, extra_offset + 4)
                    if uncompressed_size == 0xFFFFFFFF:
                        uncompressed_size = values[0]
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = values[1]
                    if local_header_offset == 0xFFFFFFFF:
                        local_header_offset = values[2]
                    break
                extra_offset += 4 + header[1]

        entries.append(
            {
                "filename": filename,
                "flags": fields[2],
                "compression": fields[4],
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
            }
        )
        offset += 46 + filename_length + extra_length + comment_length
    if offset != len(data):
        raise ValueError(f"unparsed central-directory bytes: {len(data) - offset}")
    return entries


def read_zip_member(url: str, entry: dict) -> bytes:
    if entry["flags"] & 0x0001:
        raise RuntimeError(f"encrypted member: {entry['filename']}")
    local = fetch_range(url, entry["local_header_offset"], entry["local_header_offset"] + 29)
    fields = struct.unpack_from("<4s5H3L2H", local, 0)
    filename_length = fields[9]
    extra_length = fields[10]
    data_offset = entry["local_header_offset"] + 30 + filename_length + extra_length
    compressed = fetch_range(
        url, data_offset, data_offset + entry["compressed_size"] - 1
    )
    if entry["compression"] == 0:
        return compressed
    if entry["compression"] == 8:
        return zlib.decompress(compressed, -15)
    raise RuntimeError(f"unsupported compression {entry['compression']}")


def parse_readme(text: str) -> dict:
    patterns = {
        "test_duration": r"^Test duration:\s*(.+)$",
        "stop_condition": r"^Stop condition:\s*(.+)$",
        "final_rms": r"rms exceeded\s*\(([0-9.]+)\s*m/s2\)",
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        values[name] = match.group(1).strip() if match else None
        if name == "final_rms" and match:
            values[name] = float(match.group(1))
    return values


def audit_archive(entry: dict, tail_size: int) -> dict:
    filename = entry["key"]
    url = entry["links"]["content"]
    archive_size = int(entry["size"])
    tail_start = max(0, archive_size - tail_size)
    tail = fetch_range(url, tail_start, archive_size - 1)
    eocd = unpack_eocd(tail, url=url)
    cd_size = int(eocd["central_directory_size"])
    cd_offset = int(eocd["central_directory_offset"])
    cd_end = cd_offset + cd_size - 1
    tail_end = archive_size - 1
    if cd_offset < tail_start or cd_end > tail_end:
        central = fetch_range(url, cd_offset, cd_end)
    else:
        cd_start_index = cd_offset - tail_start
        central = tail[cd_start_index : cd_start_index + cd_size]
    members = parse_central_directory(central)
    tdms_members = [member for member in members if member["filename"].lower().endswith(".tdms")]
    readmes = [member for member in members if member["filename"].lower().endswith("readme.txt")]
    if len(readmes) != 1:
        raise RuntimeError(f"expected one readme, found {len(readmes)}")
    readme_text = read_zip_member(url, readmes[0]).decode("utf-8", errors="replace")
    readme = parse_readme(readme_text)
    n_snapshots = len(tdms_members)
    eligible_failure = n_snapshots >= 90
    stop_condition = (readme.get("stop_condition") or "").lower()
    stop_condition_type = (
        "rms"
        if "rms exceeded" in stop_condition
        else "iso"
        if "iso exceeded" in stop_condition
        else "other"
        if stop_condition
        else "unknown"
    )
    result = {
        "archive": filename,
        "url": url,
        "declared_size_bytes": archive_size,
        "declared_md5": entry["checksum"].split(":", 1)[-1],
        "audit_method": "HTTP Range ZIP central directory + readme member",
        "n_central_entries": len(members),
        "n_tdms_snapshots": n_snapshots,
        "readme": readme,
        "stop_condition_type": stop_condition_type,
        "vibration_channel_evidence": "sample archive has one accelerometer TDMS channel; no per-archive multimodal channel audit",
        "inferred_vibration_axes": 1,
        "failure_protocol_eligible": bool(eligible_failure),
        "cross_axis_protocol_eligible": False,
        "healthy_censored_trajectory": False,
        "overall_eligibility": "historical_layer_supplement_only" if eligible_failure else "ineligible",
        "eligibility_reasons": (
            ["at least 90 snapshots"]
            if eligible_failure
            else [f"only {n_snapshots} snapshots (<90)"]
        )
        + ["single vibration channel", "run-to-failure stop, not an administratively censored healthy trajectory"],
    }
    return result


def load_manifest() -> dict:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text())
    return {
        "dataset": "Open multimodal dataset for rolling bearing degradation",
        "source_metadata_version": None,
        "audit_status": "incomplete",
        "archives": {},
        "summary": None,
        "generated_at_utc": None,
    }


def summarize(archives: dict) -> dict:
    rows = list(archives.values())
    successful = [row for row in rows if row.get("audit_status") == "ok"]
    snapshots = [row["n_tdms_snapshots"] for row in successful]
    return {
        "n_archives_total": len(rows),
        "n_archives_processed": len(rows),
        "n_archives_audited": len(successful),
        "n_errors": len(rows) - len(successful),
        "n_with_at_least_90_snapshots": sum(value >= 90 for value in snapshots),
        "snapshot_min": min(snapshots) if snapshots else None,
        "snapshot_median": sorted(snapshots)[len(snapshots) // 2] if snapshots else None,
        "snapshot_max": max(snapshots) if snapshots else None,
        "n_non_rms_stop_conditions": sum(
            row["readme"].get("stop_condition") is not None
            and row.get("stop_condition_type") not in ("rms", "unknown")
            for row in successful
        ),
        "n_iso_stop_conditions": sum(
            row.get("stop_condition_type") == "iso" for row in successful
        ),
        "n_multiaxis_or_healthy_censored": 0,
        "main_validation_eligible": False,
        "decision": "supplement-only: insufficient snapshots on sampled archive; single-axis design; no healthy censored fleet",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-size", type=int, default=131_072)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(METADATA_PATH.read_text())
    manifest = load_manifest()
    manifest["source_metadata_version"] = metadata["metadata"]["version"]
    entries = sorted(metadata["files"]["entries"].values(), key=lambda item: item["key"])
    processed = 0
    for entry in entries:
        key = entry["key"]
        previous_status = manifest["archives"].get(key, {}).get("audit_status")
        if previous_status == "ok" or (previous_status == "error" and not args.retry_errors):
            continue
        if args.limit is not None and processed >= args.limit:
            break
        print(f"auditing {key} ...", flush=True)
        try:
            row = audit_archive(entry, args.tail_size)
            row["audit_status"] = "ok"
        except Exception as error:
            row = {
                "archive": key,
                "url": entry["links"]["content"],
                "declared_size_bytes": int(entry["size"]),
                "declared_md5": entry["checksum"].split(":", 1)[-1],
                "audit_status": "error",
                "error": str(error),
            }
        manifest["archives"][key] = row
        manifest["audit_status"] = "incomplete"
        manifest["summary"] = summarize(manifest["archives"])
        manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        RESULT_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        processed += 1
        time.sleep(args.sleep_seconds)

    remaining = [key for key in [entry["key"] for entry in entries] if key not in manifest["archives"]]
    manifest["audit_status"] = "complete" if not remaining else "incomplete"
    manifest["summary"] = summarize(manifest["archives"])
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
