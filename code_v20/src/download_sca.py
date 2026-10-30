#!/usr/bin/env python3
"""Download SCA normal-bearing trajectories and verify their checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DATASET_API = "https://data.mendeley.com/public-api/datasets/tdn96mkkpt"
USER_AGENT = "RNED-reproduction/1.0"
CHUNK_SIZE = 512 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_download_url(url: str) -> str:
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-I",
            "-L",
            "--max-time",
            "60",
            "-A",
            USER_AGENT,
            "-o",
            "/dev/null",
            "-w",
            "%{url_effective}",
            url,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def fetch_range(url: str, start: int, end: int, output: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-sS",
            "--retry",
            "5",
            "--max-time",
            "300",
            "-A",
            USER_AGENT,
            "-r",
            f"{start}-{end}",
            url,
            "-o",
            str(output),
        ],
        check=True,
    )


def download_verified_file(url: str, size: int, expected_sha256: str, path: Path) -> None:
    direct_url = resolve_download_url(url)
    ranges = [
        (start, min(start + CHUNK_SIZE, size) - 1)
        for start in range(0, size, CHUNK_SIZE)
    ]
    with tempfile.TemporaryDirectory(prefix="sca-download-") as temporary:
        temporary_path = Path(temporary)

        def fetch(indexed_range: tuple[int, int, int]) -> None:
            index, start, end = indexed_range
            part_path = temporary_path / f"{index:05d}.part"
            fetch_range(direct_url, start, end, part_path)

        indexed_ranges = [
            (index, start, end) for index, (start, end) in enumerate(ranges)
        ]
        max_workers = min(48, max(1, len(indexed_ranges)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(fetch, indexed_ranges))

        with path.open("wb") as output:
            for part_path in sorted(temporary_path.glob("*.part")):
                output.write(part_path.read_bytes())

    if path.stat().st_size != size:
        raise RuntimeError(f"{path}: expected {size} bytes, got {path.stat().st_size}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{path}: SHA-256 mismatch; expected {expected_sha256}, got {actual_sha256}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw_archives/sca",
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    api_request = urllib.request.Request(
        DATASET_API, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(api_request, timeout=60) as response:
        metadata = json.load(response)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item for item in metadata["files"] if item["filename"] == f"{args.split}.mat"]
    for item in selected:
        details = item["content_details"]
        output_path = args.output_dir / f"{item['folder_id']}_{args.split}.mat"
        if output_path.exists() and output_path.stat().st_size == details["size"]:
            if sha256_file(output_path) == details["sha256_hash"]:
                print(f"verified existing {output_path.name}")
                continue
        print(f"downloading {output_path.name} ({details['size']} bytes)", flush=True)
        download_verified_file(
            details["download_url"],
            details["size"],
            details["sha256_hash"],
            output_path,
        )
        print(f"verified {output_path.name}")


if __name__ == "__main__":
    main()
