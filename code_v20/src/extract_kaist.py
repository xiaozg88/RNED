#!/usr/bin/env python3
"""Extract per-snapshot x/y vibration RMS from the KAIST CSV archive."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEVENZIP = Path(__file__).resolve().parents[2] / "tools/p7zip-tmp/usr/lib/p7zip/7z"
FILENAME_RE = re.compile(r"^LogFile_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.csv$")
FIELDS = (
    "dataset",
    "group",
    "bearing",
    "snapshot",
    "file",
    "axis",
    "rms",
    "n_rows",
    "n_valid_rows",
    "n_invalid_rows",
    "sampling_frequency_hz",
)


def archive_filenames(archive: Path) -> list[str]:
    listing = subprocess.check_output(
        [str(SEVENZIP), "l", "-ba", "-slt", str(archive)], text=True
    )
    filenames = [
        line[len("Path = "):]
        for line in listing.splitlines()
        if line.startswith("Path = ")
    ]
    return sorted(name for name in filenames if FILENAME_RE.match(name))


def extract_file(
    archive: Path, filename: str
) -> tuple[str, int, int, float, float]:
    awk_program = r"""
BEGIN {
    n = 0; bad = 0; sx = 0; sy = 0
}
NF != 4 || $1 ~ /^(NaN|nan|-nan)$/ || $2 ~ /^(NaN|nan|-nan)$/ {
    bad++
    next
}
{
    x = $1 + 0; y = $2 + 0
    sx += x * x; sy += y * y; n++
}
END {
    if (n == 0) exit 2
    printf "%.17g\t%.17g\t%.17g\t%.17g\n", n, bad, sqrt(sx / n), sqrt(sy / n)
}
"""
    extractor = subprocess.Popen(
        [str(SEVENZIP), "e", "-so", str(archive), filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    reducer = subprocess.Popen(
        ["awk", "-F,", awk_program],
        stdin=extractor.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert extractor.stdout is not None
    extractor.stdout.close()
    stdout, reducer_stderr = reducer.communicate()
    extractor_stderr = extractor.stderr.read() if extractor.stderr else b""
    extractor_returncode = extractor.wait()
    if extractor.stderr:
        extractor.stderr.close()
    if extractor_returncode != 0 or reducer.returncode != 0:
        raise RuntimeError(
            f"failed to extract {filename}: "
            f"7z={extractor_returncode}, awk={reducer.returncode}, "
            f"stderr={extractor_stderr!r}/{reducer_stderr!r}"
        )
    n_rows, n_bad, rms_x, rms_y = stdout.strip().split("\t")
    return filename, int(n_rows), int(n_bad), float(rms_x), float(rms_y)


def extract(archive: Path, output: Path, dataset: str, workers: int) -> None:
    filenames = archive_filenames(archive)
    if not filenames:
        raise ValueError(f"no KAIST hourly CSV entries found in {archive}")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_file, archive, filename): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            filename, n_rows, n_bad, rms_x, rms_y = future.result()
            snapshot = len(rows) + 1
            for axis, rms in (("X", rms_x), ("Y", rms_y)):
                rows.append(
                    {
                        "dataset": dataset,
                        "group": "run_to_failure",
                        "bearing": "B1",
                        "snapshot": snapshot,
                        "file": filename,
                        "axis": axis,
                        "rms": rms,
                        "n_rows": n_rows,
                        "n_valid_rows": n_rows,
                        "n_invalid_rows": n_bad,
                        "sampling_frequency_hz": 25600.0,
                    }
                )

    rows.sort(key=lambda row: row["file"])
    for snapshot, row in enumerate(rows, start=1):
        row["snapshot"] = (snapshot + 1) // 2

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows, 1 bearing to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="kaist")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    extract(args.archive, args.output, args.dataset, args.workers)


if __name__ == "__main__":
    main()
