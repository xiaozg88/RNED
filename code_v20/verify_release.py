#!/usr/bin/env python3
"""Verify the byte-level integrity of the v20 code release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            failures.append(f"modified: {row['path']}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"Verified {len(manifest['files'])} files against MANIFEST.json.")


if __name__ == "__main__":
    main()
