#!/usr/bin/env python3
"""Extract the raw archives and re-derive the feature tables from raw signals.

Archive nesting discovered from the local file headers:
  - IMS   `4. Bearings.zip`          -> `4. Bearings/IMS.7z` -> `IMS/*.rar` -> txt snapshots
  - FEMTO `10. FEMTO Bearing.zip`    -> `10. FEMTO Bearing/FEMTOBearingDataSet.zip` -> bearing CSVs
  - XJTU  provided by the user as a folder (no archive handling here)

Requires the locally built `downloads/tools/7z/7z` and `downloads/tools/unrar`.
Usage:
    python3 full_supplement/raw_features/run_extraction_pipeline.py [--rerun]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SZ = str(TOOLS / "7z" / "7z")
UNRAR = str(TOOLS / "unrar")
WORK = ROOT / "data" / "raw"
FEAT = ROOT / "data" / "features"
PY = sys.executable


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_ims():
    z = ROOT / "data" / "raw_ims" / "IMS.zip"
    if not z.exists():
        print("skip IMS: no", z)
        return
    d = WORK / "ims"
    d.mkdir(parents=True, exist_ok=True)
    if not list(d.rglob("1st_test")):
        run([SZ, "x", "-y", str(z), f"-o{d}"])
        sevens = list(d.rglob("IMS.7z"))
        if sevens:
            run([SZ, "x", "-y", str(sevens[0]), f"-o{d}"])
        for rar in sorted(d.rglob("*.rar")):
            run([SZ, "x", "-y", str(rar), f"-o{rar.parent}"])
    root = None
    for p in d.rglob("1st_test"):
        if p.is_dir():
            root = p.parent
            break
    if root is None:
        raise RuntimeError("could not find IMS test directories")
    run([PY, str(ROOT / "src/extract_ims.py"), str(root), str(FEAT)])


def extract_femto():
    z = ROOT / "data" / "raw_femto" / "FEMTO.zip"
    if not z.exists():
        print("skip FEMTO: no", z)
        return
    d = WORK / "femto"
    d.mkdir(parents=True, exist_ok=True)
    if not list(d.rglob("Full_Test_Set")) and not list(d.rglob("Learning_set")):
        run(["unzip", "-q", "-o", str(z), "-d", str(d)])
        inner = next(d.rglob("FEMTOBearingDataSet.zip"), None)
        if inner:
            run(["unzip", "-q", "-o", str(inner), "-d", str(inner.parent)])
        for sub in ("Training_set.zip", "Validation_Set.zip"):
            p = next(d.rglob(sub), None)
            if p:
                run(["unzip", "-q", "-o", str(p), "-d", str(p.parent)])
    root = d
    out = FEAT / "femto_features.csv"
    if out.exists():
        print("skip FEMTO feature extraction:", out, "exists")
        return
    run([PY, str(ROOT / "src/extract_femto.py"), str(root), str(out)])


def extract_xjtu():
    src = ROOT / "data" / "raw_xjtu"
    if not src.exists():
        print("skip XJTU: no", src)
        return
    out = FEAT / "xjtu_features.csv"
    if out.exists():
        print("skip XJTU feature extraction:", out, "exists")
        return
    run([PY, str(ROOT / "src/extract_xjtu.py"), str(src), str(out)])


def rerun_benchmarks():
    env = dict(os.environ, RTF_DATA_DIR=str(FEAT))
    for script in (
        "run_to_failure_ood.py",
        "run_to_failure_failure_vs_drift.py",
        "run_to_failure_early_warning.py",
    ):
        print("\n=== " + script + " ===", flush=True)
        subprocess.run([PY, str(ROOT / "experiments" / script)], env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="also run the three benchmark scripts")
    args = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    extract_ims()
    extract_femto()
    extract_xjtu()
    if (FEAT / "test1.csv").exists():
        run([PY, str(ROOT / "src/compare_features.py"), str(FEAT), str(ROOT / "data" / "hora_reference")])
    if args.rerun:
        rerun_benchmarks()


if __name__ == "__main__":
    main()
