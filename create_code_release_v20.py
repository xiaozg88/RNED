#!/usr/bin/env python3
"""Create the self-contained public-code directory for manuscript v20."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "code_v20"

FIGURES = [
    "fig1_architecture.pdf",
    "fig2_two_problems.pdf",
    "fig3_negative_transfer.pdf",
    "fig4_failure_vs_drift.pdf",
    "fig5_operating_curve.pdf",
    "fig6_rate_vs_cumulative.pdf",
    "fig7_rate_length_invariance.pdf",
    "fig8_fleet_false_alarm.pdf",
    "fig9_method_tradeoff.pdf",
    "fig10_ldm_upb_condition_sensitivity.pdf",
]

FIGURE_SCRIPTS = [
    "gen_fig1_architecture.py",
    "gen_fig1_two_problems.py",
    "gen_fig6_negative_transfer.py",
    "gen_fig2_failure_vs_drift.py",
    "gen_fig5_operating_curve.py",
    "gen_fig3_rate_vs_cumulative.py",
    "gen_fig7_rate_length_invariance.py",
    "gen_fig8_fleet_false_alarm.py",
    "gen_fig9_method_tradeoff.py",
    "gen_fig10_ldm_upb_condition_sensitivity.py",
    "paper_plot_style.py",
]

PAPER_FILES = [
    "manuscript_zh_v20_full.md",
    "manuscript_en_v20_full.tex",
    "references_v20.bib",
    "build_v20_full_manuscript.py",
    "build_v20_full_english.py",
    "md_to_docx.py",
    "math_omml.py",
    "PAPER_CLAIM_AUDIT_V20.md",
    "PAPER_CLAIM_AUDIT_V20.json",
]

ROOT_FILES = [
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "data/README.md",
    "experiments/run_v20_reproduction.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(relative_path: str) -> None:
    source = ROOT / relative_path
    destination = RELEASE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def collect_files() -> list[Path]:
    selected = [RELEASE / "README.md", RELEASE / "verify_release.py"]
    selected.extend(RELEASE / item for item in ROOT_FILES)
    selected.extend(sorted((RELEASE / "experiments").glob("*.py")))
    selected.extend(sorted((RELEASE / "src").glob("*.py")))
    selected.extend(sorted((RELEASE / "data" / "features").glob("*.csv")))
    selected.extend(sorted((RELEASE / "data" / "benchmark").glob("*")))
    selected.extend(sorted((RELEASE / "results").glob("*")))
    selected.extend(RELEASE / "figures" / name for name in FIGURES)
    selected.extend(RELEASE / "figures" / name for name in FIGURE_SCRIPTS)
    selected.extend(RELEASE / "paper" / name for name in PAPER_FILES)
    return sorted({path for path in selected if path.is_file()}, key=lambda path: path.as_posix())


def write_readme() -> None:
    text = """# BA-RNED v20 Public Code Release

This directory is the self-contained code and evidence release for the manuscript
**Budget-Aware Event-Level Rate-Normalized Early Warning for Bearings in
Industrial IoT: Validation in a Virtual IoT Environment** (v20).

It contains all derived feature tables, deterministic experiment scripts,
precomputed evidence files, v20 manuscript sources, figure generators, and
SHA-256 manifests needed to audit the reported analyses. Raw public datasets are
not redistributed; `data/README.md` records their source and acquisition paths.

## Quick Start

```bash
cd code_v20
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_v20_reproduction.py --list
python experiments/run_v20_reproduction.py core
```

The core chain reproduces the headline detector, mechanism audit, classical
baselines, significance tests, ablations, boundary analyses, D1 comparison,
fleet analysis, virtual IoT network, and the v20 claim audit. Optional deep
baselines require PyTorch and are run with `--level baselines`. The full chain,
including external and exploratory analyses, is `--level all`.

Hardware-specific latency and energy numbers in `results/results_run_to_failure_edge_benchmark.json`
were measured on a six-core Jetson Orin NX. Re-running that benchmark on another
machine updates the platform measurement but does not invalidate the frozen
Jetson evidence. The virtual network and detector experiments are deterministic.

## Evidence Map

| Manuscript area | Primary command | Evidence |
|---|---|---|
| Headline 31/34, lead times, 0/8 calibration | `python experiments/run_to_failure_early_warning.py` | `results/results_run_to_failure_early_warning.json` |
| Failure-vs-drift cumulative-excess mechanism | `python experiments/run_to_failure_failure_vs_drift.py` | `results/results_run_to_failure_failure_vs_drift.json` |
| Degradation OOD and cross-condition boundary | `python experiments/run_to_failure_ood.py` | `results/results_run_to_failure_ood.csv`, `results/results_run_to_failure_ood_summary.json` |
| Classical baselines and significance | `python experiments/run_to_failure_classical_baselines.py`; `python experiments/run_to_failure_detector_significance.py` | corresponding JSON files |
| Modern/Transformer/RUL baselines | `python experiments/run_v20_reproduction.py baselines` | corresponding JSON files |
| D1 endpoint versus causal semantics | `python experiments/run_sg1026_1_public_d1_rms_protocol_replication.py`; `python experiments/run_iotj_v20_risk_resolution_audit.py` | corresponding JSON files |
| Feature and mechanism ablations | `python experiments/run_to_failure_feature_ablation.py`; `python experiments/run_to_failure_rate_acceleration.py`; `python experiments/run_to_failure_cross_axis_persistence.py` | corresponding JSON files |
| Robustness and protocol sensitivity | `python experiments/run_v20_reproduction.py core` | robustness, sensitivity, sweep, and normalized-accumulator JSON files |
| Healthy-evidence ladder | `python experiments/run_to_failure_cwru_normal_stress.py`; `python experiments/run_to_failure_cross_dataset_fa_proxy.py`; `python experiments/run_iotj_v20_risk_resolution_audit.py` | corresponding JSON files |
| Fleet budget | `python experiments/run_to_failure_fleet_fa_budget.py` | `results/results_run_to_failure_fleet_fa_budget.json` |
| Virtual IoT loss/retry/deduplication | `python experiments/run_bcsy_iot_validation_simulation.py` | `results/results_bcsy_iot_validation_simulation.json` |
| v20 claim audit | `python experiments/run_iotj_v20_risk_resolution_audit.py` | `results/results_iotj_v20_risk_resolution_audit.json` |

## Directory Layout

```text
code_v20/
├── experiments/    deterministic analyses and one-command reproduction entry
├── src/            raw-data extraction and feature verification pipeline
├── data/features/  frozen derived feature tables used by the paper
├── data/benchmark/ small immutable benchmark windows
├── results/        frozen outputs for every included analysis
├── figures/        v20 figure PDFs and generation scripts
├── paper/          v20 LaTeX/Markdown sources and claim audit
├── MANIFEST.json   byte-level SHA-256 manifest
├── MANIFEST.md     human-readable manifest
└── requirements.txt
```

## Reproducibility Boundaries

- The primary IMS, XJTU-SY, and FEMTO/PRONOSTIA feature tables are included.
- External feature tables for CWRU, MCC5, LDM_UPB, KAIST, Ferrara E6, and UNSW
  analyses are included where they are used by v20 evidence scripts.
- Raw waveforms and proprietary archives are not redistributed.
- The synthetic multi-rig and network experiments are protocol stress tests,
  not real multi-node, radio-channel, broker, ADC, or field validation.
- The v20 claim audit labels the synthetic and hardware-specific evidence
  boundaries explicitly.

To rederive primary features from permitted raw archives, follow `data/README.md`
and run `python src/run_extraction_pipeline.py` where applicable. The included
frozen CSV files allow all shipped analyses to run without raw-data downloads.

## Integrity Check

```bash
python verify_release.py
```

The verifier checks every file recorded in `MANIFEST.json` and reports byte-level
SHA-256 mismatches. It does not execute experiments.
"""
    (RELEASE / "README.md").write_text(text, encoding="utf-8")


def write_verifier() -> None:
    text = '''#!/usr/bin/env python3
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
        print("\\n".join(failures))
        raise SystemExit(1)
    print(f"Verified {len(manifest['files'])} files against MANIFEST.json.")


if __name__ == "__main__":
    main()
'''
    path = RELEASE / "verify_release.py"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def write_manifest(files: list[Path]) -> None:
    rows = []
    for path in files:
        rows.append(
            {
                "path": path.relative_to(RELEASE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": "rned-code-release-v20",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_version": "20",
        "files": rows,
    }
    (RELEASE / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# v20 Code Release Manifest",
        "",
        f"- Files: {len(rows)}",
        f"- Generated UTC: {manifest['created_at_utc']}",
        "- Verification: `python verify_release.py`",
        "- Raw public datasets are not redistributed.",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| `{row['path']}` | {row['bytes']} | `{row['sha256']}` |" for row in rows)
    (RELEASE / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir()
    (RELEASE / "data" / "benchmark").mkdir(parents=True)

    for item in ROOT_FILES:
        copy_file(item)
    for source, destination in (
        ("experiments", "experiments"),
        ("src", "src"),
        ("data/features", "data/features"),
        ("results", "results"),
        ("figures", "figures"),
    ):
        shutil.copytree(
            ROOT / source,
            RELEASE / destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            dirs_exist_ok=True,
        )
    for name in PAPER_FILES:
        copy_file(f"paper/{name}")

    shutil.copy(ROOT / "data/benchmark/representative_raw_windows.npz", RELEASE / "data/benchmark/representative_raw_windows.npz")
    for path in sorted((RELEASE / "figures").iterdir()):
        if path.name not in set(FIGURES + FIGURE_SCRIPTS):
            if path.is_file():
                path.unlink()

    write_readme()
    write_verifier()
    write_manifest(collect_files())
    print(f"Wrote {RELEASE} with {len(collect_files())} audited files.")


if __name__ == "__main__":
    main()
