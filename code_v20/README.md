# BA-RNED v20 Public Code Release

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
