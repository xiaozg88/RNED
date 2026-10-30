# RNED — Rate-Normalized Excess Early-Warning Detector

Official code for the paper:

> **Budget-Aware Event-Level Rate-Normalized Early Warning for Bearings in
> Industrial IoT: Validation in a Virtual IoT Environment**
> *(submitted to IEEE Internet of Things Journal)*

RNED is a label-free, causal, fixed-threshold, edge-deployable early-warning
detector for run-to-failure bearings. It reduces the failure-versus-drift
discriminative signal to the **cumulative excess** of RMS above a $1.2\times$
early-life baseline, and its online form is the **rate-normalized (time-averaged)
excess** compared against a single fixed threshold. On three public
run-to-failure datasets (NASA IMS, XJTU-SY, FEMTO/PRONOSTIA; 44 bearings) it
detects **31 of 34 failed bearings** at the calibrated zero-false-alarm operating
point, with median lead times of **14.5% / 46.3% / 36.0%** of life, in 0.03235 ms
and 0.0383 mJ per window without an accelerator.

## Overview

- **Problem.** Early failure warning under a zero-false-alarm constraint must
  separate *failure-bound degradation* from *benign drift*, not merely detect
  deviation from a healthy baseline.
- **Mechanism audit.** $E_{\mathrm{cum}}=\sum_{i=61}^{N}e_i$ with
  $e_i=\max(r_i-c,0)$, $r_i=f_i/b$, $c=1.2$ — fully separates IMS failed vs.
  censored bearings (AUROC 100, exact permutation $p=0.00202$).
- **Online detector.** $R_t=S_t/(t-60)$, $S_t=S_{t-1}+e_t$, latching alarm
  $\tau=\min\{t\ge90:R_t>T\}$, threshold calibrated once on the censored set.
- **Theory.** Two qualitative propositions are retained: unbounded positive-drift
  accumulators cannot use a fixed zero-false-alarm threshold indefinitely, while
  the rate has duration-invariant mean and first-order scale equivariance. They
  do not provide a numeric finite-population false-alarm guarantee.

## Manuscript

The complete audited Chinese source and the English v20 translation are generated
from version-controlled scripts:

```bash
python paper/build_v20_full_manuscript.py
python paper/build_v20_full_english.py
python paper/md_to_docx.py paper/manuscript_zh_v20_full.md paper/正文_中文_v20.docx
```

Outputs are `paper/manuscript_zh_v20_full.md`, `paper/manuscript_en_v20_full.tex`,
`paper/references_v20.bib`, and the regenerated Word manuscript. The English LaTeX
file uses `references_v20.bib`, which contains exactly the 31 retained references.

## Quick start

The pre-extracted feature tables ship with this repo, so **no raw-data download
is needed** to reproduce every number in the paper.

```bash
python -m venv .venv && source .venv/bin/activate   # or use your own env
pip install -r requirements.txt

# Headline result: 31/34 detection, lead times, zero-FA threshold
python experiments/run_to_failure_early_warning.py
```

This writes `results/results_run_to_failure_early_warning.json` with
`n_failed=34`, `n_censored=8`, `zero_fa_threshold ≈ 0.0424`, and per-dataset
detection `4/4 (IMS)`, `13/13 (XJTU-SY)`, `14/17 (FEMTO)` (total **31/34**).

## Experiments ↔ paper

All scripts read features from `data/features/` and write results to `results/`.

| Script (`experiments/`) | Produces | Paper |
|---|---|---|
| `run_to_failure_ood.py` | AUROC / FPR95 per method & dataset + cross-condition rows | §4.1, Table III |
| `run_to_failure_failure_vs_drift.py` | $E_{\mathrm{cum}}$ separation, AUROC/permutation | §4.2 |
| `run_to_failure_early_warning.py` | 31/34, lead times, $c$-sweep | §4.3, headline |
| `run_to_failure_classical_baselines.py` | CUSUM/SPRT/Shewhart/EWMA comparison | §4.4, Table IV |
| `run_to_failure_detector_significance.py` | paired McNemar + Holm | §4.4 |
| `run_to_failure_feature_ablation.py` | per-feature detection & lead time | §4.5, Table V |
| `run_to_failure_protocol_sensitivity.py` | baseline/run-in/min-monitor sweeps | §4.6 |
| `run_to_failure_robustness.py` | noise & axis-dropout robustness | §4.6 |
| `run_to_failure_ae_baseline.py` | reconstruction-AE baseline | §4.1 |
| `run_to_failure_lstmae_baseline.py` | LSTM-AE baseline (needs `torch`) | §4.1 |
| `run_to_failure_edge_benchmark.py` | latency / energy on Jetson Orin NX | §4.6 |
| `run_to_failure_modern_baselines.py` | window-level MLP-AE / LSTM-AE / Deep SVDD online baselines (needs `torch`) | §4.4, App. B |
| `run_to_failure_ewma_sprt_sweep.py` | EWMA λ and SPRT prior parameter sweeps | §4.6, App. B |
| `run_to_failure_threshold_epsilon_sweep.py` | zero-FA threshold margin factor (1.01) sweep | §4.6, App. B |
| `run_to_failure_cross_dataset_fa_proxy.py` | cross-rig false-alarm proxy from early-life pseudo-health segments (point estimate + 95% CI) | §4.3, App. B |
| `run_to_failure_transformer_baselines.py` | causal Transformer-AE + TranAD online baselines (needs `torch`) | §4.4, App. B |
| `run_to_failure_rul_baseline.py` | supervised RUL-thresholded alarm baseline, in-sample (favorable to baseline; needs `torch`) | §4.4, App. B |
| `run_to_failure_fleet_fa_budget.py` | fleet-scale false-alarm budget (probability arithmetic) | §4.8 |
| `run_to_failure_normalized_accumulators.py` | normalized CUSUM/SPRT fairness check | §4.4, App. B |
| `run_to_failure_robustness_checks.py` | fleet bootstrap, FEMTO-miss analysis, mixing-coefficient check | §4.3/§4.8/§5.2 |

Figures are generated by the scripts in `figures/`:

```bash
python figures/gen_fig1_architecture.py
python figures/gen_fig1_two_problems.py     # -> fig2_two_problems.pdf
python figures/gen_fig2_failure_vs_drift.py # -> fig4_failure_vs_drift.pdf
python figures/gen_fig3_rate_vs_cumulative.py
python figures/gen_fig5_operating_curve.py
python figures/gen_fig6_negative_transfer.py # -> fig3_negative_transfer.pdf
python figures/gen_fig7_rate_length_invariance.py
python figures/gen_fig8_fleet_false_alarm.py
```

Pre-rendered PDFs of all seven figures are included in `figures/`.

## Data

### Pre-extracted features (included)

`data/features/` contains the per-snapshot, per-axis feature tables
(`rms`, `kurtosis_pearson`, `skewness`, `variance`, `peak2peak`) for all three
datasets — self-extracted from the raw signals and verified against reference
tables to machine precision ($\approx10^{-14}$). These five CSVs (~16 MB) are
all you need to run the experiments.

| File | Dataset | Bearings |
|---|---|---|
| `test1.csv`, `test2.csv`, `test3.csv` | NASA IMS | 12 (4 failed + 8 censored) |
| `xjtu_features.csv` | XJTU-SY | 15 (all failed) |
| `femto_features.csv` | FEMTO/PRONOSTIA | 17 |

### Re-deriving features from raw signals (optional)

The extraction pipeline in `src/` re-derives these tables from the raw archives.
The raw archives are **not included** (≈2.5 GB total); download URLs and SHA-256
checksums are listed in [`data/README.md`](data/README.md). Extraction requires
building the local `7z` / `unrar` binaries (p7zip 16.02 and unrar 7.00 from
source; no root needed). After placing the raw archives under `data/`:

```bash
python src/run_extraction_pipeline.py --rerun
```

## Results (precomputed)

All paper results are precomputed in `results/` so they can be diffed against
the manuscript without re-running. The `experiments/` scripts are idempotent and
overwrite these files on each run.

## Requirements

- Python $\ge$3.9; `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.
- `torch` is only required for the two neural baselines
  (`run_to_failure_ae_baseline.py`, `run_to_failure_lstmae_baseline.py`).
- The core detector (`run_to_failure_early_warning.py`,
  `run_to_failure_classical_baselines.py`, etc.) needs only `numpy` + `pandas` + `scipy`.

Tested with Python 3.10 and torch 2.1 (CUDA optional). See `requirements.txt`.

## Repository layout

```text
RNED/
├── experiments/   # 12 experiment scripts (one per paper result)
├── src/           # feature-extraction pipeline (raw signals -> data/features)
├── data/features/ # pre-extracted feature tables (included)
├── figures/       # figure-generation scripts + rendered PDF/PNG figures
├── paper/         # v20 manuscript sources, builders, references, and Word output
├── results/       # precomputed JSON/CSV results
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@article{rned2026,
  title = {Budget-Aware Event-Level Rate-Normalized Early Warning
           for Bearings in Industrial IoT: Validation in a
           Virtual IoT Environment},
  author = {Xiao, Zhiguo and Guo, Yujing and Shao, Yanqi and Chen, Guangteng
            and Hou, Dingbo and Liu, Yuanyuan and Cao, Xinyao and Liu, Junli},
  journal = {IEEE Internet of Things Journal (submitted)},
  year = {2026},
  note = {Code: https://github.com/xiaozg88/RNED}
}
```

