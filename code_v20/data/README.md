# Data provenance

The pre-extracted feature tables in `features/` are enough to reproduce every
experiment in the paper. This file documents how to re-derive them from the raw
signals.

## Raw archives (not included in this repository)

Total size is about 2.5 GB, so the raw archives are distributed by their
original hosts and are not committed here.

| Dataset | Raw file(s) | Source | SHA-256 | Size |
|---|---|---|---|---|
| NASA IMS | `4. Bearings.zip` | `phm-datasets.s3.amazonaws.com/NASA/` (PCoE mirror) | `21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83` | 1,075,597,174 B |
| FEMTO / PRONOSTIA | `10. FEMTO Bearing.zip` | `phm-datasets.s3.amazonaws.com/NASA/` (PCoE mirror) | `e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f` | 1,157,035,288 B |
| XJTU-SY | `XJTU-SY_Bearing_Datasets.part01..06.rar` | official XJTU-SY release | see table below | ≈4.4 GB |

The IMS and FEMTO archives are mirrored on
`phm-datasets.s3.amazonaws.com/NASA/`, the same mirror used by the
`VictorBauler/BearingDatasets` library (MIT).

### XJTU-SY part checksums

| File | SHA-256 |
|---|---|
| `XJTU-SY_Bearing_Datasets.part01.rar` | `c500657353f089a4ab50212ff4ddfc7b982729e92e2b127440c8ad881d80b968` |
| `XJTU-SY_Bearing_Datasets.part02.rar` | `4dfa6286a8e9c7cec1e925b347642349f36e27ab3f703c90d1d08977b7b4f61f` |
| `XJTU-SY_Bearing_Datasets.part03.rar` | `6929f531284f79e0209246b4ee23b23dd8d4faac1c0c3ff8fb131cda4a18bd3a` |
| `XJTU-SY_Bearing_Datasets.part04.rar` | `2245825acd29bed3b95e7a77879a3fbadf20f4ad4f99486384c714174621597d` |
| `XJTU-SY_Bearing_Datasets.part05.rar` | `e1b0a41a32b865e48ea7981c56f952a960d94335f3496b72eb4a65932f1671bd` |
| `XJTU-SY_Bearing_Datasets.part06.rar` | `df1854821a9d481104476379f7bc045ea1e101427c6e36145cd7677dc7c4a684` |

## Archive nesting and extraction tools

The raw archives are nested; two local binaries were built from source (no root):

- `tools/7z/7z` — p7zip 16.02 (handles 7z + rar + zip)
- `tools/unrar` — unrar 7.00 (RARLAB freeware source)

Observed nesting (from zip local file headers):

- IMS `4. Bearings.zip` → `4. Bearings/IMS.7z` → `IMS/1st_test.rar` etc. → tab-separated snapshots
- FEMTO `10. FEMTO Bearing.zip` → `10. FEMTO Bearing/FEMTOBearingDataSet.zip` → `Training_set/Learning_set/...`, `Validation_Set/Full_Test_Set/...`
- XJTU-SY is an extracted folder (no archive handling beyond the six RAR parts)

## Feature definitions

The per-snapshot features are defined identically to the label-free
early-warning pipeline (Bhavjeet Singh Hora, FIU; MIT license):

- `rms = sqrt(mean(s^2))`
- `kurtosis_pearson = scipy.stats.kurtosis(s, fisher=False)`
- `skewness = scipy.stats.skew(s)`
- `variance = np.var(s)`
- `peak2peak = max(s) - min(s)`

## Re-deriving the feature tables

Place the raw archives under `data/` as `raw_ims/`, `raw_femto/`, and
`raw_xjtu/`, build the extraction tools under `tools/`, then run:

```bash
python src/run_extraction_pipeline.py --rerun
```

The pipeline writes `data/features/*.csv` and (optionally) re-runs the benchmark
scripts. Feature tables are verified row-by-row against the reference tables to
machine precision (`max abs error ≈ 1e-14`).

