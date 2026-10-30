# Exploratory mechanism- and value-path audit

Date: 2026-10-22

Status: **exploratory only**. These results do not upgrade the paper's main
method and must not be described as independent validation.

## Cohort and protocol

- Evaluated bearings: 42 total = 34 failed + 8 IMS censored.
- Two XJTU-SY failures were excluded by the existing minimum monitoring rule.
- Fixed recent window: 10 monitored snapshots.
- Alarm evaluation starts at monitored snapshot 30.
- IMS censored trajectories provide the calibration maxima; the 1.01 margin is
  retained only for comparability with the exploratory hierarchy.
- Early-life transfer proxy: the first 150 raw snapshots (90 monitored
  snapshots) of 27 XJTU-SY/FEMTO bearings that satisfy the minimum length.

## Path 1 — cross-axis event mechanism

| Rule | Detection | IMS FA | Median lead | Early-life proxy alarms |
|---|---:|---:|---:|---:|
| Historical rate | 31/34 | 0/8 | 28.82% | 6/27 |
| Cross-axis recent rate | 32/34 | 0/8 | 13.70% | 4/27 |
| Historical OR cross-axis | 34/34 | 0/8 | 28.64% | 6/27 |
| Fixed-weight composite | 33/34 | 0/8 | 21.80% | 6/27 |
| Acceleration only | 22/34 | 0/8 | 1.77% | 0/27 |
| Second-axis persistence only | 0/34 | 0/8 | undefined | 0/27 |
| Instantaneous irreversibility only | 0/34 | 0/8 | undefined | 0/27 |

The only useful detector-level extension remains historical OR cross-axis
recent rate. Adding persistence and instantaneous irreversibility to the OR
does not change any alarm time. The fixed-weight composite is worse.

The mechanism variables are nevertheless informative as trajectory summaries:

| Variable | Failed median | IMS-censored median | AUC | Exact MW p |
|---|---:|---:|---:|---:|
| Max historical rate | 0.390 | 0.0188 | 0.974 | 3.81e-7 |
| Max second-axis recent rate | 3.065 | 0 | 0.963 | 1.15e-6 |
| Second-axis persistence | 1 | 0 | 0.838 | 1.07e-3 |
| Final irreversibility | 1 | 0.011 | 0.825 | 1.77e-3 |
| Coactivation fraction | 0.110 | 0 | 0.857 | 5.36e-4 |
| First-active fraction | 0.212 | 0.894 | 0.759 | 1.19e-2 |
| Axis propagation delay | 27 | 924 | 0.750 | 1.43e-2 |

Interpretation: failed trajectories show earlier activation, stronger second-axis
corroboration, less reversal, and shorter axis-to-axis propagation than IMS
censored drift. These are full-trajectory post-hoc summaries, not early-precision
claims. See the 2026-10-23 re-audit below: axis availability and observation-time
leakage materially weaken the cross-axis interpretation.

## Path 2 — trajectory-level conformal capacity

- Available healthy trajectories: 8.
- A single-layer finite conformal threshold at total alpha 0.125 requires at
  least 7 calibration trajectories.
- A two-layer Bonferroni threshold at total alpha 0.20 requires at least 9 per
  layer, so it is undefined for the current cohort.
- At total alpha 0.10, 19 per layer are required.

Leave-one-censored-out empirical audit:

| Score | Held-out false alarms |
|---|---:|
| Historical rate | 1/8 |
| Cross-axis recent rate | 1/8 |
| Fixed-weight composite | 1/8 |
| Second-axis persistence | 0/8 |
| Instantaneous irreversibility | 0/8 |

Conclusion: the current IMS healthy cohort is insufficient for the joint finite
sample guarantee. The 1/8 held-out result for the useful layers also shows
healthy-trajectory heterogeneity.

## Path 3 — robust scale normalization

| Feature | Detection | IMS FA | Median lead | Early-life proxy alarms |
|---|---:|---:|---:|---:|
| Original ratio excess | 31/34 | 0/8 | 28.82% | 6/27 |
| Positive robust-z excess | 9/34 | 0/8 | 53.97% | 2/27 |
| Robust-z above the 1.2-ratio-equivalent threshold | 21/34 | 0/8 | 19.44% | 3/27 |

Robust noise-scale normalization reduces early-life proxy alarms, but it loses
10--22 detections. It is more plausible as a health-confidence or gating layer
than as a replacement detector.

## Path 4 — early-life value prediction

Only FEMTO had enough long trajectories for leave-one-bearing-out prediction.

| Observed age | N | MAE change vs median baseline |
|---:|---:|---:|
| 90 | 17 | -3.40% |
| 150 | 16 | -11.54% |
| 300 | 16 | +12.20% |
| 600 | 14 | -27.26% |

The mechanism ridge model is unstable. At age 600, several degradation
variables correlate negatively with remaining life (historical rate -0.449,
cross-axis -0.456, max ratio -0.451), which is mechanically plausible, but the
small-sample prediction gains do not support a prognostic claim.

## Path 5 — failure phenotypes

Unsupervised k-means produced one broad cluster of 29 trajectories containing
all three historical misses, a three-bearing high-amplitude FEMTO cluster, and
a two-bearing IMS cluster. It did not isolate a deployable new phenotype and
should remain descriptive.

## Overall recommendation

1. Keep historical rate as the paper's main method.
2. Use the cross-axis event-propagation statistics to strengthen the mechanism
   explanation, not as a new headline detector.
3. Report conformal capacity as a limitation: 8 healthy trajectories cannot
   support the two-layer finite-sample guarantee.
4. Do not add the fixed-weight composite, robust-z replacement, RUL ridge
   model, or phenotype clustering to the main manuscript without new data.
5. If independent multi-rig data become available, the already frozen
   historical OR second-axis recent-rate hierarchy is the only candidate worth
   one-shot validation.

## Re-audit added 2026-10-23

The initial mechanism summary was re-examined for axis-count confounding and
time leakage. The audit is reproducible with
`experiments/run_to_failure_mechanism_reaudit.py` and
`results/results_run_to_failure_mechanism_reaudit.json`.

### Axis-count confounding

At the fixed 90-monitored-snapshot horizon, the healthy cohort contains six
single-axis and only two multi-axis bearings. A second-axis statistic is zero by
construction for the six single-axis healthy bearings. Therefore the
unstratified full-profile second-axis AUC of 0.963 is not by itself evidence of
a transferable cross-axis mechanism.

Restricting the full trajectory comparison to two-axis observations gives
second-axis recent-rate AUC 1.000 (exact p = 0.00178), but the comparison has
only 32 failed versus 2 healthy bearings. This is suggestive, not conclusive.

### Fixed early-horizon audit

At monitored snapshot 90 (150 raw snapshots), among 31 eligible failed and 8
healthy trajectories:

| Variable | AUC | Exact MW p |
|---|---:|---:|
| Historical rate | 0.726 | 0.0262 |
| Cross-axis recent rate | 0.597 | 0.211 |
| Coactivation fraction | 0.613 | 0.173 |
| First-active fraction | 0.726 | 0.0262 |
| Max ratio | 0.879 | 0.000247 |

On the same-platform IMS subset, historical rate and cross-axis recent rate both
have AUC 0.500. Thus the apparent cross-axis mechanism is not a stable early
precursor on the current cohort; it mainly captures late-stage corroborated
activation. The XJTU-vs-IMS comparison shows cross-axis AUC 0.750, whereas
FEMTO-vs-IMS shows 0.529, indicating strong platform heterogeneity.

The historical-rate full-profile AUC is stable under leave-one-failure-out
influence (range 0.973--0.985), but this does not remove its post-hoc nature.

### Calibration recheck

Leave-one-censored-out calibration of the historical OR cross-axis hierarchy
produces 2/8 held-out false alarms, with or without the 1.01 threshold margin.
This reproduces the earlier audit and reinforces that the hierarchy is not ready
to be a primary method.

### Revised decision

- No further method-tuning experiments on the existing 42-bearing cohort are
  justified; they would mainly increase post-hoc selection risk.
- Internal resplits or repeated runs are not scientific revalidation because the
  rules and thresholds were developed on these rigs.
- The current paper should keep the historical-rate method as its only headline.
- Cross-axis propagation may be mentioned only as a post-hoc mechanism probe.
- A claim of a deployable cross-axis mechanism requires a new, independent,
  multi-rig, multi-axis, healthy-censored cohort and one-shot frozen validation.
