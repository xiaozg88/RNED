# Paper Claim Audit — IoTJ v20

**Date:** 2026-10-30  
**Auditor:** Same-family Codex executor  
**Status:** `provisional`  
**Primary evidence:** `results/results_iotj_v20_risk_resolution_audit.json`

## Verdict

**PASS_WITH_LIMITATIONS.** All 21 quantitative claims in the Chinese and English v20 manuscripts match the v20 evidence chain: 11 exact matches and 10 acceptable roundings. No numerical mismatch or unsupported quantitative claim was found.

## Risk Resolution Audit

| Submission risk | v20 resolution | Key evidence | Status |
|---|---|---|---|
| Healthy-censored evidence thin | Five-tier evidence ladder | 0/8, 1/8, 0/3, 6/27, 2/24 | substantially mitigated |
| D1 has higher endpoint count | Causal D1 ablation | D1-causal 34/34 failed but 8/8 healthy alarms | resolved |
| Not Holm-significant vs Shewhart/EWMA | Preserve correction plus Beta audit | Holm 0.0625; posterior mean 0.875 | calibrated reporting |
| No multi-node evidence | Virtual 100-machine network suite | 98--98.043% byte reduction; loss/retry behavior | substantially mitigated |
| Method too simple | Budget-aware node--gateway--fleet protocol | fleet quantile/capacity formulation | protocol-level novelty |

## Quantitative Checks

| Claim | Paper value | Evidence value | Status |
|---|---:|---:|---|
| Primary detection | 31/34 | 31/34 | exact |
| Primary CI | 76.3--98.1% | 76.322--98.142% | rounding |
| IMS calibration | 0/8 | 0/8 | exact |
| IMS LOEO | 1/8 | 1/8 | exact |
| CWRU normal | 0/3 | 0/3 | exact |
| Early-life proxy | 6/27 | 6/27 | exact |
| Virtual healthy | 2/24; 8.33% | 2/24; 8.3333% | rounding |
| D1 endpoint | 33/34; 2/8 episodes | same | exact |
| D1 causal | 34/34; 8/8; 40.1% | 34/34; 8/8; 40.0926% | rounding |
| Holm p | 0.0625 | 0.0625 | exact |
| Beta posterior | 0.875; 0.590--0.996 | 0.875; 0.590384--0.996390 | rounding |
| Edge | 0.03235 ms; 0.0383 mJ | same | exact |
| Network baseline | 98.043%; 88.59 ms | 98.043%; 88.5916 ms | rounding |
| 5% loss, 1 retry | 99.654%; 87.41 ms | 99.6542%; 87.4108 ms | rounding |
| 10% loss, 1 retry | 98.999%; 92.61 ms | 98.9988%; 92.6119 ms | rounding |
| 10% loss, 3 retries | 100%; 84.70 ms | same | exact |
| Acceptance score | >58% threshold | 0.6081894 | exact |

## Boundary Statements

- The virtual IoT environment is not a real MQTT/CoAP, wireless-channel, or broker deployment.
- The virtual multi-rig cohort is synthetic and is not a confirmatory independent validation.
- The paper does not claim a finite-population false-alarm guarantee.
- RNED is not claimed Holm-significant over Shewhart or EWMA.

Machine-readable report: `paper/PAPER_CLAIM_AUDIT_V20.json`.
