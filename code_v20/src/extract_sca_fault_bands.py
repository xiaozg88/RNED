#!/usr/bin/env python3
"""Extract speed-normalized envelope fault-band residuals from SCA MAT files."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend, hilbert


FIELDS = (
    "dataset",
    "split",
    "case",
    "axis",
    "snapshot",
    "timestamp",
    "label",
    "rpm",
    "sampling_frequency_hz",
    "n_samples",
    "bpfo_log_snr",
    "bpfi_log_snr",
    "bsf_log_snr",
    "ftf_log_snr",
    "fault_family_log_snr",
    "bpfo_multiband_log_snr",
    "bpfi_multiband_log_snr",
    "bsf_multiband_log_snr",
    "ftf_multiband_log_snr",
    "fault_family_multiband_log_snr",
    "fault_family_multiband_top2_mean",
    "fault_family_selected_band_hz",
    "envelope_kurtosis",
    "envelope_spectral_crest",
    "rms",
)

FAMILY_FIELDS = ("bpfo", "bpfi", "bsf", "ftf")


def finite(value: object) -> float:
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def band_mask(freqs: np.ndarray, center: float, width: float) -> np.ndarray:
    return (freqs >= center - width) & (freqs <= center + width)


def family_log_snr(
    spec: np.ndarray,
    freqs: np.ndarray,
    shaft_hz: float,
    multiplier: float,
) -> tuple[float, float]:
    fundamental = shaft_hz * multiplier
    width = max(0.35, 0.025 * fundamental)
    nyquist_limit = 0.45 * float(freqs[-1])
    harmonic_ratios: list[float] = []
    harmonic_energies: list[float] = []
    control_energies: list[float] = []

    for harmonic in range(1, 6):
        center = harmonic * fundamental
        if center + width >= nyquist_limit:
            break
        fault = spec[band_mask(freqs, center, width)]
        controls: list[np.ndarray] = []
        for offset in (-5.0, -3.0, 3.0, 5.0):
            control_center = center + offset * width
            if control_center - width > freqs[1] and control_center + width < nyquist_limit:
                controls.append(spec[band_mask(freqs, control_center, width)])
        if fault.size < 3 or not controls:
            continue
        control_energy = float(np.mean([np.sum(values) for values in controls]))
        fault_energy = float(np.sum(fault))
        if control_energy > 0 and fault_energy > 0:
            harmonic_ratios.append(math.log2(fault_energy / control_energy))
            harmonic_energies.append(fault_energy)
            control_energies.append(control_energy)

    if not harmonic_ratios:
        return float("nan"), float("nan")
    family_ratio = (
        math.log2(sum(harmonic_energies) / sum(control_energies))
        if sum(control_energies) > 0
        else float("nan")
    )
    return family_ratio, max(harmonic_ratios)


def snapshot_features(
    signal: np.ndarray,
    rpm: float,
    sampling_rate: float,
    frequencies: dict[str, float],
) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]
    if signal.size < 256:
        raise ValueError("short vibration snapshot")

    centered = detrend(signal, type="constant")
    envelope = np.abs(hilbert(centered))
    envelope = detrend(envelope, type="constant")
    window = np.hanning(envelope.size)
    spec = np.square(np.abs(np.fft.rfft(envelope * window)))
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / sampling_rate)
    spec[0] = 0.0
    shaft_hz = rpm / 60.0

    values: dict[str, float] = {}
    family_scores: list[float] = []
    for family, multiplier in frequencies.items():
        family_ratio, _ = family_log_snr(spec, freqs, shaft_hz, multiplier)
        values[f"{family}_log_snr"] = family_ratio
        if family != "ftf" and math.isfinite(family_ratio):
            family_scores.append(family_ratio)

    broad = spec[freqs >= max(2.0, float(freqs[1]))]
    values["fault_family_log_snr"] = max(family_scores) if family_scores else float("nan")
    values["envelope_kurtosis"] = finite(
        np.mean((envelope - np.mean(envelope)) ** 4)
        / max(np.mean((envelope - np.mean(envelope)) ** 2) ** 2, 1e-30)
    )
    values["envelope_spectral_crest"] = finite(
        np.max(broad) / max(float(np.mean(broad)), 1e-30)
    )
    values["rms"] = finite(np.sqrt(np.mean(np.square(centered))))

    raw_spec = np.fft.rfft(centered)
    raw_freqs = np.fft.rfftfreq(centered.size, d=1.0 / sampling_rate)
    upper_limit = 0.45 * sampling_rate
    edges = [3.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    edges = [edge for edge in edges if edge < upper_limit]
    if len(edges) < 2:
        edges = [max(2.0, upper_limit / 4.0), upper_limit]
    band_candidates: list[tuple[float, float, float]] = []
    band_values: dict[str, list[float]] = {
        family: [] for family in FAMILY_FIELDS
    }
    for low, high in zip(edges[:-1], edges[1:]):
        filtered_spec = raw_spec.copy()
        filtered_spec[(raw_freqs < low) | (raw_freqs > high)] = 0.0
        filtered = np.fft.irfft(filtered_spec, n=centered.size)
        band_envelope = np.abs(hilbert(filtered))
        band_envelope = detrend(band_envelope, type="constant")
        band_window = np.hanning(band_envelope.size)
        band_spec = np.square(np.abs(np.fft.rfft(band_envelope * band_window)))
        band_spec[0] = 0.0
        band_scores: list[float] = []
        for family, multiplier in frequencies.items():
            family_ratio, _ = family_log_snr(
                band_spec, raw_freqs, shaft_hz, multiplier
            )
            if math.isfinite(family_ratio):
                band_values[family].append(family_ratio)
                if family != "ftf":
                    band_scores.append(family_ratio)
        if band_scores:
            best_score = max(band_scores)
            band_candidates.append((best_score, math.sqrt(low * high), best_score))
    if band_candidates:
        all_scores = [value for _, _, value in band_candidates]
        best_band = max(band_candidates)
        values["fault_family_multiband_log_snr"] = best_band[0]
        values["fault_family_selected_band_hz"] = best_band[1]
        values["fault_family_multiband_top2_mean"] = float(
            np.mean(sorted(all_scores, reverse=True)[:2])
        )
        for family in FAMILY_FIELDS:
            values[f"{family}_multiband_log_snr"] = (
                max(band_values[family]) if band_values[family] else float("nan")
            )
    else:
        values.update(
            {
                "fault_family_multiband_log_snr": float("nan"),
                "fault_family_selected_band_hz": float("nan"),
                "fault_family_multiband_top2_mean": float("nan"),
                **{
                    f"{family}_multiband_log_snr": float("nan")
                    for family in FAMILY_FIELDS
                },
            }
        )
    return values


def read_frequencies(axis_data: object) -> dict[str, float]:
    embedded = axis_data.faultFrequencies
    result = {
        "bpfo": float(embedded.BPFOMultiple),
        "bpfi": float(embedded.BPFIMultiple),
        "bsf": float(embedded.BPFMultiple),
        "ftf": float(embedded.FTFMultiple),
    }
    if not all(math.isfinite(value) and value > 0 for value in result.values()):
        raise ValueError(f"invalid fault-frequency multipliers: {result}")
    return result


def extract(input_dir: Path, output: Path) -> None:
    paths = sorted(input_dir.glob("*_train.mat")) + sorted(input_dir.glob("*_test.mat"))
    if not paths:
        raise FileNotFoundError(f"no SCA MAT files found in {input_dir}")

    rows: list[dict[str, object]] = []
    for path in paths:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        split = path.stem.split("_", 1)[1]
        case = int(data["id"])
        axes = [
            (name, value)
            for name, value in data.items()
            if not name.startswith("__")
            and hasattr(value, "_fieldnames")
            and "rawData" in value._fieldnames
        ]
        if not axes:
            raise ValueError(f"{path}: no vibration channels")

        for axis_name, axis_data in axes:
            raw = np.asarray(axis_data.rawData)
            if raw.dtype == object:
                signals = [np.asarray(record, dtype=np.float64).ravel() for record in raw]
            else:
                signals = [raw[index].astype(np.float64) for index in range(raw.shape[0])]
            rpm_values = np.asarray(axis_data.RPM, dtype=np.float64)
            labels = np.asarray(axis_data.label, dtype=np.int64)
            timestamps = np.asarray(axis_data.time).astype(str)
            rates = np.asarray(axis_data.samplingRate, dtype=np.float64)
            frequencies = read_frequencies(axis_data)
            eligible = np.flatnonzero((labels != -1) & (rpm_values > 0))
            for snapshot, index in enumerate(eligible, start=1):
                features = snapshot_features(
                    signals[index],
                    float(rpm_values[index]),
                    float(rates[index]),
                    frequencies,
                )
                rows.append(
                    {
                        "dataset": "sca",
                        "split": split,
                        "case": case,
                        "axis": axis_name,
                        "snapshot": snapshot,
                        "timestamp": str(timestamps[index]),
                        "label": int(labels[index]),
                        "rpm": finite(rpm_values[index]),
                        "sampling_frequency_hz": finite(rates[index]),
                        "n_samples": int(signals[index].size),
                        **features,
                    }
                )
        print(f"extracted {path.name}: cumulative rows={len(rows)}", flush=True)

    rows.sort(key=lambda row: (row["split"], row["case"], row["axis"], row["snapshot"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw_archives/sca",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/features/sca_fault_band_features.csv",
    )
    args = parser.parse_args()
    extract(args.input_dir, args.output)


if __name__ == "__main__":
    main()
