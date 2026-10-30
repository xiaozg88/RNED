#!/usr/bin/env python3
"""Extract and evaluate edge-feasible statistical signal-processing features.

The experiment keeps the frozen causal protocol of the RMS detector and only
replaces the scalar feature.  IMS censored bearings calibrate the zero-false-
alarm threshold; XJTU-SY and FEMTO/PRONOSTIA are validation streams.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, spectrogram
from scipy.stats import kurtosis


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "env_rms",
    "env_kurtosis",
    "fault_harmonic_ratio",
    "env_spectral_concentration",
    "env_spectral_gini",
    "env_spectral_crest",
    "env_spectral_kurtosis",
    "stft_spectral_kurtosis_max",
    "hf_env_ratio",
]
GEOMETRY = {
    "ims": (16, 0.331, 2.815, 15.17, 2000.0, 20000.0),
    "xjtu": (8, 7.92, 34.55, 0.0, None, 25600.0),
    "femto": (13, 3.5, 25.6, 0.0, None, 25600.0),
}
XJTU_RPM = {"35Hz12kN": 2100.0, "37.5Hz11kN": 2250.0, "40Hz10kN": 2400.0}
FEMTO_RPM = {"1": 1800.0, "2": 1650.0, "3": 1500.0}


def fault_frequencies(n_balls, ball_diameter, pitch_diameter, angle_deg, rpm):
    shaft = rpm / 60.0
    cosine = math.cos(math.radians(angle_deg))
    bpfo = (n_balls / 2.0) * shaft * (1.0 - ball_diameter / pitch_diameter * cosine)
    bpfi = (n_balls / 2.0) * shaft * (1.0 + ball_diameter / pitch_diameter * cosine)
    bsf = (pitch_diameter / (2.0 * ball_diameter)) * shaft * (
        1.0 - (ball_diameter / pitch_diameter * cosine) ** 2
    )
    return bpfo, bpfi, bsf


def permutation_complexity(signal, order=4, max_windows=4096):
    count = np.zeros(math.factorial(order), dtype=np.int64)
    stride = max(1, (len(signal) - order + 1) // max_windows)
    windows = np.lib.stride_tricks.sliding_window_view(signal, order)[::stride]
    ranks = np.argsort(windows, axis=1)
    codes = np.zeros(len(windows), dtype=np.int64)
    for column in range(order):
        codes += ranks[:, column] * (order ** column)
    np.add.at(count, codes, 1)
    probabilities = count[count > 0] / count.sum()
    entropy = -np.sum(probabilities * np.log(probabilities)) / math.log(len(count))
    return float(np.exp(-entropy))


def signal_features(signal, sampling_rate, bpfo, bpfi, bsf):
    signal = np.asarray(signal, dtype=np.float64)
    signal -= signal.mean()
    envelope = np.abs(hilbert(signal))
    envelope -= envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope)) ** 2
    frequencies = np.fft.rfftfreq(len(envelope), d=1.0 / sampling_rate)
    band = (frequencies >= 10.0) & (frequencies <= min(5000.0, frequencies[-1]))
    power = spectrum[band]
    total_power = float(power.sum())
    probabilities = power / total_power if total_power > 0 else np.full(len(power), 1.0 / len(power))

    normalized_entropy = -np.sum(probabilities * np.log(probabilities + 1e-15)) / math.log(len(power))
    sorted_power = np.sort(power)
    index = np.arange(1, len(power) + 1, dtype=np.float64)
    denominator = len(power) * sorted_power.sum()
    gini = float(np.sum((2.0 * index - len(power) - 1.0) * sorted_power) / denominator) if denominator > 0 else 0.0

    fault_power = 0.0
    for center_frequency in (bpfo, bpfi, bsf):
        for harmonic in range(1, 6):
            target = harmonic * center_frequency
            if target + 2.0 >= frequencies[-1]:
                break
            fault_power += float(spectrum[(frequencies >= target - 2.0) & (frequencies <= target + 2.0)].sum())

    if len(signal) >= 512:
        frequencies_stft, _, spectral_density = spectrogram(
            signal, fs=sampling_rate, nperseg=256, noverlap=128, window="hann", scaling="density"
        )
        useful = frequencies_stft >= 10.0
        if spectral_density.shape[1] >= 4:
            stft_kurtosis = kurtosis(spectral_density[useful], axis=1, fisher=False, bias=False)
            stft_max = float(np.nanmax(stft_kurtosis))
        else:
            stft_max = float(kurtosis(spectral_density[useful].ravel(), fisher=False, bias=False))
    else:
        stft_max = float(kurtosis(signal, fisher=False, bias=False))

    high_frequency = frequencies >= 1000.0
    return {
        "env_rms": float(np.sqrt(np.mean(envelope ** 2))),
        "env_kurtosis": float(kurtosis(envelope, fisher=False, bias=False)),
        "fault_harmonic_ratio": float(fault_power / total_power) if total_power > 0 else 0.0,
        "env_spectral_concentration": float(np.exp(-normalized_entropy)),
        "env_spectral_gini": gini,
        "env_spectral_crest": float(probabilities.max() / (probabilities.mean() + 1e-15)),
        "env_spectral_kurtosis": float(kurtosis(power, fisher=False, bias=False)),
        "stft_spectral_kurtosis_max": stft_max,
        "hf_env_ratio": float(spectrum[high_frequency].sum() / spectrum.sum()) if spectrum.sum() > 0 else 0.0,
    }


def process_task(task):
    dataset, group, bearing, snapshot, path, axes, skip_header, delimiter = task
    if delimiter == ",":
        with open(path, "r") as raw_stream:
            first_line = raw_stream.readline()
        if ";" in first_line:
            delimiter = ";"
    signal_matrix = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix[:, None]
    if dataset == "ims":
        geometry = GEOMETRY["ims"]
        bpfo, bpfi, bsf = fault_frequencies(*geometry[:5])
        sampling_rate = geometry[5]
    elif dataset == "xjtu":
        rpm = XJTU_RPM[group]
        bpfo, bpfi, bsf = fault_frequencies(*GEOMETRY["xjtu"][:4], rpm)
        sampling_rate = GEOMETRY["xjtu"][5]
    else:
        rpm = FEMTO_RPM[bearing.split("_")[0][-1]]
        bpfo, bpfi, bsf = fault_frequencies(*GEOMETRY["femto"][:4], rpm)
        sampling_rate = GEOMETRY["femto"][5]

    rows = []
    for axis, column, bearing_id in axes:
        values = signal_features(signal_matrix[:, column], sampling_rate, bpfo, bpfi, bsf)
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "bearing": bearing_id,
                "axis": axis,
                "snapshot": snapshot,
                **values,
            }
        )
    return rows


def task_output_keys(task):
    dataset, group, _bearing, snapshot, _path, axes, _skip, _delimiter = task
    return {
        (dataset, group, bearing_id, axis, str(snapshot))
        for axis, _column, bearing_id in axes
    }


def make_tasks(feature_dir, ims_root, xjtu_root, femto_root):
    tasks = []
    ims_files = {}
    for dirpath, dirs, files in os.walk(ims_root):
        for filename in files:
            if filename in {"1st_test", "2nd_test", "4th_test"} or len(filename.split(".")) == 6:
                ims_files[filename] = Path(dirpath) / filename
        dirs.sort()

    grouped = {}
    for test_name, csv_name in (("test1", "test1.csv"), ("test2", "test2.csv"), ("test3", "test3.csv")):
        with (feature_dir / csv_name).open() as stream:
            for row in csv.DictReader(stream):
                key = (test_name, row["filename"])
                grouped.setdefault(key, []).append(int(row["channel"]) - 1)
    for (test_name, filename), columns in grouped.items():
        if filename not in ims_files:
            raise FileNotFoundError(filename)
        axes = []
        for column in columns:
            bearing_number = column // 2 + 1 if test_name == "test1" else column + 1
            axes.append((str(column + 1), column, f"{test_name}_b{bearing_number}"))
        tasks.append(
            (
                "ims",
                test_name,
                "multi",
                len([key for key in grouped if key[0] == test_name and key[1] <= filename]),
                str(ims_files[filename]),
                axes,
                0,
                None,
            )
        )

    for dirpath, dirs, files in os.walk(xjtu_root):
        dirs.sort()
        bearing = Path(dirpath).name
        group = Path(dirpath).parent.name
        if group not in XJTU_RPM or not bearing.startswith("Bearing"):
            continue
        for filename in sorted((name for name in files if name.endswith(".csv")), key=lambda name: int(name[:-4])):
            tasks.append(
                (
                    "xjtu",
                    group,
                    bearing,
                    int(filename[:-4]),
                    str(Path(dirpath) / filename),
                    [("H", 0, bearing), ("V", 1, bearing)],
                    1,
                    ",",
                )
            )

    for dirpath, dirs, files in os.walk(femto_root):
        dirs.sort()
        bearing = Path(dirpath).name
        group = Path(dirpath).parent.name
        if group not in {"Learning_set", "Full_Test_Set"} or not bearing.startswith("Bearing"):
            continue
        for filename in sorted((name for name in files if name.startswith("acc_") and name.endswith(".csv"))):
            tasks.append(
                (
                    "femto",
                    group,
                    bearing,
                    int(filename[4:-4]),
                    str(Path(dirpath) / filename),
                    [("H", 4, bearing), ("V", 5, bearing)],
                    0,
                    ",",
                )
            )
    return tasks


def extract(feature_dir, output_path):
    tasks = make_tasks(
        feature_dir,
        ROOT.parent / "downloads/raw_extracted/ims",
        ROOT.parent / "downloads/raw_xjtu/XJTU-SY_Bearing_Datasets",
        ROOT.parent / "downloads/raw_extracted/femto/10. FEMTO Bearing",
    )
    print(f"extracting {len(tasks)} snapshots", flush=True)
    temporary_path = output_path.with_suffix(".csv.tmp")
    workers = min(8, os.cpu_count() or 1)
    fieldnames = ["dataset", "group", "bearing", "axis", "snapshot", *FEATURES]
    resume = temporary_path.exists()
    completed = set()
    if resume:
        with temporary_path.open() as stream:
            completed = {
                (row["dataset"], row["group"], row["bearing"], row["axis"], row["snapshot"])
                for row in csv.DictReader(stream)
            }
        tasks = [task for task in tasks if not task_output_keys(task).issubset(completed)]
        print(f"resuming; {len(completed)} axis rows already completed", flush=True)
    with temporary_path.open("a" if resume else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not resume:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(process_task, tasks, chunksize=64), 1):
                writer.writerows(result)
                if index % 500 == 0 or index == len(tasks):
                    print(f"processed {index}/{len(tasks)} snapshots", flush=True)
                    stream.flush()
    temporary_path.replace(output_path)
    print(f"wrote feature rows to {output_path}")


def evaluate(feature_path, result_path):
    streams = {}
    with feature_path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["dataset"], row["group"], row["bearing"], row["axis"])
            streams.setdefault(key, []).append(row)

    bearings = {}
    for (dataset, group, bearing, axis), rows in streams.items():
        rows.sort(key=lambda row: int(row["snapshot"]))
        if len(rows) < 70:
            continue
        bearing_key = f"{dataset}_{bearing}"
        bearings.setdefault(
            bearing_key,
            {"dataset": dataset, "axes": {}},
        )["axes"][axis] = rows

    ims_failed = {"test1_b3", "test1_b4", "test2_b1", "test3_b3"}
    output = {}
    for feature in FEATURES:
        rates = {}
        for bearing_key, bearing in bearings.items():
            axis_rates = {}
            for axis, rows in bearing["axes"].items():
                values = np.array([float(row[feature]) for row in rows], dtype=np.float64)
                baseline = float(np.median(values[10:60]))
                excess = np.maximum(values[60:] / baseline - 1.2, 0.0)
                axis_rates[axis] = np.cumsum(excess) / np.arange(1, len(excess) + 1)
            rates[bearing_key] = axis_rates

        censored = [key for key in bearings if key.startswith("ims_") and key[4:] not in ims_failed]
        threshold = 1.01 * max(float(np.max(rate)) for key in censored for rate in rates[key].values())
        detected = []
        leads = []
        per_dataset = {"ims": [0, 0], "xjtu": [0, 0], "femto": [0, 0]}
        for key, bearing in bearings.items():
            dataset = bearing["dataset"]
            failed = key[4:] in ims_failed if dataset == "ims" else True
            if not failed:
                continue
            per_dataset[dataset][1] += 1
            bearing_rates = rates[key]
            first_alarm = min(
                int(np.flatnonzero(rate[29:] > threshold)[0] + 29) if np.any(rate[29:] > threshold) else len(rate)
                for rate in bearing_rates.values()
            )
            if first_alarm < len(next(iter(bearing_rates.values()))):
                detected.append(key)
                leads.append(100.0 * (len(next(iter(bearing_rates.values()))) - first_alarm) / len(next(iter(bearing_rates.values()))))
                per_dataset[dataset][0] += 1

        false_alarm_axes = 0
        false_alarm_bearings = 0
        pseudo_axes = 0
        pseudo_bearings = 0
        for key, bearing_rates in rates.items():
            if not key.startswith(("xjtu_", "femto_")):
                continue
            eligible_axes = [
                axis
                for axis, rate in bearing_rates.items()
                if len(bearings[key]["axes"][axis]) >= 150
            ]
            if not eligible_axes:
                continue
            pseudo_bearings += 1
            bearing_alarm = False
            for axis, rate in bearing_rates.items():
                if len(bearings[key]["axes"][axis]) < 150:
                    continue
                pseudo_axes += 1
                if np.any(rate[29:90] > threshold):
                    false_alarm_axes += 1
                    bearing_alarm = True
            false_alarm_bearings += int(bearing_alarm)

        output[feature] = {
            "detected": f"{len(detected)}/34",
            "detection_rate": 100.0 * len(detected) / 34,
            "median_lead_pct_life": float(np.median(leads)) if leads else None,
            "zero_fa_threshold": threshold,
            "per_dataset_detection": {name: f"{hit}/{total}" for name, (hit, total) in per_dataset.items()},
            "cross_rig_pseudo_health_axis_fa_rate": 100.0 * false_alarm_axes / pseudo_axes if pseudo_axes else None,
            "n_false_alarm_axes": false_alarm_axes,
            "n_pseudo_health_axes": pseudo_axes,
            "cross_rig_pseudo_health_bearing_fa_rate": 100.0 * false_alarm_bearings / pseudo_bearings if pseudo_bearings else None,
            "n_false_alarm_bearings": false_alarm_bearings,
            "n_pseudo_health_bearings": pseudo_bearings,
        }
    result_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def main():
    feature_path = ROOT / "data/features/modern_signal_features.csv"
    result_path = ROOT / "results/results_run_to_failure_modern_signal_features.json"
    if not feature_path.exists():
        extract(ROOT / "data/features", feature_path)
    evaluate(feature_path, result_path)


if __name__ == "__main__":
    main()
