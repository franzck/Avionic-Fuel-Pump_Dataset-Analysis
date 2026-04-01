#!/usr/bin/env python3
"""
compute_residuals.py
────────────────────
1. Mean healthy reference from all healthy runs
2. Residual = faulty - mean_healthy per run
3. Kurvendiskussion per sensor per time window per run + averaged
4. All values as % of healthy range

Output:
  ./residuals/healthy_reference.csv
  ./residuals/healthy_ranges.csv
  ./residuals/{fault}/
    {fault}_run{NN}_residual.csv
    {fault}_run{NN}_residual_pct.csv
    {fault}_run{NN}_kurvendiskussion.csv
    {fault}_run{NN}_kurvendiskussion.txt       ← per-run text
    {fault}_kurvendiskussion_mean.csv
    {fault}_kurvendiskussion_mean.txt           ← averaged text
"""

import os
import numpy as np
import pandas as pd

EXTRACTED_DIR = "./extracted_runs"
OUTPUT_DIR = "./residuals"
NUM_RUNS = 10
DOWNSAMPLE_FACTOR = 40

TIME_WINDOWS = [
    ("00-05s_idle",            0.0,  5.0),
    ("05-07s_rpm_ramp",        5.0,  7.0),
    ("07-10s_rpm_stable",      7.0, 10.0),
    ("10-15s_throttle_ramp",  10.0, 15.0),
    ("15-18s_high_plateau",   15.0, 18.0),
    ("18-22s_throttle_down",  18.0, 22.0),
    ("22-30s_final_stable",   22.0, 30.0),
]

FAULT_NAMES = [
    "fault01_whiteNoise",
    "fault02_sensorDrop_pFMU",
    "fault03_sensorDrop_Engine1",
    "fault04_pumpLeakage",
    "fault05_pumpDisplacement",
    "fault06_PRVLeakage",
    "fault07_orifice_FMU",
    "fault08_orifice_Bypass",
    "fault09_highBoostPressure",
]

# Only skip non-numeric ID columns
SKIP_COLS = {"UID1", "UID2"}


def downsample(df, factor):
    return df.iloc[::factor].reset_index(drop=True)


def kurvendiskussion(times, values):
    n = len(values)
    if n < 2:
        return {
            "start_val": float(values[0]) if n > 0 else 0.0,
            "end_val": float(values[-1]) if n > 0 else 0.0,
            "mean": float(np.mean(values)),
            "std": 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "slope": 0.0,
            "zero_crossings": 0,
            "monotonic": True,
        }

    start_val = float(values[0])
    end_val = float(values[-1])
    mean = float(np.mean(values))
    std = float(np.std(values))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    coeffs = np.polyfit(times, values, 1)
    slope = float(coeffs[0])
    signs = np.sign(values)
    signs[signs == 0] = 1
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    diffs = np.diff(values)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    return {
        "start_val": start_val, "end_val": end_val,
        "mean": mean, "std": std, "min": vmin, "max": vmax,
        "slope": slope, "zero_crossings": crossings, "monotonic": monotonic,
    }


def features_to_txt(title, sensor_cols, feature_df, time_windows):
    """Generate text table from a feature dataframe."""
    lines = [title, "All values as % of healthy sensor range", "Slope unit: %/second", ""]

    for wname, _, _ in time_windows:
        w_data = feature_df[feature_df["time_window"] == wname].copy()
        if w_data.empty:
            continue
        lines.append(f"--- {wname} ---")
        lines.append(
            f"{'Sensor':<25s} {'Start%':>8s} {'End%':>8s} {'Mean%':>8s} "
            f"{'Std%':>8s} {'Min%':>8s} {'Max%':>8s} {'Slope':>8s} "
            f"{'ZeroCr':>7s} {'Mono':>5s}"
        )
        for _, row in w_data.sort_values("sensor").iterrows():
            lines.append(
                f"{row['sensor']:<25s} "
                f"{row['start_val']:>8.1f} {row['end_val']:>8.1f} "
                f"{row['mean']:>8.1f} {row['std']:>8.1f} "
                f"{row['min']:>8.1f} {row['max']:>8.1f} "
                f"{row['slope']:>8.2f} "
                f"{row['zero_crossings']:>7.0f} "
                f"{'yes' if row['monotonic'] else 'no':>5s}"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load healthy
    healthy_dir = os.path.join(EXTRACTED_DIR, "healthyFuelPumpRuns")
    if not os.path.exists(healthy_dir):
        print(f"❌ Not found: {healthy_dir}")
        return

    healthy_list = []
    for r in range(1, NUM_RUNS + 1):
        path = os.path.join(healthy_dir, f"healthyFuelPumpRuns_run{r:02d}.csv")
        if not os.path.exists(path):
            continue
        healthy_list.append(downsample(pd.read_csv(path), DOWNSAMPLE_FACTOR))

    print(f"✅ {len(healthy_list)} healthy runs loaded")

    time_col = healthy_list[0].columns[0]
    sensor_cols = [c for c in healthy_list[0].columns if c != time_col and c not in SKIP_COLS]

    # Mean healthy reference
    min_len = min(len(df) for df in healthy_list)
    trimmed = [df.iloc[:min_len].reset_index(drop=True) for df in healthy_list]

    mean_healthy = pd.DataFrame()
    mean_healthy[time_col] = trimmed[0][time_col]
    for col in sensor_cols:
        stacked = np.stack([df[col].values[:min_len] for df in trimmed])
        mean_healthy[col] = np.mean(stacked, axis=0)

    ref_path = os.path.join(OUTPUT_DIR, "healthy_reference.csv")
    mean_healthy.to_csv(ref_path, index=False)
    print(f"📊 Mean healthy: {ref_path} ({min_len} samples)")

    # Healthy ranges
    healthy_range = {}
    for col in sensor_cols:
        r = mean_healthy[col].max() - mean_healthy[col].min()
        healthy_range[col] = r if r > 0 else 1e-20

    range_df = pd.DataFrame([{"sensor": k, "range": v} for k, v in healthy_range.items()])
    range_df.to_csv(os.path.join(OUTPUT_DIR, "healthy_ranges.csv"), index=False)

    print(f"\n📊 Healthy ranges:")
    for col in sensor_cols:
        print(f"   {col:25s}: {healthy_range[col]:.4g}")
    print()

    # Process faults
    for fault_name in FAULT_NAMES:
        fault_in_dir = os.path.join(EXTRACTED_DIR, fault_name)
        fault_out_dir = os.path.join(OUTPUT_DIR, fault_name)
        os.makedirs(fault_out_dir, exist_ok=True)

        print(f"{'='*60}")
        print(f"  {fault_name}")
        print(f"{'='*60}")

        if not os.path.exists(fault_in_dir):
            print(f"  ⚠️  Not found\n")
            continue

        all_run_features = []

        for r in range(1, NUM_RUNS + 1):
            fault_path = os.path.join(fault_in_dir, f"{fault_name}_run{r:02d}.csv")
            if not os.path.exists(fault_path):
                continue

            faulty = downsample(pd.read_csv(fault_path), DOWNSAMPLE_FACTOR)
            flen = min(len(faulty), len(mean_healthy))
            f = faulty.iloc[:flen].reset_index(drop=True)
            h = mean_healthy.iloc[:flen].reset_index(drop=True)
            shared = [c for c in sensor_cols if c in f.columns]

            # Raw residual
            residual = pd.DataFrame()
            residual[time_col] = h[time_col]
            for col in shared:
                residual[col] = f[col] - h[col]
            residual.to_csv(
                os.path.join(fault_out_dir, f"{fault_name}_run{r:02d}_residual.csv"),
                index=False)

            # Pct residual
            residual_pct = pd.DataFrame()
            residual_pct[time_col] = h[time_col]
            for col in shared:
                residual_pct[col] = (residual[col] / healthy_range[col]) * 100.0
            residual_pct.to_csv(
                os.path.join(fault_out_dir, f"{fault_name}_run{r:02d}_residual_pct.csv"),
                index=False)

            # Kurvendiskussion per run
            times = residual_pct[time_col].values
            run_features = []
            for wname, t_start, t_end in TIME_WINDOWS:
                mask = (times >= t_start) & (times < t_end)
                t_seg = times[mask]
                if len(t_seg) < 2:
                    continue
                for col in shared:
                    feats = kurvendiskussion(t_seg, residual_pct[col].values[mask])
                    feats["time_window"] = wname
                    feats["sensor"] = col
                    feats["run"] = r
                    run_features.append(feats)

            run_df = pd.DataFrame(run_features)

            # Save per-run CSV
            run_df.to_csv(
                os.path.join(fault_out_dir, f"{fault_name}_run{r:02d}_kurvendiskussion.csv"),
                index=False)

            # Save per-run TXT
            run_txt = features_to_txt(
                f"FAULT: {fault_name} — Run {r:02d}\n"
                f"Reference: mean of {len(healthy_list)} healthy runs",
                shared, run_df, TIME_WINDOWS)
            with open(os.path.join(fault_out_dir, f"{fault_name}_run{r:02d}_kurvendiskussion.txt"), "w") as fout:
                fout.write(run_txt)

            all_run_features.append(run_df)

            max_mean = run_df.groupby("sensor")["mean"].apply(lambda x: x.abs().max())
            top = max_mean.idxmax()
            print(f"  Run {r:2d}: largest mean dev {max_mean[top]:.1f}% in {top}")

        if not all_run_features:
            print(f"  ⚠️  No runs\n")
            continue

        # Mean Kurvendiskussion
        all_df = pd.concat(all_run_features, ignore_index=True)
        numeric_cols = ["start_val", "end_val", "mean", "std", "min", "max",
                        "slope", "zero_crossings"]

        avg = all_df.groupby(["time_window", "sensor"])[numeric_cols].mean().reset_index()
        mono = all_df.groupby(["time_window", "sensor"])["monotonic"].all().reset_index()
        avg = avg.merge(mono, on=["time_window", "sensor"])
        for c in numeric_cols:
            avg[c] = avg[c].round(2)

        avg.to_csv(
            os.path.join(fault_out_dir, f"{fault_name}_kurvendiskussion_mean.csv"),
            index=False)

        avg_txt = features_to_txt(
            f"FAULT: {fault_name}\n"
            f"Reference: mean of {len(healthy_list)} healthy runs\n"
            f"Averaged over {len(all_run_features)} faulty runs",
            sensor_cols, avg, TIME_WINDOWS)
        txt_path = os.path.join(fault_out_dir, f"{fault_name}_kurvendiskussion_mean.txt")
        with open(txt_path, "w") as fout:
            fout.write(avg_txt)

        print(f"  💾 {len(all_run_features)} runs + mean → {fault_out_dir}/\n")

    print("Done.")


if __name__ == "__main__":
    main()
