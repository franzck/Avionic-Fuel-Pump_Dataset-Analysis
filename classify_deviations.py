#!/usr/bin/env python3
"""
classify_deviations.py
──────────────────────
Generalized, transferable deviation classifier.

Severity is a continuous 0-100 scale derived from the residual magnitude.
Pattern tags describe WHAT happened, severity describes HOW MUCH.
No output for a sensor-window = healthy.

Severity bands:
  0-10   negligible (suppressed, not emitted)
  10-30  minor
  30-60  moderate
  60-90  major
  90-100 critical (e.g. sensor dropout)

Patterns detected:
  offset          sustained shift from healthy
  noise           added variability around healthy
  noisy_offset    offset + instability
  drift           deviation changing within window
  step            deviation appears/disappears at boundary
  transient       brief spike, not sustained
  oscillation     repeated sign changes
  dropout         sensor reads near zero / lost signal (~100% negative offset)
  saturation      sensor reads near max / stuck high (~100% positive offset)
"""

import os
import math
import numpy as np
import pandas as pd

RESIDUALS_DIR = "./residuals"
OUTPUT_DIR = "./classifications"

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

# ──────────────────────────────────────────────
# Minimum thresholds to suppress noise floor
# These are the ONLY tunables — set once per plant/testrig
# Everything else scales automatically
# ──────────────────────────────────────────────
MIN_OFFSET_PCT = 3.0      # below this, offset is noise
MIN_STD_PCT = 2.0         # below this, variability is normal
MIN_SLOPE_PCT_S = 1.0     # below this, slope is flat
MIN_PEAK_PCT = 5.0        # below this, peaks are noise
MIN_STEP_PCT = 5.0        # below this, boundary change is noise
MIN_ZEROCROSS = 3         # below this, crossings are normal


def severity_from_pct(value_pct):
    """Map absolute percentage deviation to 0-100 severity.
    Uses sqrt scaling so small deviations still register
    but large ones saturate toward 100."""
    clamped = min(abs(value_pct), 100.0)
    return round(math.sqrt(clamped / 100.0) * 100.0, 1)


def severity_band(severity):
    if severity < 10:
        return "negligible"
    elif severity < 30:
        return "minor"
    elif severity < 60:
        return "moderate"
    elif severity < 90:
        return "major"
    else:
        return "critical"


def classify(row):
    """Return list of findings. Empty = healthy."""
    findings = []

    mean = row["mean"]
    std = row["std"]
    slope = row["slope"]
    vmin = row["min"]
    vmax = row["max"]
    start = row["start_val"]
    end = row["end_val"]
    zc = row["zero_crossings"]
    mono = row["monotonic"]

    abs_mean = abs(mean)
    abs_slope = abs(slope)
    abs_step = abs(end - start)
    peak = max(abs(vmin), abs(vmax))

    # ── Dropout / Saturation (special cases of extreme offset) ──
    if mean <= -90:
        sev = severity_from_pct(abs_mean)
        findings.append({
            "pattern": "dropout",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"signal lost: {mean:.1f}% offset from healthy",
        })
        # Still check noise on top of dropout
        if std >= MIN_STD_PCT:
            sev_n = severity_from_pct(std)
            findings.append({
                "pattern": "noisy_dropout",
                "severity": sev_n,
                "band": severity_band(sev_n),
                "detail": f"std={std:.1f}% on top of dropout",
            })
        return findings

    if mean >= 90:
        sev = severity_from_pct(abs_mean)
        findings.append({
            "pattern": "saturation",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"signal saturated: +{mean:.1f}% above healthy",
        })
        return findings

    # ── Sustained offset ──
    if abs_mean >= MIN_OFFSET_PCT:
        direction = "lower" if mean < 0 else "higher"
        sev = severity_from_pct(abs_mean)
        if std >= MIN_STD_PCT:
            findings.append({
                "pattern": "noisy_offset",
                "severity": sev,
                "band": severity_band(sev),
                "detail": f"{abs_mean:.1f}% {direction}, std={std:.1f}%",
            })
        else:
            findings.append({
                "pattern": "offset",
                "severity": sev,
                "band": severity_band(sev),
                "detail": f"{abs_mean:.1f}% {direction} than healthy",
            })

    # ── Added noise (mean near zero but variable) ──
    elif std >= MIN_STD_PCT:
        sev = severity_from_pct(std)
        findings.append({
            "pattern": "noise",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"std={std:.1f}% around healthy mean",
        })

    # ── Drift (deviation changing within window) ──
    if abs_slope >= MIN_SLOPE_PCT_S:
        direction = "increasing" if slope > 0 else "decreasing"
        # Severity from total drift over window duration
        # (slope alone doesn't capture severity well)
        window_drift = abs_step if abs_step > abs_slope else abs_slope * 5
        sev = severity_from_pct(window_drift)
        findings.append({
            "pattern": "drift",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"{direction} at {slope:.2f}%/s ({start:.1f}%→{end:.1f}%)",
        })

    # ── Step change at boundary ──
    if abs_step >= MIN_STEP_PCT and abs_slope < MIN_SLOPE_PCT_S:
        # Step but no continuous drift → abrupt change
        direction = "appears" if abs(end) > abs(start) else "disappears"
        sev = severity_from_pct(abs_step)
        findings.append({
            "pattern": "step",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"deviation {direction}: {start:.1f}%→{end:.1f}%",
        })

    # ── Transient spike ──
    if peak >= MIN_PEAK_PCT and abs_mean < MIN_OFFSET_PCT:
        sev = severity_from_pct(peak)
        findings.append({
            "pattern": "transient",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"peak {peak:.1f}% but mean≈{mean:.1f}%",
        })

    # ── Oscillation ──
    if zc >= MIN_ZEROCROSS:
        # Severity from combination of crossing frequency and amplitude
        osc_amplitude = (abs(vmax) + abs(vmin)) / 2
        sev = severity_from_pct(osc_amplitude)
        findings.append({
            "pattern": "oscillation",
            "severity": sev,
            "band": severity_band(sev),
            "detail": f"{zc} zero crossings, amplitude ±{osc_amplitude:.1f}%",
        })

    return findings


def format_txt(fault_name, out_df):
    lines = [f"FAULT: {fault_name}", ""]
    if out_df.empty:
        lines.append("No deviations above noise floor.")
        return "\n".join(lines)

    current_window = None
    for _, r in out_df.iterrows():
        if r["time_window"] != current_window:
            current_window = r["time_window"]
            lines.append(f"--- {current_window} ---")
        lines.append(
            f"  {r['sensor']:25s} {r['band']:10s} "
            f"(sev={r['severity']:5.1f}) [{r['pattern']}] {r['detail']}"
        )
    lines.append("")
    lines.append(f"Total: {len(out_df)} findings")
    lines.append(f"Critical: {len(out_df[out_df['band']=='critical'])}")
    lines.append(f"Major:    {len(out_df[out_df['band']=='major'])}")
    lines.append(f"Moderate: {len(out_df[out_df['band']=='moderate'])}")
    lines.append(f"Minor:    {len(out_df[out_df['band']=='minor'])}")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fault_name in FAULT_NAMES:
        kd_path = os.path.join(RESIDUALS_DIR, fault_name,
                               f"{fault_name}_kurvendiskussion_mean.csv")
        if not os.path.exists(kd_path):
            print(f"⚠️  Missing: {kd_path}")
            continue

        df = pd.read_csv(kd_path)

        rows_out = []
        for _, row in df.iterrows():
            findings = classify(row)
            for f in findings:
                rows_out.append({
                    "fault": fault_name,
                    "time_window": row["time_window"],
                    "sensor": row["sensor"],
                    "pattern": f["pattern"],
                    "severity": f["severity"],
                    "band": f["band"],
                    "detail": f["detail"],
                    # Raw values for ontology
                    "mean_pct": round(row["mean"], 2),
                    "std_pct": round(row["std"], 2),
                    "slope_pct_s": round(row["slope"], 2),
                    "min_pct": round(row["min"], 2),
                    "max_pct": round(row["max"], 2),
                    "start_pct": round(row["start_val"], 2),
                    "end_pct": round(row["end_val"], 2),
                    "zero_crossings": int(row["zero_crossings"]),
                })

        out_df = pd.DataFrame(rows_out)

        csv_path = os.path.join(OUTPUT_DIR, f"{fault_name}_classifications.csv")
        out_df.to_csv(csv_path, index=False)

        txt_path = os.path.join(OUTPUT_DIR, f"{fault_name}_classifications.txt")
        with open(txt_path, "w") as fout:
            fout.write(format_txt(fault_name, out_df))

        n = len(out_df)
        sensors = out_df["sensor"].nunique() if n > 0 else 0
        crits = len(out_df[out_df["band"] == "critical"]) if n > 0 else 0
        print(f"📋 {fault_name}: {n} findings, {sensors} sensors, {crits} critical → {csv_path}")

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
