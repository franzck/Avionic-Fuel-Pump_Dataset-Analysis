#!/usr/bin/env python3
"""
summarize_faults.py
───────────────────
Send Kurvendiskussion tables to LLM for natural language summaries.
All fault names, file paths, and ground truth labels are STRIPPED
before the LLM sees the data. The LLM only sees anonymous sensor data.

Output:
  ./fault_summaries/{fault}/{fault}_run{NN}_summary.txt
  ./fault_summaries/{fault}_mean_summary.txt
"""

import os
import re
import sys
import time
import glob
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:70b"

RESIDUALS_DIR = "./residuals"
OUTPUT_DIR = "./fault_summaries"

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

# Patterns to strip from text before sending to LLM
SANITIZE_PATTERNS = [
    r"(?i)FAULT:.*\n",           # FAULT: header lines
    r"(?i)fault\d+_\S+",         # fault01_whiteNoise etc
    r"(?i)healthy\S*",           # healthyFuelPumpRuns etc
    r"\./\S+",                   # file paths
    r"/home/\S+",                # absolute paths
    r"Reference:.*\n",           # Reference: line
    r"Averaged over.*\n",        # Averaged over line
    r"(?i)source:.*\n",          # Source: line
]


def sanitize(text):
    """Remove all fault names, file paths, and ground truth labels."""
    cleaned = text
    for pattern in SANITIZE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    # Remove empty lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


PROMPT_TEMPLATE = """\
You are an expert in industrial process-plant diagnostics.

Below is a residual analysis table for an UNKNOWN fault scenario in a
fuel system test rig. {run_info}

HOW TO READ THE DATA:
The residuals are computed as (test_value - reference_value) for each
sensor at each time step, then expressed as a percentage of that sensor's
reference operating range (max - min over the reference data).

- A residual of 0% means the test sensor reads EXACTLY like the reference.
- A residual of -50% means the test sensor reads LOWER than reference by
  half the sensor's total reference range.
- A residual of +10% means the test sensor reads HIGHER than reference by
  10% of its reference range.

COLUMN DEFINITIONS:
- Start%: residual at the beginning of the time window
- End%: residual at the end of the time window
- Mean%: average residual across the window (signed: negative = test is
  lower than reference, positive = test is higher)
- Std%: standard deviation of the residual — high values indicate added
  noise, oscillation, or instability NOT present in the reference data
- Min%/Max%: extreme residual values in the window
- Slope: linear trend of the residual in %/second — nonzero slope means
  the deviation is growing or shrinking over time within the window
- ZeroCr: number of zero crossings — the residual changes sign, indicating
  oscillation around the reference value
- Mono: "yes" if the residual changes monotonically (steadily grows or
  shrinks), "no" if it fluctuates

Sensors:
- p_Pump, p_FMU, p_Shut, p_Combustion: pressure sensors [Pa]
- Flow_PRV, Flow_Bypass, Flow_Pump: flow sensors [m³/s]
- Flow_Engine1..4: individual engine flow sensors [m³/s]
- RPM_Motor: motor speed [rad/s] (control input)
- Signal_Throttle: throttle position (control input)

DATA:
{kd_text}

INSTRUCTIONS:
For each time regime, describe what the residual data reveals. Consider
ALL features — not just mean values:
- A sensor with mean≈0% but high Std% has ADDED NOISE
- A sensor with large negative Mean% has a SUSTAINED OFFSET (reads lower)
- A sensor with low Mean% but steep Slope is DRIFTING within the window
- A sensor with many ZeroCr is OSCILLATING around the reference value
- Compare Start% vs End% to see if a deviation appears or disappears

Focus on sensors with any feature clearly above noise (~2% absolute).
For quiet regimes state "No significant deviations."
Do NOT speculate about fault types or root causes. Do NOT try to name
or identify the fault. Only describe what the numbers show."""


def query_llm(prompt, timeout=6000):
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 4096},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def summarize_file(kd_path, out_path, run_info):
    if os.path.exists(out_path):
        return "skip"

    with open(kd_path) as f:
        raw_text = f.read()

    clean_text = sanitize(raw_text)
    prompt = PROMPT_TEMPLATE.format(run_info=run_info, kd_text=clean_text)

    t0 = time.time()
    summary = query_llm(prompt)
    elapsed = time.time() - t0

    # Output file keeps fault name for OUR reference, but LLM never saw it
    with open(out_path, "w") as f:
        f.write(f"Source: {kd_path}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Note: LLM received sanitized data with no fault labels\n")
        f.write(f"{'='*60}\n\n")
        f.write(summary)

    return elapsed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        r = requests.get("http://localhost:11434", timeout=5)
        print(f"✅ Ollama: {r.text.strip()}  Model: {MODEL}\n")
    except Exception as e:
        print(f"❌ Cannot reach Ollama: {e}")
        sys.exit(1)

    for fault_name in FAULT_NAMES:
        fault_dir = os.path.join(RESIDUALS_DIR, fault_name)
        if not os.path.exists(fault_dir):
            print(f"⚠️  Missing: {fault_dir}")
            continue

        out_dir = os.path.join(OUTPUT_DIR, fault_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"{'='*50}")
        print(f"  {fault_name}")
        print(f"{'='*50}")

        # Per-run summaries
        run_files = sorted(glob.glob(
            os.path.join(fault_dir, f"{fault_name}_run*_kurvendiskussion.txt")))

        for idx, kd_path in enumerate(run_files, 1):
            basename = os.path.basename(kd_path).replace("_kurvendiskussion.txt", "")
            out_path = os.path.join(out_dir, f"{basename}_summary.txt")

            # Anonymous run info — no fault name
            result = summarize_file(
                kd_path, out_path,
                f"This is run {idx} of {len(run_files)} from the same test scenario.")

            if result == "skip":
                print(f"  ⏭️  run {idx:02d}")
            else:
                print(f"  ✅ run {idx:02d} ({result:.1f}s)")

        # Mean summary
        mean_kd = os.path.join(fault_dir, f"{fault_name}_kurvendiskussion_mean.txt")
        mean_out = os.path.join(out_dir, f"{fault_name}_mean_summary.txt")

        if os.path.exists(mean_kd):
            result = summarize_file(
                mean_kd, mean_out,
                "This data is averaged over 10 runs from the same test scenario.")

            if result == "skip":
                print(f"  ⏭️  mean")
            else:
                print(f"  ✅ mean ({result:.1f}s)")

        print()

    print(f"Done. Summaries in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
