#!/usr/bin/env python
"""
pick_failure_samples.py — Identify difficult / failure-case samples from a
baseline_sweep.py run for use in a qualitative failure-analysis figure.

Reads the per-image CSV produced by baseline_sweep.py for a chosen
FreqSpec method (default: freqspec_default) and ranks images by:
  - lowest acceptance rate (hard for the draft)
  - or lowest speedup (hard for lookahead)
  - or highest acceptance rate (easy / sanity-check pair)

Prints the indices (and prompts if manifest.json is available) so you can
feed them into assemble_qualitative_figure.py via --sample_indices.

Usage:
    python pick_failure_samples.py \\
        --sweep_dir /mnt/HDD_12TB/bam_ki/results/qualitative_coco_run \\
        --method freqspec_default \\
        --mode low_accept --n 4

Then pass the printed indices to assemble_qualitative_figure.py.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="baseline_sweep.py --out_root path")
    p.add_argument("--method", default="freqspec_default",
                   help="Method whose results.csv we read")
    p.add_argument("--mode", default="low_accept",
                   choices=["low_accept", "high_accept",
                            "low_speedup", "high_speedup"],
                   help="Ranking criterion")
    p.add_argument("--n", type=int, default=4,
                   help="Number of samples to print")
    p.add_argument("--target_baseline", default="target_s50",
                   help="Method used for speedup denominator")
    args = p.parse_args()

    method_dir = Path(args.sweep_dir) / args.method
    csv_path = method_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    # Read freqspec results
    fs_rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row["accept_rate"] = float(row["accept_rate"]) if row.get(
                "accept_rate") else 0.0
            row["time_sec"] = float(row["time_sec"])
            row["idx"] = int(row["idx"])
            fs_rows.append(row)

    # Read target baseline for per-sample speedup
    base_csv = Path(args.sweep_dir) / args.target_baseline / "results.csv"
    base_time = {}
    if base_csv.exists():
        with open(base_csv) as f:
            for row in csv.DictReader(f):
                base_time[int(row["idx"])] = float(row["time_sec"])
    else:
        print(f"[warn] baseline csv {base_csv} not found, "
              f"speedup ranking unavailable")

    # Compute per-row speedup
    for r in fs_rows:
        tb = base_time.get(r["idx"])
        r["speedup"] = (tb / r["time_sec"]) if (tb and r["time_sec"] > 0) else 0.0

    # Rank
    if args.mode == "low_accept":
        fs_rows.sort(key=lambda r: r["accept_rate"])
    elif args.mode == "high_accept":
        fs_rows.sort(key=lambda r: -r["accept_rate"])
    elif args.mode == "low_speedup":
        fs_rows.sort(key=lambda r: r["speedup"])
    elif args.mode == "high_speedup":
        fs_rows.sort(key=lambda r: -r["speedup"])

    picks = fs_rows[:args.n]

    # Load manifest for prompts
    manifest_path = Path(args.sweep_dir) / "manifest.json"
    prompts = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        prompts = {it["idx"]: it.get("prompt", "") for it in manifest}

    # Print summary
    print(f"\n[picker] mode={args.mode}  method={args.method}  n={args.n}")
    print(f"[picker] manifest: {len(prompts)} prompts loaded\n")
    print(f"{'idx':>4}  {'accept':>7}  {'speedup':>7}  prompt")
    print("-" * 80)
    for r in picks:
        prompt = prompts.get(r["idx"], "")
        short = (prompt[:55] + "...") if len(prompt) > 55 else prompt
        print(f"{r['idx']:>4d}  {r['accept_rate']:>7.3f}  "
              f"{r['speedup']:>7.2f}  {short}")

    # Print one-line for copy-paste
    idx_list = " ".join(str(r["idx"]) for r in picks)
    print(f"\n[picker] copy into assemble_qualitative_figure.py:")
    print(f"    --sample_indices {idx_list}")


if __name__ == "__main__":
    main()
