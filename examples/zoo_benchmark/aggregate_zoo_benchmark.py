"""Collect zoo-benchmark results into one CSV and print a body x brain table.

Usage:
  python aggregate_zoo_benchmark.py /scratch/jed/ariel_zoo_benchmark
Writes <root>/summary.csv (one row per finished run).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/summary.json")):
        row = json.loads(path.read_text())
        row["run_dir"] = path.parent.name
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary.json found under {args.root}")

    df = pd.DataFrame(rows)
    df.drop(columns=["stop"]).to_csv(args.root / "summary.csv", index=False)
    print(f"{len(df)} runs -> {args.root / 'summary.csv'}\n")

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("Median best x-speed (m/s):")
        print(df.pivot_table(index="body", columns="brain", values="best_xspeed", aggfunc="median").round(4))
        print("\nRuns per cell:")
        print(df.pivot_table(index="body", columns="brain", values="seed", aggfunc="count").fillna(0).astype(int))


if __name__ == "__main__":
    main()
