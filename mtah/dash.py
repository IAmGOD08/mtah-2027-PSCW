"""Label-free sanity checks for a possession labelling.

These are the numbers to watch before you look at any score:
- possession count (a football match has roughly 250-400)
- mean and median possession duration in seconds
- share of possessions under 2 seconds (flicker indicator)
- dead-ball stretch count and total duration
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]


def possession_stats(labels):
    """Count and duration of maximal runs of H or A. D and X break runs."""
    codes = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    changes = np.flatnonzero(np.diff(codes, prepend=codes[0] - 1))
    runs = np.split(np.arange(len(codes)), changes[1:])
    poss = [r for r in runs if codes[r[0]] in (0, 1)]
    lengths = np.array([len(r) for r in poss], dtype=float)
    return {
        "n_possessions": len(poss),
        "mean_dur_s": lengths.mean() / FPS if len(poss) else 0.0,
        "median_dur_s": np.median(lengths) / FPS if len(poss) else 0.0,
        "share_under_2s": float(np.mean(lengths < 2 * FPS)) if len(poss) else 0.0,
    }


def deadball_stats(labels):
    """Count and total duration of D stretches."""
    is_d = labels == "D"
    changes = np.flatnonzero(np.diff(is_d.astype(int), prepend=0))
    runs = np.split(np.arange(len(is_d)), changes[1:])
    d_runs = [r for r in runs if is_d[r[0]]]
    lengths = np.array([len(r) for r in d_runs], dtype=float)
    return {
        "n_deadball": len(d_runs),
        "total_deadball_s": lengths.sum() / FPS if len(d_runs) else 0.0,
        "mean_deadball_s": lengths.mean() / FPS if len(d_runs) else 0.0,
    }


def report(csv_path):
    df = pd.read_csv(csv_path)
    labels = df["label"].values
    p = possession_stats(labels)
    d = deadball_stats(labels)
    print(f"--- {os.path.basename(csv_path)} ---")
    for k, v in {**p, **d}.items():
        print(f"  {k:>20}: {v:.3f}" if isinstance(v, float) else f"  {k:>20}: {v}")
    return {**p, **d}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", help="one or more submission CSVs")
    args = parser.parse_args()
    for path in args.csv:
        report(path)
