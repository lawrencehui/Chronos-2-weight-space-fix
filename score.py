"""Leaderboard-style score for a GIFT-Eval results CSV.

Usage: python score.py <results.csv> [--ref ref/leaderboard/t0-gamma/all_results.csv]
CRPS and MASE are each divided by Seasonal Naive per config, then geometric-meaned
over the configs present (the leaderboard's aggregation; 0.4520 for t0-gamma on all 97).
With --ref, also compares config by config and reports wins/losses and the largest changes.
"""

import argparse
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CRPS, MASE = "eval_metrics/mean_weighted_sum_quantile_loss", "eval_metrics/MASE[0.5]"


def load(path):
    return {r["dataset"]: (float(r[CRPS]), float(r[MASE])) for r in csv.DictReader(open(path))}


def gmean_ratio(rows, base, keys):
    c = math.exp(sum(math.log(rows[k][0] / base[k][0]) for k in keys) / len(keys))
    m = math.exp(sum(math.log(rows[k][1] / base[k][1]) for k in keys) / len(keys))
    return c, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    base = load(os.path.join(HERE, "ref/leaderboard/Seasonal_Naive/all_results.csv"))
    rows = load(args.results)
    keys = sorted(k for k in rows if k in base)
    missing = sorted(set(base) - set(rows))
    c, m = gmean_ratio(rows, base, keys)
    print(f"configs={len(keys)}/97  CRPS={c:.4f}  MASE={m:.4f}" + (f"  (missing {len(missing)})" if missing else ""))
    if not args.ref:
        return
    ref = load(args.ref)
    keys = [k for k in keys if k in ref]
    rc, rm = gmean_ratio(ref, base, keys)
    print(f"ref on same configs: CRPS={rc:.4f}  MASE={rm:.4f}")
    d = sorted(((rows[k][0] / ref[k][0] - 1) * 100, k) for k in keys)
    wins = sum(x < -0.5 for x, _ in d)
    losses = sum(x > 0.5 for x, _ in d)
    print(f"CRPS vs ref per config: better>0.5% {wins}  worse>0.5% {losses}  within {len(d) - wins - losses}")
    for x, k in d[:5] + d[-5:]:
        print(f"  {x:+7.2f}%  {k}")


if __name__ == "__main__":
    main()
