"""Fast evaluation of t0-gamma (optionally edited) on the frozen dev windows (dev_cache.py).

Usage: python dev_eval.py <fit|select> <out.csv> [--edit edit.npz] [--base base.csv] [--only cfg-substring ...]
Same predictor settings as the leaderboard run. Per config and horizon third it writes
crps, mase, bias, below_q10/50/90, width80 (see diagnose.py). With --base, prints the
leaderboard-style change: geomean over configs of CRPS_edit / CRPS_base, plus wins and losses.
Also importable: load_split(), evaluate(model, split) for in-process sweeps.
"""

import argparse
import csv
import glob
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from t0 import T0Forecaster

from diagnose import stats
from run_eval import CONTEXT_LENGTH, MODEL_PATH, make_predictor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "data", "dev")
COLS = ["config", "part", "n", "crps", "mase", "bias", "below_q10", "below_q50", "below_q90", "width80"]
START = pd.Period("2000-01-01", freq="D")  # only used to stamp forecast objects; metrics don't read it


def load_split(split, only=None):
    out = {}
    for path in sorted(glob.glob(os.path.join(DEV, split, "*.npz"))):
        cfg = os.path.basename(path)[:-4].replace("__", "/")
        if only and not any(s in cfg for s in only):
            continue
        z = np.load(path)
        lens = z["ctx_len"]
        ctx = np.split(z["ctx"], np.cumsum(lens)[:-1])
        out[cfg] = dict(ctx=ctx, label=z["label"], item=z["item"], window=z["window"], m=int(z["m"]), pl=int(z["pl"]))
    return out


def seasonal_errors(ctx, m):
    out = []
    for c in ctx:
        c = c[~np.isnan(c)].astype(np.float64)
        mm = m if len(c) > m else 1
        e = np.mean(np.abs(c[mm:] - c[:-mm])) if len(c) > mm else np.nan
        out.append(e if e > 0 else np.nan)
    return np.array(out)


def predict(model, d, fast=False):
    """Quantiles [windows, pl, 9] in cache order.

    T0Predictor groups series of the same window (an item's occurrence count) for cross-inference,
    so entries go in item-major, time-ordered, exactly as GIFT-Eval's generate_instances yields them.
    """
    idx = np.lexsort((d["window"], d["item"]))
    entries = [{"target": d["ctx"][i], "start": START, "item_id": f"s{d['item'][i]}"} for i in idx]
    fcs = make_predictor(model, d["pl"], fast).predict(entries)
    Q = np.empty((len(idx), d["pl"], 9), dtype=np.float64)
    for j, fc in zip(idx, fcs):
        Q[j] = fc.forecast_array.T
    return Q


def evaluate(model, split_data, parts=("all",), fast=False, keep_q=None):
    """Per-config stats; if keep_q is a dict, the quantiles are stored in it by config."""
    res = {}
    for cfg, d in split_data.items():
        Q = predict(model, d, fast)
        if keep_q is not None:
            keep_q[cfg] = Q.astype(np.float32)
        pl, n = d["pl"], len(d["ctx"])
        y = d["label"].astype(np.float64).reshape(-1)
        s = np.repeat(seasonal_errors(d["ctx"], d["m"]), pl)
        h = np.tile(np.arange(pl), n)
        Qf = Q.reshape(-1, 9)
        sel = {"all": h >= 0, "early": h < pl / 3, "mid": (h >= pl / 3) & (h < 2 * pl / 3), "late": h >= 2 * pl / 3}
        res[cfg] = {p: stats(y[sel[p]], Qf[sel[p]], s[sel[p]]) for p in parts if sel[p].any()}
    return res


def compare(res, base_csv):
    base = {(r["config"]): float(r["crps"]) for r in csv.DictReader(open(base_csv)) if r["part"] == "all"}
    basem = {(r["config"]): float(r["mase"]) for r in csv.DictReader(open(base_csv)) if r["part"] == "all"}
    keys = [k for k in res if k in base]
    rc = [res[k]["all"]["crps"] / base[k] for k in keys]
    rm = [res[k]["all"]["mase"] / basem[k] for k in keys]
    g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    wins, losses = sum(r < 0.995 for r in rc), sum(r > 1.005 for r in rc)
    return dict(crps_ratio=g(rc), mase_ratio=g(rm), wins=wins, losses=losses, n=len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["fit", "select"])
    ap.add_argument("out")
    ap.add_argument("--edit", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--fast", action="store_true", help="2048 context, no mirrored pass (dev iteration)")
    ap.add_argument("--save-q", action="store_true", help="also save quantiles to <out>.q.npz")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    model = T0Forecaster.from_pretrained(MODEL_PATH).to(args.device).eval()
    if args.edit:
        from edits import load_edit
        load_edit(model, args.edit)
    data = load_split(args.split, args.only)
    t = time.monotonic()
    qs = {} if args.save_q else None
    res = evaluate(model, data, parts=("all", "early", "mid", "late"), fast=args.fast, keep_q=qs)
    if qs:
        np.savez(args.out[:-4] + ".q.npz", **{k.replace("/", "__"): v for k, v in qs.items()})
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for cfg, parts in res.items():
            for p, st in parts.items():
                w.writerow([cfg, p] + [st[c] for c in COLS[2:]])
    print(f"{len(res)} configs in {time.monotonic() - t:.0f}s -> {args.out}")
    if args.base:
        c = compare(res, args.base)
        print(f"vs base: CRPS x{c['crps_ratio']:.4f}  MASE x{c['mase_ratio']:.4f}  better {c['wins']} / worse {c['losses']} of {c['n']}")


if __name__ == "__main__":
    main()
