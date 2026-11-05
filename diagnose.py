"""Where is the base model systematically wrong? Measured on the FIT windows only.

Usage: python diagnose.py <out.csv> [--split fit|select|test] [--max-series 50] [--only ds ...]
Per config and per horizon third (early/mid/late), reports:
  crps, mase          same formulas as the leaderboard (checked against gluonts on the test split)
  bias                mean (median - truth) / seasonal_error   (>0 = forecasts too high)
  below_q{10,50,90}   share of truths below that quantile      (calibrated = 0.1, 0.5, 0.9)
  width80             mean (q90 - q10) / seasonal_error
A systematic error worth editing shows up as a consistent sign across many configs of one domain/frequency.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from t0 import T0Forecaster

from run_eval import MODEL_PATH, PROPS, QUANTILE_LEVELS, configs, load_dataset, make_predictor
from splits import backtest

COLS = ["config", "domain", "freq", "term", "part", "n", "crps", "mase", "bias", "below_q10", "below_q50", "below_q90", "width80"]


def seasonal_error(past, m):
    past = past[~np.isnan(past)]
    if len(past) <= m:
        m = 1
    if len(past) < 2:
        return np.nan
    e = np.mean(np.abs(past[m:] - past[:-m]))
    return e if e > 0 else np.nan


def collect(predictor, data, m):
    """Arrays over all (window, step): truth y, quantiles Q [.., 9], seasonal error s, step index h."""
    Y, Q, S, H = [], [], [], []
    for inp, lab, fc in zip(data.input, data.label, predictor.predict(data.input)):
        y = np.asarray(lab["target"], dtype=np.float64)
        q = np.stack([np.asarray(fc.quantile(t), dtype=np.float64) for t in QUANTILE_LEVELS], axis=-1)
        s = seasonal_error(np.asarray(inp["target"], dtype=np.float64), m)
        Y.append(y); Q.append(q); S.append(np.full(len(y), s)); H.append(np.arange(len(y)))
    return np.concatenate(Y), np.concatenate(Q), np.concatenate(S), np.concatenate(H)


def stats(y, Q, s):
    ok = ~np.isnan(y)
    y, Q, s = y[ok], Q[ok], s[ok]
    taus = np.array(QUANTILE_LEVELS)
    ql = 2 * np.abs((y[:, None] - Q) * ((y[:, None] <= Q) - taus))
    crps = float(np.mean(ql.sum(0) / np.abs(y).sum()))
    med = Q[:, 4]
    ms = ~np.isnan(s)
    mase = float(np.mean(np.abs(y[ms] - med[ms]) / s[ms]))
    bias = float(np.mean((med[ms] - y[ms]) / s[ms]))
    width = float(np.mean((Q[ms, 8] - Q[ms, 0]) / s[ms]))
    return dict(n=int(ok.sum()), crps=crps, mase=mase, bias=bias, below_q10=float(np.mean(y < Q[:, 0])),
                below_q50=float(np.mean(y < Q[:, 4])), below_q90=float(np.mean(y < Q[:, 8])), width80=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--split", default="fit", choices=["fit", "select", "test"])
    ap.add_argument("--max-series", type=int, default=50)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    from gluonts.time_feature import get_seasonality

    done = set()
    if os.path.exists(args.out):
        done = {r["config"] for r in csv.DictReader(open(args.out))}
    else:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerow(COLS)

    model = T0Forecaster.from_pretrained(MODEL_PATH).to(args.device).eval()
    for ds_name, term, cfg, key in configs():
        if cfg in done or (args.only and ds_name not in args.only):
            continue
        ds = load_dataset(ds_name, term)
        data = ds.test_data if args.split == "test" else backtest(ds, args.split, args.max_series)
        if data is None:
            print(f"{cfg}: no windows long enough", flush=True)
            continue
        t = time.monotonic()
        y, Q, s, h = collect(make_predictor(model, ds.prediction_length), data, get_seasonality(ds.freq))
        pl = ds.prediction_length
        parts = {"all": np.ones_like(h, bool), "early": h < pl / 3, "mid": (h >= pl / 3) & (h < 2 * pl / 3), "late": h >= 2 * pl / 3}
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            for part, sel in parts.items():
                if sel.sum() == 0:
                    continue
                st = stats(y[sel], Q[sel], s[sel])
                w.writerow([cfg, PROPS[key]["domain"], ds.freq, term, part] + [st[c] for c in COLS[5:]])
        st = stats(y, Q, s)
        print(f"{cfg}: crps={st['crps']:.4f} mase={st['mase']:.3f} bias={st['bias']:+.3f} "
              f"below q10/50/90={st['below_q10']:.2f}/{st['below_q50']:.2f}/{st['below_q90']:.2f} "
              f"({time.monotonic() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
