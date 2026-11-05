"""Freeze a small, fixed dev set of fit / select windows to disk, so each experiment runs in seconds.

Usage: python dev_cache.py [--series 24] [--windows 3]
Writes data/dev/<split>/<config>.npz for split in (fit, select): per config, up to --series series
(fixed random sample) x up to --windows of that split's windows (see splits.py; none overlap the test).
Each file holds contexts (last 8192 points, flattened + lengths), labels, item/window ids, seasonality.
The test split is never cached: it is only run once, with run_eval.py, on the frozen edit.
"""

import argparse
import os
import time

import numpy as np
from gluonts.time_feature import get_seasonality

from run_eval import CONTEXT_LENGTH, configs, load_dataset
from splits import backtest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "dev")


def fname(cfg):
    return cfg.replace("/", "__") + ".npz"


def cache(ds, which, n_series, n_windows):
    data = backtest(ds, which, max_series=n_series, seed=0)
    if data is None:
        return None
    ctx, lab, item, win, seen, ids = [], [], [], [], {}, {}
    for inp, l in zip(data.input, data.label):
        # Windows come out in time order per item; keep the last n_windows (closest to the test).
        k = seen.get(inp["item_id"], 0)
        seen[inp["item_id"]] = k + 1
        ctx.append(np.asarray(inp["target"], dtype=np.float32)[-CONTEXT_LENGTH:])
        lab.append(np.asarray(l["target"], dtype=np.float32))
        item.append(ids.setdefault(inp["item_id"], len(ids)))
        win.append(k)
    drop = max(seen.values()) - n_windows  # every item has the same window count
    keep = [k >= drop for k in win]
    ctx, lab, item = ([x for x, k in zip(xs, keep) if k] for xs in (ctx, lab, item))
    win = [k - max(drop, 0) for k in win if k >= drop]
    lens = np.array([len(c) for c in ctx])
    return dict(ctx=np.concatenate(ctx), ctx_len=lens, label=np.stack(lab), item=np.array(item),
                window=np.array(win), m=np.array(get_seasonality(ds.freq)), pl=np.array(ds.prediction_length))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=24)
    ap.add_argument("--windows", type=int, default=3)
    args = ap.parse_args()
    for which in ("fit", "select"):
        os.makedirs(os.path.join(OUT, which), exist_ok=True)
    t0 = time.monotonic()
    for ds_name, term, cfg, _ in configs():
        ds = load_dataset(ds_name, term)
        for which in ("fit", "select"):
            path = os.path.join(OUT, which, fname(cfg))
            if os.path.exists(path):
                continue
            d = cache(ds, which, args.series, args.windows)
            if d is None:
                print(f"{cfg} {which}: no windows", flush=True)
                continue
            np.savez(path, **d)
            print(f"{cfg} {which}: {len(d['ctx_len'])} windows, pl={int(d['pl'])}, median ctx={int(np.median(d['ctx_len']))}", flush=True)
    print(f"done in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
