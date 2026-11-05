"""Weight extrapolation between t0 releases: theta = gamma + lam * (gamma - beta).

Usage: python extrapolate.py <fit|select> <lam> [<lam> ...] [--full]
lam > 0 continues past gamma along the beta->gamma update; lam < 0 interpolates back toward beta
(lam = -1 is beta itself). No training, no data: the only "fit" is choosing lam on the FIT windows.
Prints the leaderboard-style change vs gamma on the same dev windows (geomean CRPS ratio, wins/losses).
"""

import argparse
import time

import numpy as np
import torch
from t0 import T0Forecaster

from dev_eval import evaluate, load_split
from run_eval import MODEL_PATH

BETA = "theforecastingcompany/t0-beta"


def ratio_stats(res, base):
    logs = np.array([np.log(res[k]["all"]["crps"] / base[k]["all"]["crps"]) for k in base])
    mlogs = np.array([np.log(res[k]["all"]["mase"] / base[k]["all"]["mase"]) for k in base])
    return np.exp(logs.mean()), np.exp(mlogs.mean()), int((logs < -0.005).sum()), int((logs > 0.005).sum()), len(logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["fit", "select"])
    ap.add_argument("lams", nargs="+", type=float)
    ap.add_argument("--full", action="store_true", help="leaderboard settings instead of fast mode")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    fast = not args.full

    model = T0Forecaster.from_pretrained(MODEL_PATH).to(dev).eval()
    beta = T0Forecaster.from_pretrained(BETA).to(dev).eval()
    g = {k: v.detach().clone() for k, v in model.state_dict().items()}
    b = beta.state_dict()
    assert g.keys() == b.keys() and all(g[k].shape == b[k].shape for k in g), "architectures differ"
    rel = torch.sqrt(sum(((g[k] - b[k]).float() ** 2).sum() for k in g if g[k].is_floating_point())) / torch.sqrt(
        sum((g[k].float() ** 2).sum() for k in g if g[k].is_floating_point()))
    print(f"|gamma - beta| / |gamma| = {rel:.4f}  (small = gamma is a continuation of beta, extrapolation is meaningful)")
    del beta

    data = load_split(args.split)
    t = time.monotonic()
    base = evaluate(model, data, fast=fast)
    print(f"gamma baseline: {time.monotonic() - t:.0f}s")
    for lam in args.lams:
        with torch.no_grad():
            model.load_state_dict({k: (g[k] + lam * (g[k] - b[k])) if g[k].is_floating_point() else g[k] for k in g})
        res = evaluate(model, data, fast=fast)
        c, m, w, l, n = ratio_stats(res, base)
        print(f"lam={lam:+.2f}: CRPS x{c:.4f}  MASE x{m:.4f}  better {w} / worse {l} of {n}", flush=True)
    model.load_state_dict(g)


if __name__ == "__main__":
    main()
