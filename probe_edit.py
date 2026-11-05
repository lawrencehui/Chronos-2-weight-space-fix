"""Closed-form readout edit: teach t0-gamma's decoder to correct its own predictable error.

Usage: python probe_edit.py [--bake results/edit_probe.npz]
1. capture the decoder input h at every forecast position, on FIT and SELECT windows (fast mode)
2. target z = arcsinh((y - loc)/scale) - arcsinh((median - loc)/scale): the model's own median error in its
   internal (scaled) space, averaged over each forecast patch
3. ridge regression z ~ w.h + b, configs weighted equally (the leaderboard averages configs); lambda and a
   shrink gamma are chosen by 2-fold cross-validation over configs WITHIN the fit windows
4. score on SELECT: every quantile moves by delta = gamma (w.h + b) in scaled space - exactly what the baked edit does:
   decoder.residual_layer rows (t*21 + 0), t = 0..31, get +gamma*w and bias +gamma*b (quantile 0 is the head's
   base; the other 20 are cumulative increments on top of it, so all 21 shift together)
"""

import argparse

import numpy as np
import torch
from t0 import T0Forecaster

from dev_eval import load_split
from edits import capture
from run_eval import MODEL_PATH

TAUS = np.arange(1, 10) / 10
EPS = 0.01  # t0-gamma scaler_eps, std_clamp mode


def loc_scale(ctx, n=2048):
    c = ctx[-n:].astype(np.float64)
    c = c[~np.isnan(c)]
    if len(c) == 0:
        return 0.0, 1.0
    return float(c.mean()), max(float(c.std(ddof=1)) if len(c) > 1 else 0.0, EPS)


def build(data, cap):
    """Per config: features [patches, 1024], targets [patches], and what's needed to apply a shift."""
    out = {}
    for cfg, d in data.items():
        c = cap[cfg]
        H, Q, pl = c["dec"].astype(np.float32), c["Q"], d["pl"]
        n, hp = H.shape[:2]
        ls = np.array([loc_scale(x) for x in d["ctx"]])  # [n, 2]
        loc, sc = ls[:, :1], ls[:, 1:]
        z = np.arcsinh((d["label"] - loc) / sc) - np.arcsinh((Q[:, :, 4] - loc) / sc)  # [n, pl]
        zp = np.full((n, hp), np.nan)
        for p in range(hp):
            seg = z[:, 32 * p:min(32 * (p + 1), pl)]
            if seg.shape[1]:
                zp[:, p] = np.nanmean(seg, 1)
        out[cfg] = dict(H=H, z=np.clip(zp, -3, 3), loc=loc, sc=sc, Q=Q, y=d["label"].astype(np.float64),
                        nonneg=np.array([np.nanmin(x) >= 0 if np.any(~np.isnan(x)) else False for x in d["ctx"]]))
    return out


LOSS = "l1"


def fit_ridge(parts, lam, iters=15):
    """Weighted ridge (LOSS='l2') or median regression by IRLS (LOSS='l1', the right target for CRPS/MASE)."""
    X = np.concatenate([p["H"].reshape(-1, p["H"].shape[-1]) for p in parts])
    y = np.concatenate([p["z"].reshape(-1) for p in parts])
    w0 = np.concatenate([np.full(p["z"].size, 1.0 / p["z"].size) for p in parts])
    ok = ~np.isnan(y)
    X, y, w0 = X[ok].astype(np.float64), y[ok], w0[ok] / w0[ok].sum()
    Xa = np.hstack([X, np.ones((len(X), 1))])
    reg = lam * np.eye(Xa.shape[1]); reg[-1, -1] = 0
    w = w0.copy()
    for _ in range(iters if LOSS == "l1" else 1):
        beta = np.linalg.solve((Xa * w[:, None]).T @ Xa + reg, (Xa * w[:, None]).T @ y)
        w = w0 / np.maximum(np.abs(y - Xa @ beta), 1e-3)
        w /= w.sum()
    return beta[:-1], beta[-1]


def crps(y, Q):
    y, Q = y.reshape(-1), Q.reshape(-1, Q.shape[-1])
    ok = ~np.isnan(y)
    y, Q = y[ok], Q[ok]
    return float(np.mean((2 * np.abs((y[:, None] - Q) * ((y[:, None] <= Q) - TAUS))).sum(0) / np.abs(y).sum()))


def mase_num(y, med):
    return np.nanmean(np.abs(y - med))


def apply_shift(p, coef, b, gamma):
    n, hp, _ = p["H"].shape
    pl = p["Q"].shape[1]
    delta = gamma * (p["H"].astype(np.float64) @ coef + b)  # [n, hp]
    dstep = np.repeat(delta, 32, axis=1)[:, :pl, None]
    loc, sc = p["loc"][:, :, None], p["sc"][:, :, None]
    Q = loc + sc * np.sinh(np.arcsinh((p["Q"] - loc) / sc) + dstep)
    Q[p["nonneg"]] = np.maximum(Q[p["nonneg"]], 0)
    return Q


def score(parts, coef, b, gamma):
    lc, lm = [], []
    for p in parts:
        Q = apply_shift(p, coef, b, gamma)
        lc.append(np.log(crps(p["y"], Q) / crps(p["y"], p["Q"])))
        lm.append(np.log(mase_num(p["y"], Q[:, :, 4]) / mase_num(p["y"], p["Q"][:, :, 4])))
    lc = np.array(lc)
    return np.exp(lc.mean()), np.exp(np.mean(lm)), int((lc < -.005).sum()), int((lc > .005).sum()), len(lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=None, help="write the chosen edit as replaced decoder tensors")
    ap.add_argument("--loss", default="l1", choices=["l1", "l2"])
    args = ap.parse_args()
    global LOSS
    LOSS = args.loss
    model = T0Forecaster.from_pretrained(MODEL_PATH).to("mps").eval()
    import os
    import pickle
    cache = "results/probe_parts.pkl"  # our own derived arrays, written by this script
    if os.path.exists(cache):
        fit, sel = pickle.load(open(cache, "rb"))
    else:
        fitd, seld = load_split("fit"), load_split("select")
        fit = build(fitd, capture(model, fitd, neuron_layers=[], dec=True)[0])
        sel = build(seld, capture(model, seld, neuron_layers=[], dec=True)[0])
        pickle.dump((fit, sel), open(cache, "wb"))
    names = sorted(fit)
    halves = [[fit[c] for c in names[0::2]], [fit[c] for c in names[1::2]]]

    lams, gammas = [1e-3, 1e-2, 1e-1, 1.0], [0.25, 0.5, 0.75, 1.0]
    cv = {}
    for lam in lams:
        fits = [fit_ridge(h, lam) for h in halves]
        for g in gammas:
            r = [score(halves[1 - i], *fits[i], g)[0] for i in range(2)]
            cv[(lam, g)] = float(np.exp(np.mean(np.log(r))))
    print("2-fold CV over configs within FIT (CRPS ratio):")
    for lam in lams:
        print(f"  lambda={lam:g}: " + "  ".join(f"g={g}:x{cv[(lam, g)]:.4f}" for g in gammas))
    lam, g = min(cv, key=cv.get)
    coef, b = fit_ridge([fit[c] for c in names], lam)
    print(f"chosen lambda={lam:g} gamma={g}  |w|={np.linalg.norm(coef):.3f} b={b:+.4f}")
    c, m, w, l, n = score([fit[c] for c in names], coef, b, g)
    print(f"FIT (in-sample):   CRPS x{c:.4f}  MASE x{m:.4f}  better {w} / worse {l} of {n}")
    c, m, w, l, n = score(list(sel.values()), coef, b, g)
    print(f"SELECT (held out): CRPS x{c:.4f}  MASE x{m:.4f}  better {w} / worse {l} of {n}")

    if args.bake:
        rl = model.decoder.residual_layer
        W, B = rl.weight.detach().cpu().numpy().copy(), rl.bias.detach().cpu().numpy().copy()
        rows = np.arange(32) * model.head.n_quantiles  # quantile index 0 of each of the 32 steps
        W[rows] += g * coef[None, :].astype(np.float32)
        B[rows] += np.float32(g * b)
        np.savez(args.bake, **{"decoder.residual_layer.weight": W, "decoder.residual_layer.bias": B})
        print(f"baked -> {args.bake}  (rows {rows[:3]}... of decoder.residual_layer)")


if __name__ == "__main__":
    main()
