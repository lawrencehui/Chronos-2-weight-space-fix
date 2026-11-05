"""Gated closed-form edit of t0-gamma's 90% quantile (the only quantile whose miss its own state predicts).

Usage: python q90_edit.py [--bake results/edit_q90.npz]
Head: q_18 (tau=0.9) = q_17 + softplus(raw_18). Edit: raw_18 += gamma * (w.h + b) at every forecast step,
h = decoder input at that forecast patch. Baked exactly as decoder.residual_layer rows t*21+18 (+gamma*w, bias +gamma*b).
Fit: linear quantile regression at tau=0.9 (IRLS) of the scaled-space q90 miss z, using the linearisation
d q90 ~= sigmoid(raw_18) * d raw_18; configs weighted equally. lambda and gamma by 2-fold CV over configs
within FIT; the held-out score on SELECT uses the exact softplus, i.e. exactly what the baked weights compute.
Captured arrays are cached in results/q90_parts.pkl (our own derived data) so refits take seconds.
"""

import argparse
import os
import pickle

import numpy as np
from t0 import T0Forecaster

from dev_eval import load_split
from edits import capture
from probe_edit import crps, loc_scale
from run_eval import MODEL_PATH

TAU, IDX, NQ = 0.9, 18, 21
CACHE = "results/q90_parts.pkl"


def softplus(x):
    return np.logaddexp(0, x)


def build(data, cap):
    out = {}
    for cfg, d in data.items():
        c = cap[cfg]
        pl, (n, hp) = d["pl"], c["dec"].shape[:2]
        ls = np.array([loc_scale(x) for x in d["ctx"]])
        loc, sc = ls[:, :1], ls[:, 1:]
        q90s = np.arcsinh((c["Q"][:, :, 8] - loc) / sc)
        z = np.arcsinh((d["label"] - loc) / sc) - q90s  # [n, pl] scaled-space miss of q90
        raw = c["raw"].reshape(n, hp * 32)[:, :pl].astype(np.float64)
        out[cfg] = dict(H=c["dec"], z=z, raw=raw, q90s=q90s, loc=loc, sc=sc, Q=c["Q"], y=d["label"].astype(np.float64),
                        nonneg=np.array([np.nanmin(x) >= 0 if np.any(~np.isnan(x)) else False for x in d["ctx"]]))
    return out


def design(parts, per_patch=4, seed=0):
    """Rows: (sigmoid(raw) * [h, 1], z) for up to per_patch random steps of every forecast patch."""
    rng = np.random.default_rng(seed)
    X, Y, W = [], [], []
    for p in parts:
        n, hp, D = p["H"].shape
        pl = p["z"].shape[1]
        rows_x, rows_y = [], []
        for j in range(hp):
            steps = np.arange(32 * j, min(32 * (j + 1), pl))
            if len(steps) == 0:
                continue
            pick = rng.choice(steps, size=min(per_patch, len(steps)), replace=False)
            for t in pick:
                s = 1 / (1 + np.exp(-p["raw"][:, t]))  # d softplus / d raw
                rows_x.append(np.hstack([p["H"][:, j].astype(np.float32), np.ones((n, 1), np.float32)]) * s[:, None].astype(np.float32))
                rows_y.append(p["z"][:, t])
        x, y = np.concatenate(rows_x), np.concatenate(rows_y)
        ok = ~np.isnan(y)
        X.append(x[ok]); Y.append(y[ok]); W.append(np.full(ok.sum(), 1.0 / max(ok.sum(), 1)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W)


def fit_qr(X, y, w0, lam, iters=12):
    """Linear quantile regression at TAU by IRLS: rho_tau(r) ~= c r^2 with c = (tau or 1-tau)/|r|."""
    w0 = w0 / w0.sum()
    reg = lam * np.eye(X.shape[1]); reg[-1, -1] = 0
    c = w0.copy()
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        Xc = X * c[:, None]
        beta = np.linalg.solve((Xc.T @ X).astype(np.float64) + reg, (Xc.T @ y).astype(np.float64))
        r = y - X @ beta
        c = w0 * np.where(r > 0, TAU, 1 - TAU) / np.maximum(np.abs(r), 1e-3)
        c /= c.sum()
    return beta[:-1], beta[-1]


def apply(p, coef, b, gamma):
    n, hp, _ = p["H"].shape
    pl = p["z"].shape[1]
    d = gamma * (p["H"].astype(np.float64) @ coef + b)  # [n, hp] raw shift per patch
    draw = np.repeat(d, 32, axis=1)[:, :pl]
    q90s = p["q90s"] + softplus(p["raw"] + draw) - softplus(p["raw"])  # exact head arithmetic
    Q = p["Q"].copy()
    Q[:, :, 8] = p["loc"] + p["sc"] * np.sinh(q90s)
    Q[p["nonneg"], :, 8] = np.maximum(Q[p["nonneg"], :, 8], 0)
    Q[:, :, 8] = np.maximum(Q[:, :, 8], Q[:, :, 7])  # the head keeps q90 >= q85 >= q80 by construction
    return Q


def score(parts, coef, b, gamma):
    lc, cov0, cov1 = [], [], []
    for p in parts:
        Q = apply(p, coef, b, gamma)
        lc.append(np.log(crps(p["y"], Q) / crps(p["y"], p["Q"])))
        cov0.append(np.nanmean(p["y"] < p["Q"][:, :, 8])); cov1.append(np.nanmean(p["y"] < Q[:, :, 8]))
    lc = np.array(lc)
    return np.exp(lc.mean()), int((lc < -.005).sum()), int((lc > .005).sum()), len(lc), np.median(cov0), np.median(cov1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=None)
    args = ap.parse_args()
    model = T0Forecaster.from_pretrained(MODEL_PATH).to("mps").eval()
    if os.path.exists(CACHE):
        fit, sel = pickle.load(open(CACHE, "rb"))
    else:
        fitd, seld = load_split("fit"), load_split("select")
        fit = build(fitd, capture(model, fitd, neuron_layers=[], dec=True, raw_q=IDX)[0])
        sel = build(seld, capture(model, seld, neuron_layers=[], dec=True, raw_q=IDX)[0])
        pickle.dump((fit, sel), open(CACHE, "wb"))
    names = sorted(fit)
    halves = [[fit[c] for c in names[0::2]], [fit[c] for c in names[1::2]]]
    lams, gammas = [1e-3, 1e-2, 1e-1], [0.25, 0.5, 1.0]
    cv = {}
    for lam in lams:
        fits = [fit_qr(*design(h), lam) for h in halves]
        for g in gammas:
            cv[(lam, g)] = float(np.exp(np.mean([np.log(score(halves[1 - i], *fits[i], g)[0]) for i in range(2)])))
        print(f"  CV lambda={lam:g}: " + "  ".join(f"g={g}:x{cv[(lam, g)]:.4f}" for g in gammas), flush=True)
    lam, g = min(cv, key=cv.get)
    coef, b = fit_qr(*design([fit[c] for c in names]), lam)
    print(f"chosen lambda={lam:g} gamma={g}  |w|={np.linalg.norm(coef):.3f}  b={b:+.3f}")
    for name, parts in (("FIT (in-sample)", [fit[c] for c in names]), ("SELECT (held out)", list(sel.values()))):
        c, w, l, n, c0, c1 = score(parts, coef, b, g)
        print(f"{name}: CRPS x{c:.4f}  better {w} / worse {l} of {n}  | median share below q90: {c0:.3f} -> {c1:.3f} (target 0.900)")
    if args.bake:
        rl = model.decoder.residual_layer
        W, B = rl.weight.detach().cpu().numpy().copy(), rl.bias.detach().cpu().numpy().copy()
        rows = np.arange(32) * NQ + IDX
        W[rows] += np.float32(g) * coef[None, :].astype(np.float32)
        B[rows] += np.float32(g * b)
        np.savez(args.bake, **{"decoder.residual_layer.weight": W, "decoder.residual_layer.bias": B})
        print(f"baked -> {args.bake}")


if __name__ == "__main__":
    main()
