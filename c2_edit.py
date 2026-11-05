"""Closed-form gated readout edit for Chronos-2: each output quantile corrects its own predictable miss.

Usage: python c2_edit.py [--bake results/c2_edit.npz] [--verify]
Chronos-2's head outputs every quantile directly (no cumulative chain), so for requested quantile k:
  raw_k(p) += gamma * (w_k . h + b_k)  at every step p of a forecast patch, h = head input of that patch
is exactly rows (k_index*16 + p) of output_patch_embedding.residual_layer (+gamma*w_k, bias +gamma*b_k).
w_k, b_k: linear quantile regression at tau_k (IRLS) of z_k = arcsinh((y-loc)/scale) - raw_k, fit on FIT windows,
configs weighted equally. lambda and gamma by 2-fold CV over configs within FIT. Scored on SELECT; --verify
reruns the real edited model on SELECT to confirm the weights reproduce the estimate.
"""

import argparse

import numpy as np
import torch

from chronos_dev import LEVELS, NQ, PATCH, cached, load_pipe, loc_scale, ratio, run, scores
from dev_eval import load_split

TAUS = np.array(LEVELS)


def build(data, res, ctx=2048):
    parts = {}
    for cfg, d in data.items():
        r = res[cfg]
        n, nop = r["dec"].shape[:2]
        pl = d["pl"]
        ls = np.array([loc_scale(c[-ctx:]) for c in d["ctx"]])
        ys = np.arcsinh((d["label"] - ls[:, :1]) / ls[:, 1:])  # [n, pl] truth in the model's scaled space
        raw = r["raw"].transpose(0, 1, 3, 2).reshape(n, nop * PATCH, 9)[:, :pl].astype(np.float64)  # [n, pl, 9]
        parts[cfg] = dict(H=r["dec"], ys=ys, raw=raw, loc=ls[:, :1], sc=ls[:, 1:], Q=r["Q"], y=d["label"].astype(np.float64))
    return parts


def check_mapping(parts):
    errs = [np.nanmax(np.abs(p["loc"][:, :, None] + p["sc"][:, :, None] * np.sinh(p["raw"]) - p["Q"]) / (np.abs(p["Q"]) + 1)) for p in parts.values()]
    print(f"mapping check: max relative |loc + scale*sinh(raw) - forecast| = {max(errs):.2e}")


def design(parts, k, seed=0):
    rng = np.random.default_rng(seed + k)
    X, Y, W = [], [], []
    for p in parts:
        n, nop, D = p["H"].shape
        pl = p["ys"].shape[1]
        xs, ys = [], []
        for j in range(nop):
            steps = np.arange(PATCH * j, min(PATCH * (j + 1), pl))
            if len(steps) == 0:
                continue
            t = rng.choice(steps)
            xs.append(np.hstack([p["H"][:, j].astype(np.float32), np.ones((n, 1), np.float32)]))
            ys.append(p["ys"][:, t] - p["raw"][:, t, k])
        x, y = np.concatenate(xs), np.concatenate(ys)
        ok = ~np.isnan(y)
        X.append(x[ok]); Y.append(y[ok]); W.append(np.full(ok.sum(), 1.0 / max(ok.sum(), 1)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W)


def fit_qr(X, y, w0, tau, lam, iters=8):
    w0 = w0 / w0.sum()
    reg = lam * np.eye(X.shape[1]); reg[-1, -1] = 0
    c, beta = w0.copy(), np.zeros(X.shape[1])
    for _ in range(iters):
        Xc = X * c[:, None].astype(np.float32)
        beta = np.linalg.solve((Xc.T @ X).astype(np.float64) + reg, (Xc.T @ y.astype(np.float32)).astype(np.float64))
        r = y - X @ beta
        c = w0 * np.where(r > 0, tau, 1 - tau) / np.maximum(np.abs(r), 1e-3)
        c /= c.sum()
    return beta


def designs(parts):
    return [design(parts, k) for k in range(9)]


def fit_all(ds, lam):
    """ds: designs(parts), computed once per set of parts (they don't depend on lambda)."""
    return [fit_qr(*ds[k], TAUS[k], lam) for k in range(9)]


def apply(p, betas, gamma):
    n, nop, D = p["H"].shape
    pl = p["raw"].shape[1]
    Ha = np.concatenate([p["H"].astype(np.float64), np.ones((n, nop, 1))], -1)
    raw = p["raw"].copy()
    for k, b in enumerate(betas):
        raw[:, :, k] += gamma * np.repeat(Ha @ b, PATCH, axis=1)[:, :pl]
    return p["loc"][:, :, None] + p["sc"][:, :, None] * np.sinh(raw)


def crps(y, Q):
    y, Q = y.reshape(-1), Q.reshape(-1, 9)
    ok = ~np.isnan(y)
    y, Q = y[ok], Q[ok]
    return float(np.mean((2 * np.abs((y[:, None] - Q) * ((y[:, None] <= Q) - TAUS))).sum(0) / np.abs(y).sum()))


def score(parts, betas, gamma):
    lc, lm = [], []
    for p in parts:
        Q = apply(p, betas, gamma)
        lc.append(np.log(crps(p["y"], Q) / crps(p["y"], p["Q"])))
        lm.append(np.log(np.nanmean(np.abs(p["y"] - Q[:, :, 4])) / np.nanmean(np.abs(p["y"] - p["Q"][:, :, 4]))))
    lc = np.array(lc)
    return np.exp(lc.mean()), np.exp(np.mean(lm)), int((lc < -.005).sum()), int((lc > .005).sum()), len(lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--lams", nargs="+", type=float, default=[1e-3, 1e-1])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    args = ap.parse_args()
    fitd, seld = load_split("fit"), load_split("select")
    fit, sel = build(fitd, cached("fit")), build(seld, cached("select"))
    check_mapping(sel)
    names = sorted(fit)
    halves = [[fit[c] for c in names[0::2]], [fit[c] for c in names[1::2]]]
    cv = {}
    hd = [designs(h) for h in halves]
    for lam in args.lams:
        fits = [fit_all(hd[i], lam) for i in range(2)]
        for g in args.gammas:
            cv[(lam, g)] = float(np.exp(np.mean([np.log(score(halves[1 - i], fits[i], g)[0]) for i in range(2)])))
        print(f"  CV lambda={lam:g}: " + "  ".join(f"g={g}:x{cv[(lam, g)]:.4f}" for g in args.gammas), flush=True)
    lam, g = min(cv, key=cv.get)
    betas = fit_all(designs([fit[c] for c in names]), lam)
    print(f"chosen lambda={lam:g} gamma={g}")
    for name, parts in (("FIT (in-sample)", [fit[c] for c in names]), ("SELECT (held out)", list(sel.values()))):
        c, m, w, l, n = score(parts, betas, g)
        print(f"{name}: CRPS x{c:.4f}  MASE x{m:.4f}  better {w} / worse {l} of {n}", flush=True)

    if args.bake or args.verify:
        pipe = load_pipe()
        rl = pipe.model.output_patch_embedding.residual_layer
        qidx = [pipe.model.chronos_config.quantiles.index(q) for q in LEVELS]
        with torch.no_grad():
            for k, b in enumerate(betas):
                rows = torch.arange(PATCH) + qidx[k] * PATCH  # layout "(q p)": quantile-major
                rl.weight[rows] += torch.as_tensor(g * b[:-1], dtype=rl.weight.dtype, device=rl.weight.device)[None, :]
                rl.bias[rows] += float(g * b[-1])
        if args.bake:
            np.savez(args.bake, **{"output_patch_embedding.residual_layer.weight": rl.weight.detach().cpu().numpy(),
                                   "output_patch_embedding.residual_layer.bias": rl.bias.detach().cpu().numpy()})
            print(f"baked -> {args.bake}")
        if args.verify:
            res = run(pipe, seld, capture=False)
            base = {c: v for c, v in scores(seld, cached("select")).items()}
            print("REAL edited model on SELECT:", ratio(scores(seld, res), base))


if __name__ == "__main__":
    main()
