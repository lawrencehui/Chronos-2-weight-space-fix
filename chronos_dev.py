"""Chronos-2 on the frozen GIFT-Eval dev windows: base eval, activation capture, and the error-predictability screen.

Usage:
  python chronos_dev.py base   [--ctx 2048]      # base CRPS/MASE on fit and select; caches forecasts + activations
  python chronos_dev.py screen                   # can any block's state predict the signed q10/q50/q90 miss? (fit -> select R^2)
Dev windows are univariate (multivariate datasets split per variate), forecast independently; fine for
comparing an edit against the base on identical windows, not identical to the leaderboard's multivariate run.
Caches (our own derived arrays): results/c2_{fit,select}.pkl
"""

import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from chronos import Chronos2Pipeline

from dev_eval import load_split, seasonal_errors
from diagnose import stats

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCH, NQ = 16, 21


def load_pipe():
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")


def loc_scale(ctx):
    c = ctx[~np.isnan(ctx)].astype(np.float64)
    if len(c) == 0:
        return 0.0, 1.0
    loc = c.mean()
    sc = np.sqrt(np.mean((c - loc) ** 2))
    return float(loc), float(sc) if sc > 0 else 1e-5


@torch.no_grad()
def run(pipe, data, ctx=2048, capture=False):
    """Forecasts [n, pl, 9] per config; with capture, per-block window-mean states [n, 12, 768],
    per-patch decoder input [n, nop, 768] and the raw head output for the 9 requested quantiles [n, nop, 9, 16]."""
    m = pipe.model
    st = {"nop": 1}
    blocks, dec, raw, hooks = {}, [], [], []
    qidx = [m.chronos_config.quantiles.index(q) for q in LEVELS]
    if capture:
        def fwin(x):
            x = x if torch.is_tensor(x) else x[0]  # blocks return a ModelOutput whose first field is the hidden states
            return x[:, -st["nop"]:].float().mean(1).cpu().numpy().astype(np.float16)
        for i, b in enumerate(m.encoder.block):
            hooks.append(b.register_forward_hook(lambda mod, inp, out, i=i: blocks.setdefault(i, []).append(fwin(out))))
        def head_hook(mod, inp, out):  # must return None, or PyTorch replaces the layer's output
            dec.append(inp[0].float().cpu().numpy().astype(np.float16))
            raw.append(out.float().cpu().numpy().reshape(out.shape[0], out.shape[1], NQ, PATCH)[:, :, qidx])
        hooks.append(m.output_patch_embedding.register_forward_hook(head_hook))
    out = {}
    try:
        for cfg, d in data.items():
            blocks.clear(); dec.clear(); raw.clear()
            st["nop"] = math.ceil(d["pl"] / PATCH)
            inputs = [{"target": c[-ctx:].astype(np.float32)} for c in d["ctx"]]
            q, _ = pipe.predict_quantiles(inputs, prediction_length=d["pl"], quantile_levels=LEVELS, batch_size=256)
            Q = torch.stack([x[0] for x in q]).cpu().numpy().astype(np.float64)  # [n, pl, 9]
            o = dict(Q=Q)
            if capture:
                o["blocks"] = np.stack([np.concatenate(blocks[i]) for i in range(len(m.encoder.block))], 1)
                o["dec"] = np.concatenate(dec)
                o["raw"] = np.concatenate(raw)
                assert len(o["dec"]) == len(Q), (cfg, len(o["dec"]), len(Q))
            out[cfg] = o
    finally:
        for h in hooks:
            h.remove()
    return out


def scores(data, res):
    out = {}
    for cfg, d in data.items():
        pl, n = d["pl"], len(d["ctx"])
        y = d["label"].astype(np.float64).reshape(-1)
        s = np.repeat(seasonal_errors(d["ctx"], d["m"]), pl)
        out[cfg] = stats(y, res[cfg]["Q"].reshape(-1, 9), s)
    return out


def ratio(a, b):
    lc = np.array([np.log(a[k]["crps"] / b[k]["crps"]) for k in b])
    lm = np.array([np.log(a[k]["mase"] / b[k]["mase"]) for k in b])
    return f"CRPS x{np.exp(lc.mean()):.4f}  MASE x{np.exp(lm.mean()):.4f}  better {(lc < -.005).sum()} / worse {(lc > .005).sum()} of {len(lc)}"


def cached(split, pipe=None, ctx=2048):
    path = f"results/c2_{split}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    data = load_split(split)
    t = time.monotonic()
    res = run(pipe or load_pipe(), data, ctx, capture=True)
    print(f"{split}: {len(res)} configs in {time.monotonic() - t:.0f}s", flush=True)
    pickle.dump(res, open(path, "wb"))
    return res


def screen():
    fitd, seld = load_split("fit"), load_split("select")
    pipe = None if os.path.exists("results/c2_fit.pkl") else load_pipe()
    fit, sel = cached("fit", pipe), cached("select", pipe)

    def xy(data, res):
        X, G, grp = [], [], []
        for k, (cfg, d) in enumerate(data.items()):
            y = d["label"][:, :, None].astype(np.float64)
            g = np.nanmean((y < res[cfg]["Q"][:, :, [0, 4, 8]]) - np.array([0.1, 0.5, 0.9]), axis=1)
            X.append(res[cfg]["blocks"].astype(np.float32)); G.append(g); grp.append(np.full(len(g), k))
        return np.concatenate(X), np.concatenate(G), np.concatenate(grp)

    Xf, Gf, gf = xy(fitd, fit)
    Xs, Gs, gs = xy(seld, sel)
    wf = 1.0 / np.bincount(gf)[gf]; wf /= wf.sum()
    ws = 1.0 / np.bincount(gs)[gs]; ws /= ws.sum()
    print("held-out R^2 on SELECT of a ridge probe fit on FIT   (block: tau=0.1 / 0.5 / 0.9)")
    best = np.full(3, -np.inf)
    for L in range(Xf.shape[1]):
        A, B = Xf[:, L].astype(np.float64), Xs[:, L].astype(np.float64)
        mu = wf @ A; sd = np.sqrt(wf @ (A - mu) ** 2) + 1e-6
        A, B = (A - mu) / sd, (B - mu) / sd
        row = []
        for t in range(3):
            y = Gf[:, t]; ok = ~np.isnan(y); ym = wf[ok] @ y[ok] / wf[ok].sum()
            ys = Gs[:, t]; oks = ~np.isnan(ys)
            r2 = -np.inf
            for lam in (1.0, 10.0, 100.0):
                Aw = A[ok] * wf[ok, None]
                beta = np.linalg.solve(Aw.T @ A[ok] + lam / len(A) * np.eye(A.shape[1]), Aw.T @ (y[ok] - ym))
                pred = B[oks] @ beta + ym
                sse = ws[oks] @ (ys[oks] - pred) ** 2
                sst = ws[oks] @ (ys[oks] - ws[oks] @ ys[oks] / ws[oks].sum()) ** 2
                r2 = max(r2, 1 - sse / sst)
            row.append(r2)
        best = np.maximum(best, row)
        print(f"  block {L:2d}: " + "  ".join(f"{r:+.3f}" for r in row))
    print("best held-out R^2 per tau:", " / ".join(f"{b:+.3f}" for b in best))


def base():
    ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 2048
    pipe = load_pipe()
    for split in ("fit", "select"):
        data = load_split(split)
        res = cached(split, pipe, ctx)
        sc = scores(data, res)
        a = np.array([[v["crps"], v["below_q10"], v["below_q50"], v["below_q90"]] for v in sc.values()])
        print(f"{split}: median per-config share below q10/q50/q90 = {np.median(a[:, 1]):.3f}/{np.median(a[:, 2]):.3f}/{np.median(a[:, 3]):.3f}")


if __name__ == "__main__":
    {"base": base, "screen": screen}[sys.argv[1]]()
