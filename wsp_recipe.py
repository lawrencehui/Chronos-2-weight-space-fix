"""The blog's writer-neuron recipe on t0-gamma, on the FIT windows (fast mode).

Usage: python wsp_recipe.py <contrast: err|sign> [--alphas 1 2 4 6] [--k 64] [--layers N]
  err : worst third vs best third of windows (scaled quantile loss), matched within each config
  sign: forecasts too high (top third of bias) vs too low (bottom third), matched within each config
1. capture the residual stream after every layer and MLP neuron activity at the forecast positions
2. u_L = unit(mean over configs of (mean_bad - mean_good)); score each layer by held-out separation
   (direction from half the configs, measured on the other half)
3. in the best --layers layers, pick the --k MLP neurons whose extra activity in bad windows writes most along u
4. W <- W - alpha * u (mask * u^T W); dose-response over --alphas, plus a norm-matched random-direction control
Prints CRPS change vs the unedited model on the same windows. Saves results/edit_<contrast>_a<alpha>.npz per alpha.
"""

import argparse
import time

import numpy as np
import torch
from t0 import T0Forecaster

from dev_eval import evaluate, load_split
from edits import capture, column_edit, down_proj, restore, save_edit, window_errors
from run_eval import MODEL_PATH


def split_groups(data, cap, contrast):
    """Per config: indices of 'bad' and 'good' windows (top / bottom third)."""
    groups = {}
    for cfg, d in data.items():
        err, bias = window_errors(d, cap[cfg]["Q"])
        score = err if contrast == "err" else bias
        ok = np.where(~np.isnan(score))[0]
        if len(ok) < 6:
            continue
        o = ok[np.argsort(score[ok])]
        t = len(o) // 3
        groups[cfg] = (o[-t:], o[:t])  # bad (high error / too high), good (low error / too low)
    return groups


def directions(cap, groups):
    cfgs = sorted(groups)
    diffs = np.stack([cap[c]["res"][groups[c][0]].astype(np.float32).mean(0) - cap[c]["res"][groups[c][1]].astype(np.float32).mean(0)
                      for c in cfgs])  # [configs, L, D]
    u = diffs.mean(0)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    # held-out separation per layer: direction from even-index configs, measured on odd, and vice versa
    sep = np.zeros(u.shape[0])
    for a, b in ((0, 1), (1, 0)):
        ua = diffs[a::2].mean(0); ua /= np.linalg.norm(ua, axis=-1, keepdims=True)
        ds = []
        for c in cfgs[b::2]:
            bad, good = groups[c]
            r = cap[c]["res"].astype(np.float32)
            pb, pg = np.einsum("nld,ld->nl", r[bad], ua), np.einsum("nld,ld->nl", r[good], ua)
            ds.append((pb.mean(0) - pg.mean(0)) / (np.concatenate([pb, pg]).std(0) + 1e-6))
        sep += np.mean(ds, 0) / 2
    return u, sep


def writer_neurons(model, cap, groups, layer, u, k):
    W = down_proj(model, layer).weight.detach().float().cpu().numpy()  # [1024, 2048]
    write = u @ W
    dact = np.mean([cap[c]["act"][groups[c][0], layer].astype(np.float32).mean(0) - cap[c]["act"][groups[c][1], layer].astype(np.float32).mean(0)
                    for c in groups], 0)
    contrib = dact * write
    return np.argsort(contrib)[-k:], float(contrib[np.argsort(contrib)[-k:]].sum() / max(contrib[contrib > 0].sum(), 1e-9))


def ratio(res, base):
    logs = np.array([np.log(res[c]["all"]["crps"] / base[c]["all"]["crps"]) for c in base])
    mlogs = np.array([np.log(res[c]["all"]["mase"] / base[c]["all"]["mase"]) for c in base])
    return f"CRPS x{np.exp(logs.mean()):.4f}  MASE x{np.exp(mlogs.mean()):.4f}  better {(logs < -.005).sum()} / worse {(logs > .005).sum()} of {len(logs)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast", choices=["err", "sign"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1, 2, 4, 6])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args()

    model = T0Forecaster.from_pretrained(MODEL_PATH).to("mps").eval()
    data = load_split("fit")
    t = time.monotonic()
    cap, _ = capture(model, data)
    print(f"captured {sum(len(v['res']) for v in cap.values())} windows in {time.monotonic() - t:.0f}s", flush=True)
    base = evaluate(model, data, fast=True)

    groups = split_groups(data, cap, args.contrast)
    u, sep = directions(cap, groups)
    print("held-out separation by layer:", " ".join(f"{i}:{s:.2f}" for i, s in enumerate(sep)), flush=True)
    layers = sorted(np.argsort(sep)[-args.layers:].tolist())
    picks = {}
    for L in layers:
        cols, share = writer_neurons(model, cap, groups, L, u[L], args.k)
        picks[L] = cols
        print(f"layer {L}: top {args.k} writer neurons carry {share:.0%} of the positive write along u", flush=True)

    for alpha in args.alphas:
        saved = {L: column_edit(model, L, u[L], picks[L], alpha) for L in layers}
        dn = sum(float(torch.linalg.norm(down_proj(model, L).weight - saved[L])) for L in layers)
        res = evaluate(model, data, fast=True)
        print(f"alpha={alpha:g} layers {layers} |dW|={dn:.2f}: {ratio(res, base)}", flush=True)
        save_edit(model, layers, f"results/edit_{args.contrast}_a{alpha:g}.npz")
        for L in layers:
            restore(model, L, saved[L])
        if args.no_control:
            continue
        # norm-matched random control: same neurons, random direction orthogonal to u
        rng = np.random.default_rng(int(alpha * 10))
        saved = {}
        for L in layers:
            r = rng.standard_normal(u.shape[1]).astype(np.float32)
            r -= (r @ u[L]) * u[L]; r /= np.linalg.norm(r)
            W = down_proj(model, L).weight.detach().float().cpu().numpy()
            real = np.linalg.norm(np.outer(u[L], (u[L] @ W)[picks[L]])) * alpha
            rand = np.linalg.norm(np.outer(r, (r @ W)[picks[L]]))
            saved[L] = column_edit(model, L, r, picks[L], real / max(rand, 1e-9))
        res = evaluate(model, data, fast=True)
        print(f"   random control, same |dW|: {ratio(res, base)}", flush=True)
        for L in layers:
            restore(model, L, saved[L])


if __name__ == "__main__":
    main()
