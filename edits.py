"""Activation capture and writer-neuron weight edits for t0-gamma (the blog's recipe, exp8 p50c).

capture(model, data)   -> per config: residual stream after every layer and MLP neuron activity, both
                          averaged over the forecast positions of each window, plus the forecasts.
window_errors(d, Q)    -> per window: scaled quantile loss (lower = better) and signed bias of the median.
column_edit(...)       -> W <- W - alpha * u (mask * (u^T W)) on one MLP down-projection (layer.mlp[2]),
                          i.e. remove (alpha=1) or reverse (alpha>1) what the chosen neurons write along u.
load_edit(model, path) -> apply a saved edit (npz of replaced weight tensors) in place.
"""

import math

import numpy as np
import torch

from dev_eval import START, seasonal_errors
from run_eval import CROSS_INFERENCE, make_predictor

TAUS = np.arange(1, 10) / 10


def down_proj(model, layer):
    return model.transformer.layers[layer].mlp[2]  # Linear(2048 -> 1024): its columns are the MLP neurons


def _grouped_order(entries):
    """Row order T0Predictor._predict_grouped feeds the model (window-major, chunks of CROSS_INFERENCE)."""
    seen, by_window = {}, {}
    for i, e in enumerate(entries):
        by_window.setdefault(seen.get(e["item_id"], 0), []).append(i)
        seen[e["item_id"]] = seen.get(e["item_id"], 0) + 1
    return [i for _, rows in sorted(by_window.items()) for s in range(0, len(rows), CROSS_INFERENCE) for i in rows[s:s + CROSS_INFERENCE]]


@torch.no_grad()
def capture(model, data, neuron_layers=None, dec=False, raw_q=None):
    """Fast-mode forecasts plus forecast-position activations, aligned to each config's cache order.
    dec=True also keeps the decoder input at every forecast position: [windows, Hp, 1024] (not averaged).
    raw_q=i also keeps the decoder's raw output for head quantile index i at every forecast step: [windows, Hp, 32]."""
    L = len(model.transformer.layers)
    neuron_layers = list(range(L)) if neuron_layers is None else list(neuron_layers)
    state = {"hp": 1, "res": [], "act": []}

    def fpos(x):
        P, hp = x.shape[1], state["hp"]
        return x[:, P - hp - 1:P - 1].float().mean(1).cpu().numpy().astype(np.float16)

    def fpos_all(x):
        P, hp = x.shape[1], state["hp"]
        return x[:, P - hp - 1:P - 1].float().cpu().numpy().astype(np.float16)

    hooks, buf_res, buf_act, buf_dec, buf_raw = [], {}, {}, [], []
    if raw_q is not None:
        nq = model.head.n_quantiles
        hooks.append(model.decoder.register_forward_hook(
            lambda m, inp, o: buf_raw.append(fpos_all(o)[..., raw_q::nq].astype(np.float32))))
    if dec:
        hooks.append(model.decoder.register_forward_pre_hook(lambda m, inp: buf_dec.append(fpos_all(inp[0]))))
    for i, layer in enumerate(model.transformer.layers):
        hooks.append(layer.register_forward_hook(lambda m, inp, out, i=i: buf_res.setdefault(i, []).append(fpos(out))))
    for i in neuron_layers:
        hooks.append(down_proj(model, i).register_forward_pre_hook(lambda m, inp, i=i: buf_act.setdefault(i, []).append(fpos(inp[0]))))
    out = {}
    try:
        for cfg, d in data.items():
            buf_res.clear(); buf_act.clear(); buf_dec.clear(); buf_raw.clear()
            state["hp"] = math.ceil(d["pl"] / 32)
            idx = np.lexsort((d["window"], d["item"]))
            entries = [{"target": d["ctx"][i], "start": START, "item_id": f"s{d['item'][i]}"} for i in idx]
            fcs = make_predictor(model, d["pl"], fast=True).predict(entries)
            Q = np.empty((len(idx), d["pl"], 9))
            for j, fc in zip(idx, fcs):
                Q[j] = fc.forecast_array.T
            fed = idx[_grouped_order(entries)]  # cache index of each captured row, in feed order
            res = np.stack([np.concatenate(buf_res[i]) for i in range(L)], 1)  # [rows, L, D]
            act = np.stack([np.concatenate(buf_act[i]) for i in neuron_layers], 1) if neuron_layers else None
            assert len(res) == len(fed), (cfg, len(res), len(fed))
            inv = np.empty_like(fed); inv[fed] = np.arange(len(fed))
            out[cfg] = dict(res=res[inv], act=None if act is None else act[inv], Q=Q,
                            dec=np.concatenate(buf_dec)[inv] if dec else None,
                            raw=np.concatenate(buf_raw)[inv] if raw_q is not None else None)
    finally:
        for h in hooks:
            h.remove()
    return out, neuron_layers


def window_errors(d, Q):
    """Per window: mean quantile loss / seasonal error (scale-free), and (median - truth) / seasonal error."""
    s = seasonal_errors(d["ctx"], d["m"])
    y = d["label"].astype(np.float64)[:, :, None]
    ql = 2 * np.abs((y - Q) * ((y <= Q) - TAUS)).mean(-1)  # [n, pl]
    err = np.nanmean(ql, 1) / s
    bias = np.nanmean(Q[:, :, 4] - y[:, :, 0], 1) / s
    return err, bias


def column_edit(model, layer, u, cols, alpha):
    """In place; returns the original weight so the caller can restore it."""
    lin = down_proj(model, layer)
    W0 = lin.weight.data.clone()
    u_t = torch.as_tensor(u, dtype=W0.dtype, device=W0.device)
    mask = torch.zeros(W0.shape[1], dtype=W0.dtype, device=W0.device)
    mask[torch.as_tensor(cols, device=W0.device)] = 1.0
    lin.weight.data = W0 - alpha * torch.outer(u_t, mask * (u_t @ W0))
    return W0


def restore(model, layer, W0):
    down_proj(model, layer).weight.data = W0


def save_edit(model, layers, path):
    np.savez(path, **{f"transformer.layers.{i}.mlp.2.weight": down_proj(model, i).weight.detach().cpu().numpy() for i in layers})


def load_edit(model, path):
    sd = model.state_dict()
    for k, v in np.load(path).items():
        sd[k].copy_(torch.as_tensor(v))
