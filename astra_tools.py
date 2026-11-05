"""Harness for an LLM agent editing Chronos-2's weights directly (modelled on the weight-space-programming
blog's Experiment 8 harness). Every action goes through this CLI; the budget and rules are enforced here.

  python astra_tools.py status
  python astra_tools.py forward <edit.npz|base> [--only SUBSTR ...]      # 1 pass: score an edit on the FIT windows
  python astra_tools.py read <edit.npz|base> --out FILE.npz [--blocks 0 5 11] [--only SUBSTR ...]
                                                                         # 1 pass: activations at forecast positions
  python astra_tools.py save_edit <name> <file.npz>                      # free: validate + store astra/edits/<name>.npz
  python astra_tools.py final <name>                                     # once only: score the final edit on SELECT

Rules (enforced): at most BUDGET passes ('forward' or 'read', any size); one 'final'. Edits may replace only tensors
under encoder.block.* (attention q/k/v/o, layer norms, MLP wi/wo); the input/output patch embeddings, the [REG]
embedding table and the final layer norm are refused, as is replacing a tensor by a constant multiple of itself.
Free (no pass): reading the weights, reading the FIT windows (contexts AND truths, like practice problems with answers).
Forbidden by instruction: gradients/backward/optimisers/iterative fitting to a loss, running the model outside this
tool, reading the SELECT windows or the test data.
Metric: GIFT-Eval-style CRPS (mean weighted quantile loss, 9 quantiles) and MASE per config; the score of an edit is
the geometric mean over configs of CRPS_edit / CRPS_base (below 1 = better), as on the leaderboard.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "astra")
STATE = os.path.join(WORK, "state.json")
BUDGET = 50
EDITABLE_PREFIX = "encoder.block."


def _state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"budget": BUDGET, "passes_used": 0, "final_used": False, "log": []}


def _save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)


def _spend(kind, detail):
    s = _state()
    if s["passes_used"] >= s["budget"]:
        sys.exit(f"REFUSED: forward-pass budget exhausted ({s['passes_used']}/{s['budget']}).")
    s["passes_used"] += 1
    s["log"].append({"pass": s["passes_used"], "kind": kind, "detail": detail, "time": time.strftime("%H:%M:%S")})
    _save_state(s)
    return s["passes_used"]


def _pipe(edit):
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    if edit and edit != "base":
        path = edit if edit.endswith(".npz") else os.path.join(WORK, "edits", edit + ".npz")
        sd = pipe.model.state_dict()
        for k, v in np.load(path).items():
            sd[k].copy_(torch.as_tensor(v))
    return pipe


def _data(split, only):
    from dev_eval import load_split
    return load_split(split, only)


def _base(split):
    """Unedited forecasts on a split, computed once by the harness (free)."""
    path = os.path.join(WORK, f"base_{split}.pkl")
    if not os.path.exists(path):
        from chronos_dev import run
        res = run(_pipe("base"), _data(split, None), capture=False)
        pickle.dump({k: v["Q"] for k, v in res.items()}, open(path, "wb"))
    return pickle.load(open(path, "rb"))


def _stats(data, Qs):
    from dev_eval import seasonal_errors
    from diagnose import stats
    out = {}
    for cfg, d in data.items():
        pl = d["pl"]
        s = np.repeat(seasonal_errors(d["ctx"], d["m"]), pl)
        out[cfg] = stats(d["label"].astype(np.float64).reshape(-1), Qs[cfg].reshape(-1, 9), s)
    return out


def _report(data, Q, base_Q, tag):
    import csv
    e, b = _stats(data, Q), _stats(data, base_Q)
    keys = list(e)
    lc = np.array([math.log(e[k]["crps"] / b[k]["crps"]) for k in keys])
    lm = np.array([math.log(e[k]["mase"] / b[k]["mase"]) for k in keys])
    os.makedirs(os.path.join(WORK, "passes"), exist_ok=True)
    path = os.path.join(WORK, "passes", f"{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "crps_ratio", "mase_ratio", "crps", "base_crps", "below_q10", "below_q50", "below_q90", "bias"])
        for k, x, y in zip(keys, lc, lm):
            w.writerow([k, f"{math.exp(x):.4f}", f"{math.exp(y):.4f}", f"{e[k]['crps']:.5f}", f"{b[k]['crps']:.5f}",
                        f"{e[k]['below_q10']:.3f}", f"{e[k]['below_q50']:.3f}", f"{e[k]['below_q90']:.3f}", f"{e[k]['bias']:+.3f}"])
    print(f"SCORE (geomean CRPS ratio vs unedited, <1 is better): x{math.exp(lc.mean()):.4f}   MASE x{math.exp(lm.mean()):.4f}")
    print(f"configs better (>0.5%): {(lc < -0.005).sum()}   worse: {(lc > 0.005).sum()}   of {len(keys)}")
    o = np.argsort(lc)
    print("largest gains:", ", ".join(f"{keys[i]} x{math.exp(lc[i]):.3f}" for i in o[:5]))
    print("largest losses:", ", ".join(f"{keys[i]} x{math.exp(lc[i]):.3f}" for i in o[-5:]))
    print(f"per-config table: {path}")


def cmd_status(a):
    s = _state()
    print(f"passes used {s['passes_used']}/{s['budget']}   final used: {s['final_used']}")
    for e in s["log"][-10:]:
        print(f"  #{e['pass']} {e['time']} {e['kind']} {e['detail']}")


def cmd_forward(a):
    n = _spend("forward", f"{a.edit} only={a.only}")
    from chronos_dev import run
    data = _data("fit", a.only)
    base = _base("fit")
    res = run(_pipe(a.edit), data, capture=False)
    _report(data, {k: v["Q"] for k, v in res.items()}, {k: base[k] for k in data}, f"pass{n:02d}_{os.path.basename(a.edit).replace('.npz', '')}")


def cmd_read(a):
    import torch
    n = _spend("read", f"{a.edit} blocks={a.blocks} only={a.only}")
    from dev_eval import START  # noqa: F401  (same windows as forward)
    data = _data("fit", a.only)
    pipe = _pipe(a.edit)
    m = pipe.model
    st = {"nop": 1}
    buf = {}

    def fpos(x):
        x = x if torch.is_tensor(x) else x[0]
        return x[:, -st["nop"]:].float().mean(1).cpu().numpy().astype(np.float16)

    hooks = []
    for i, b in enumerate(m.encoder.block):
        hooks.append(b.register_forward_hook(lambda mod, inp, out, i=i: buf.setdefault(f"block{i}_out", []).append(fpos(out))))
    for i in a.blocks:
        blk = m.encoder.block[i]
        hooks.append(blk.layer[2].mlp.wo.register_forward_pre_hook(lambda mod, inp, i=i: buf.setdefault(f"block{i}_mlp_act", []).append(fpos(inp[0]))))
        hooks.append(blk.layer[0].self_attention.o.register_forward_pre_hook(lambda mod, inp, i=i: buf.setdefault(f"block{i}_timeattn_heads", []).append(fpos(inp[0]))))
    out = {"config": [], "window_index": [], "Q": [], "label": []}
    arrays = {}
    try:
        for cfg, d in data.items():
            for v in buf.values():
                v.clear()
            st["nop"] = math.ceil(d["pl"] / 16)
            q, _ = pipe.predict_quantiles([{"target": c[-2048:].astype(np.float32)} for c in d["ctx"]],
                                          prediction_length=d["pl"], quantile_levels=[0.1 * k for k in range(1, 10)], batch_size=256)
            for k, v in buf.items():
                arrays.setdefault(k, []).append(np.concatenate(v))
            out["config"] += [cfg] * len(d["ctx"])
            out["window_index"] += list(range(len(d["ctx"])))
    finally:
        for h in hooks:
            h.remove()
    np.savez(a.out, config=np.array(out["config"]), window_index=np.array(out["window_index"]),
             **{k: np.concatenate(v) for k, v in arrays.items()})
    print(f"saved {a.out}: " + ", ".join(f"{k} {np.concatenate(v).shape}" for k, v in arrays.items()))
    print("rows follow configs in sorted order, each config's windows in data/dev/fit order (window_index).")


def cmd_save_edit(a):
    import torch
    from chronos import Chronos2Pipeline
    sd = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu").model.state_dict()
    z = dict(np.load(a.file))
    for k, v in z.items():
        if k not in sd:
            sys.exit(f"REFUSED: unknown tensor {k}")
        if not k.startswith(EDITABLE_PREFIX):
            sys.exit(f"REFUSED: {k} is not editable (only encoder.block.* tensors may change)")
        w = sd[k].numpy()
        if v.shape != w.shape or not np.all(np.isfinite(v)):
            sys.exit(f"REFUSED: {k} has the wrong shape or non-finite values")
        c = float((v * w).sum() / max((w * w).sum(), 1e-30))
        if np.allclose(v, c * w, rtol=1e-6, atol=1e-9) and not np.allclose(v, w):
            sys.exit(f"REFUSED: {k} is a constant multiple ({c:.4f}) of the original")
    os.makedirs(os.path.join(WORK, "edits"), exist_ok=True)
    dst = os.path.join(WORK, "edits", a.name + ".npz")
    np.savez(dst, **z)
    changed = sum(int((z[k] != sd[k].numpy()).sum()) for k in z)
    print(f"saved {dst}: {len(z)} tensors, {changed} changed values")


def cmd_final(a):
    s = _state()
    if s["final_used"]:
        sys.exit("REFUSED: the final SELECT check was already used.")
    s["final_used"] = True
    s["log"].append({"pass": "final", "kind": "final", "detail": a.name, "time": time.strftime("%H:%M:%S")})
    _save_state(s)
    from chronos_dev import run
    data = _data("select", None)
    base = _base("select")
    res = run(_pipe(a.name), data, capture=False)
    print("FINAL CHECK on held-out SELECT windows:")
    _report(data, {k: v["Q"] for k, v in res.items()}, base, f"final_{a.name}")


def main():
    os.makedirs(WORK, exist_ok=True)
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    f = sp.add_parser("forward"); f.add_argument("edit"); f.add_argument("--only", nargs="*", default=None)
    r = sp.add_parser("read"); r.add_argument("edit"); r.add_argument("--out", required=True)
    r.add_argument("--blocks", nargs="*", type=int, default=[]); r.add_argument("--only", nargs="*", default=None)
    s = sp.add_parser("save_edit"); s.add_argument("name"); s.add_argument("file")
    fi = sp.add_parser("final"); fi.add_argument("name")
    a = ap.parse_args()
    {"status": cmd_status, "forward": cmd_forward, "read": cmd_read, "save_edit": cmd_save_edit, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
