"""Inference settings sweep (no weight change): context length x mirrored pass, on dev windows.

Usage: python settings_sweep.py <fit|select> ctx:mirror [ctx:mirror ...]    e.g. 2048:1 4096:0
Reference is the leaderboard setting (8192 context, mirrored). Prints CRPS/MASE ratios vs it,
overall and by frequency group, so a per-frequency context choice can be read off the fit split.
"""
import sys
import time

import numpy as np
import torch
from t0 import T0Forecaster

import dev_eval
from run_eval import MODEL_PATH, make_predictor


def run(model, data, ctx, mirror):
    def pred(m, d, fast=False):
        idx = np.lexsort((d["window"], d["item"]))
        entries = [{"target": d["ctx"][i], "start": dev_eval.START, "item_id": f"s{d['item'][i]}"} for i in idx]
        fcs = make_predictor(m, d["pl"], ctx=ctx, mirror=mirror).predict(entries)
        Q = np.empty((len(idx), d["pl"], 9))
        for j, fc in zip(idx, fcs):
            Q[j] = fc.forecast_array.T
        return Q
    dev_eval.predict = pred
    return dev_eval.evaluate(model, data)


split, settings = sys.argv[1], [s.split(":") for s in sys.argv[2:]]
model = T0Forecaster.from_pretrained(MODEL_PATH).to("mps").eval()
data = dev_eval.load_split(split)
t = time.monotonic()
ref = run(model, data, 8192, True)
print(f"reference 8192:mirror {time.monotonic() - t:.0f}s")
freq = lambda k: k.split("/")[1]
for ctx, mir in settings:
    t = time.monotonic()
    res = run(model, data, int(ctx), mir == "1")
    logs = {k: np.log(res[k]["all"]["crps"] / ref[k]["all"]["crps"]) for k in ref}
    ml = np.mean([np.log(res[k]["all"]["mase"] / ref[k]["all"]["mase"]) for k in ref])
    by = {}
    for k, v in logs.items():
        by.setdefault(freq(k), []).append(v)
    v = np.array(list(logs.values()))
    print(f"{ctx}:{'mirror' if mir == '1' else 'single'}  CRPS x{np.exp(v.mean()):.4f}  MASE x{np.exp(ml):.4f}  "
          f"better {(v < -.005).sum()} / worse {(v > .005).sum()}  ({time.monotonic() - t:.0f}s)")
    print("   by freq:", " ".join(f"{f}:x{np.exp(np.mean(x)):.3f}" for f, x in sorted(by.items())), flush=True)
