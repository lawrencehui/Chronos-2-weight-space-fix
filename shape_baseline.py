"""Second trivial check: is the forecast shape too flat? Stretch it and see if error drops.

Usage: python shape_baseline.py <fit_prefix> <select_prefix>
Each window's quantile paths are moved by a*(median - mean of median over the horizon) - that, i.e. the
median's swing around its own mean is scaled by a, and every quantile shifts with it. a chosen on FIT, scored on SELECT.
"""
import sys

import numpy as np

from calib_baseline import crps
from dev_eval import load_split


def stretch(Q, a):
    med = Q[:, :, 4]
    shift = (a - 1) * (med - med.mean(axis=1, keepdims=True))
    return Q + shift[:, :, None]


def score(split, qpath, a, keep=None):
    data, qs = load_split(split), np.load(qpath)
    logs = {}
    for cfg, d in data.items():
        if keep and not keep(cfg):
            continue
        Q = qs[cfg.replace("/", "__")].astype(np.float64)
        y = d["label"].astype(np.float64).reshape(-1)
        logs[cfg] = np.log(crps(y, stretch(Q, a).reshape(-1, 9)) / crps(y, Q.reshape(-1, 9)))
    return float(np.exp(np.mean(list(logs.values())))), logs


fit_q, sel_q = sys.argv[1] + ".q.npz", sys.argv[2] + ".q.npz"
grid = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
for label, keep in [("all configs", None), ("medium+long only", lambda c: not c.endswith("short"))]:
    fits = {a: score("fit", fit_q, a, keep)[0] for a in grid}
    a = min(fits, key=fits.get)
    sel, logs = score("select", sel_q, a, keep)
    better = sum(v < -0.005 for v in logs.values()); worse = sum(v > 0.005 for v in logs.values())
    print(f"{label}: fit {' '.join(f'a={k}:x{v:.4f}' for k, v in fits.items())}")
    print(f"   chosen a={a} -> select x{sel:.4f}  (better {better} / worse {worse} of {len(logs)})")
