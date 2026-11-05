"""Upper bound for a conditional fix: per-config spread / shape / level corrections chosen on FIT, scored on SELECT.

Usage: python percfg_baseline.py <fit_prefix> <select_prefix>
If corrections that are tuned per config on fit windows don't carry over to select windows, a weight edit
conditioned on the same signals has little to gain either.
"""
import itertools
import sys

import numpy as np

from dev_eval import load_split, seasonal_errors

TAUS = np.arange(1, 10) / 10


def crps(y, Q):
    ok = ~np.isnan(y)
    y, Q = y[ok], Q[ok]
    return float(np.mean((2 * np.abs((y[:, None] - Q) * ((y[:, None] <= Q) - TAUS))).sum(0) / np.abs(y).sum()))


def fix(Q, s, c, a, b):
    med = Q[:, :, 4:5]
    Q = med + c * (Q - med)                                     # spread
    Q = Q + (a - 1) * (med - med.mean(axis=1, keepdims=True))   # shape amplitude
    return Q + b * s[:, None, None]                             # level, in seasonal-error units


fit_q, sel_q = np.load(sys.argv[1] + ".q.npz"), np.load(sys.argv[2] + ".q.npz")
fit, sel = load_split("fit"), load_split("select")
grid = list(itertools.product([0.85, 0.95, 1.0, 1.05, 1.15], [0.9, 1.0, 1.1], [-0.3, -0.1, 0.0, 0.1, 0.3]))
logs, chosen = [], {}
for cfg in sel:
    key = cfg.replace("/", "__")
    Qs, ys = sel_q[key].astype(np.float64), sel[cfg]["label"].astype(np.float64)
    ss = np.nan_to_num(seasonal_errors(sel[cfg]["ctx"], sel[cfg]["m"]))
    best = (1.0, 1.0, 0.0)
    if cfg in fit:
        Qf, yf = fit_q[key].astype(np.float64), fit[cfg]["label"].astype(np.float64)
        sf = np.nan_to_num(seasonal_errors(fit[cfg]["ctx"], fit[cfg]["m"]))
        best = min(grid, key=lambda p: crps(yf.reshape(-1), fix(Qf, sf, *p).reshape(-1, 9)))
    chosen[cfg] = best
    logs.append(np.log(crps(ys.reshape(-1), fix(Qs, ss, *best).reshape(-1, 9)) / crps(ys.reshape(-1), Qs.reshape(-1, 9))))
logs = np.array(logs)
print(f"per-config fixes chosen on fit -> select x{np.exp(logs.mean()):.4f}  better {(logs < -0.005).sum()} / worse {(logs > 0.005).sum()} of {len(logs)}")
print(f"configs where a fix was chosen at all: {sum(v != (1.0, 1.0, 0.0) for v in chosen.values())}")
order = np.argsort(logs)
names = list(sel)
print("largest select gains:", [(names[i], round(float(np.exp(logs[i])), 3), chosen[names[i]]) for i in order[:6]])
print("largest select losses:", [(names[i], round(float(np.exp(logs[i])), 3), chosen[names[i]]) for i in order[-6:]])
