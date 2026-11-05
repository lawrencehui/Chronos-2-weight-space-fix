"""Trivial fix the weight edit must beat: shrink each forecast's spread around its median.

Usage: python calib_baseline.py <fit_prefix> <select_prefix>   (e.g. results/fast_base_fit results/fast_base_select)
q' = median + c * (q - median), with c per horizon third chosen on FIT (grid), scored on SELECT.
Reported as the leaderboard aggregate: geomean over configs of CRPS_new / CRPS_base.
"""
import sys

import numpy as np

from dev_eval import load_split

TAUS = np.arange(1, 10) / 10


def crps(y, Q):
    ok = ~np.isnan(y)
    y, Q = y[ok], Q[ok]
    ql = 2 * np.abs((y[:, None] - Q) * ((y[:, None] <= Q) - TAUS))
    return float(np.mean(ql.sum(0) / np.abs(y).sum()))


def shrink(Q, c_by_third):
    pl = Q.shape[1]
    out = Q.copy()
    for t, c in enumerate(c_by_third):
        sl = slice(int(t * pl / 3), int((t + 1) * pl / 3))
        med = Q[:, sl, 4:5]
        out[:, sl] = med + c * (Q[:, sl] - med)
    return out


def score(split, qpath, cs):
    data, qs = load_split(split), np.load(qpath)
    logs = []
    for cfg, d in data.items():
        Q = qs[cfg.replace("/", "__")].astype(np.float64)
        y = d["label"].astype(np.float64)
        logs.append(np.log(crps(y.reshape(-1), shrink(Q, cs).reshape(-1, 9)) / crps(y.reshape(-1), Q.reshape(-1, 9))))
    return float(np.exp(np.mean(logs)))


fit_q, sel_q = sys.argv[1] + ".q.npz", sys.argv[2] + ".q.npz"
grid = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
best1 = min(grid, key=lambda c: score("fit", fit_q, [c] * 3))
print(f"one global factor: c={best1} chosen on fit -> fit x{score('fit', fit_q, [best1]*3):.4f}, select x{score('select', sel_q, [best1]*3):.4f}")
cs = [best1] * 3
for t in range(3):  # coordinate search per horizon third
    cs[t] = min(grid, key=lambda c: score("fit", fit_q, cs[:t] + [c] + cs[t + 1:]))
print(f"per horizon third: c={cs} chosen on fit -> fit x{score('fit', fit_q, cs):.4f}, select x{score('select', sel_q, cs):.4f}")
