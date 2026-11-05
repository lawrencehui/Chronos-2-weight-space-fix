"""Screen before editing: can ANY layer's state predict the model's own signed quantile miss?

Usage: python screen.py
For tau in (0.1, 0.5, 0.9): g = mean over each window of (1{y < q_tau} - tau)  (0 when calibrated for that window).
For every layer: ridge probe g ~ w.h on FIT windows (residual stream averaged over forecast positions),
scored on SELECT windows by out-of-sample R^2. An edit built from a layer's state can only help if R^2 > 0
there (first-order gain = E[c(h) g]; see Blasiok et al. 2023 on calibration and post-processing).
"""

import numpy as np
from t0 import T0Forecaster

from dev_eval import load_split
from edits import capture
from run_eval import MODEL_PATH


def targets(data, cap):
    X, G, grp = [], [], []
    for k, (cfg, d) in enumerate(data.items()):
        Q, y = cap[cfg]["Q"], d["label"][:, :, None].astype(np.float64)
        g = np.nanmean((y < Q[:, :, [0, 4, 8]]) - np.array([0.1, 0.5, 0.9]), axis=1)  # [n, 3]
        X.append(cap[cfg]["res"].astype(np.float32)); G.append(g); grp.append(np.full(len(g), k))
    return np.concatenate(X), np.concatenate(G), np.concatenate(grp)


def main():
    model = T0Forecaster.from_pretrained(MODEL_PATH).to("mps").eval()
    fitd, seld = load_split("fit"), load_split("select")
    Xf, Gf, gf = targets(fitd, capture(model, fitd, neuron_layers=[])[0])
    Xs, Gs, gs = targets(seld, capture(model, seld, neuron_layers=[])[0])
    # equal weight per config (the leaderboard averages configs)
    wf = 1.0 / np.bincount(gf)[gf]; wf /= wf.sum()
    ws = 1.0 / np.bincount(gs)[gs]; ws /= ws.sum()
    print("out-of-sample R^2 on SELECT of a ridge probe fit on FIT  (rows: layer; cols: tau=0.1 / 0.5 / 0.9)")
    best = np.full(3, -np.inf)
    for L in range(Xf.shape[1]):
        A, B = Xf[:, L].astype(np.float64), Xs[:, L].astype(np.float64)
        mu, sd = (wf @ A), np.sqrt(wf @ (A - wf @ A) ** 2) + 1e-6
        A, B = (A - mu) / sd, (B - mu) / sd
        row = []
        for t in range(3):
            y = Gf[:, t]; ok = ~np.isnan(y)
            ym = wf[ok] @ y[ok] / wf[ok].sum()
            r2s = []
            for lam in (1.0, 10.0, 100.0):
                Aw = A[ok] * wf[ok, None]
                beta = np.linalg.solve(Aw.T @ A[ok] + lam / len(A) * np.eye(A.shape[1]), Aw.T @ (y[ok] - ym))
                ys = Gs[:, t]; oks = ~np.isnan(ys)
                pred = B[oks] @ beta + ym
                sse = ws[oks] @ (ys[oks] - pred) ** 2
                sst = ws[oks] @ (ys[oks] - ws[oks] @ ys[oks] / ws[oks].sum()) ** 2
                r2s.append(1 - sse / sst)
            row.append(max(r2s))
        best = np.maximum(best, row)
        print(f"  layer {L:2d}: " + "  ".join(f"{r:+.3f}" for r in row))
    print("best held-out R^2 per tau:", " / ".join(f"{b:+.3f}" for b in best))


if __name__ == "__main__":
    main()
