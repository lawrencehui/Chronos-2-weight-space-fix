"""Which GIFT-Eval test windows have an entirely missing context (the case the agent's fix targets)?

Usage: python scan_empty_context.py [--split test|select]
Data only, no model. A window counts if any variate's last 8192 context points (Chronos-2's context length)
are all NaN. The fix only changes those windows, so only configs with at least one need re-running.
"""
import sys

import numpy as np

from gifteval_data import Dataset
from run_eval import configs
from splits import backtest

split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
hits = []
for ds_name, term, cfg, _ in configs():
    ds = Dataset(name=ds_name, term=term, to_univariate=False)
    data = ds.test_data if split == "test" else backtest(ds, "select", max_series=24, seed=0)
    n = empty = 0
    for e in data.input:
        t = np.atleast_2d(np.asarray(e["target"], dtype=np.float64))[:, -8192:]
        n += 1
        empty += int(np.isnan(t).all(axis=1).any())
    if empty:
        hits.append((cfg, ds_name, empty, n))
        print(f"{cfg}: {empty} of {n} windows have an all-missing context", flush=True)
print(f"configs affected: {len(hits)}")
print("datasets:", " ".join(sorted({h[1] for h in hits})))
