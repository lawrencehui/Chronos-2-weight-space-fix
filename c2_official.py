"""Chronos-2 on GIFT-Eval with the official notebook's settings, optionally with a weight edit.

Usage: python c2_official.py <select|test> <out.csv> [--edit results/c2_edit_v2.npz] [--only ds ...]
Mirrors SalesforceAIResearch/gift-eval notebooks/chronos-2.ipynb: multivariate data kept together,
predict_batches_jointly=True with batch_size=100, default 8192 context, float32, each rolling window predicted
separately, metrics by gluonts evaluate_forecasts. 'select' runs the same settings on the frozen select windows
(up to 24 series x 3 windows per config, all before the test period) to check an edit before the one test run.
Appends per config and skips configs already in <out.csv>, so a killed run resumes.
"""

import argparse
import csv
import itertools
import os
import time
import warnings

import numpy as np
import torch
from chronos import Chronos2Pipeline
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from gifteval_data import Dataset
from run_eval import METRIC_COLS, PROPS, configs
from splits import backtest

warnings.filterwarnings("ignore")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def forecasts_for(pipe, entries, pl):
    inputs = [{"target": e["target"]} for e in entries]
    univariate = inputs[0]["target"].ndim == 1
    q, _ = pipe.predict_quantiles(inputs=inputs, prediction_length=pl, batch_size=100, quantile_levels=LEVELS,
                                  predict_batches_jointly=True)
    q = torch.stack(q).permute(0, 3, 2, 1).cpu().numpy()  # [batch, quantiles, pl, variates]
    if univariate:
        q = q.squeeze(-1)
    return [QuantileForecast(forecast_arrays=a, forecast_keys=list(map(str, LEVELS)), start_date=e["start"] + len(e["target"]))
            for a, e in zip(q, entries)]


def evaluate(pipe, data, pl, freq):
    n_windows = data.windows
    per_window = [forecasts_for(pipe, list(itertools.islice(data.input, k, None, n_windows)), pl) for k in range(n_windows)]
    fcs = [f for group in zip(*per_window) for f in group]  # interleave back to item-major order
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(),
               NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=LEVELS)]
    res = evaluate_forecasts(fcs, test_data=data, metrics=metrics, batch_size=1024, axis=None,
                             mask_invalid_label=True, allow_nan_forecast=False, seasonality=get_seasonality(freq))
    return [res[c].iloc[0] for c in METRIC_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["select", "test"])
    ap.add_argument("out")
    ap.add_argument("--edit", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps", torch_dtype=torch.float32)
    name = "Chronos-2"
    if args.edit:
        sd = pipe.model.state_dict()
        for k, v in np.load(args.edit).items():
            sd[k].copy_(torch.as_tensor(v))
        name = "Chronos-2+wsp"
    done = {r["dataset"] for r in csv.DictReader(open(args.out))} if os.path.exists(args.out) else set()
    if not done:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerow(["dataset", "model"] + [f"eval_metrics/{c}" for c in METRIC_COLS] + ["domain", "num_variates", "seconds"])
    for ds_name, term, cfg, key in configs():
        if cfg in done or (args.only and ds_name not in args.only):
            continue
        ds = Dataset(name=ds_name, term=term, to_univariate=False)
        data = ds.test_data if args.split == "test" else backtest(ds, "select", max_series=24, seed=0)
        if data is None:
            continue
        t = time.monotonic()
        row = evaluate(pipe, data, ds.prediction_length, ds.freq)
        dt = time.monotonic() - t
        with open(args.out, "a", newline="") as f:
            csv.writer(f).writerow([cfg, name] + row + [PROPS[key]["domain"], PROPS[key]["num_variates"], f"{dt:.1f}"])
        print(f"{cfg}: MASE={row[3]:.4f} CRPS={row[-1]:.4f} ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
