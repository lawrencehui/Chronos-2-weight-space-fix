"""Evaluate t0-gamma on GIFT-Eval exactly as the official t0-gamma notebook does.

Usage: python run_eval.py <out.csv> [--only ds_name ...] [--device mps|cpu]
Appends one row per config and skips configs already in <out.csv>, so a killed run resumes.
Settings mirror tfc-t0/notebooks/t0-gamma.ipynb (context 8192, 9 quantiles, all four predict flags).
"""

import argparse
import csv
import json
import logging
import os
import time
import warnings

import torch
from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from t0 import T0Forecaster
from t0.evaluation import T0Predictor

from gifteval_data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("gluonts.model.forecast").addFilter(
    lambda r: "The mean prediction is not stored" not in r.getMessage()
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "theforecastingcompany/t0-gamma"
CONTEXT_LENGTH = 8192
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH_SIZE = 64
CROSS_INFERENCE = 64

SHORT = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
    "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D "
    "ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T "
    "bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
METRIC_COLS = ["MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS",
               "RMSE[mean]", "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss"]
PROPS = json.load(open(os.path.join(HERE, "ref", "dataset_properties.json")))


def configs():
    """All 97 (ds_name, term, config_name) triples, in a fixed order."""
    for ds_name in sorted(set(SHORT + MED_LONG)):
        for term in ("short", "medium", "long"):
            if term != "short" and ds_name not in MED_LONG:
                continue
            if "/" in ds_name:
                key, freq = ds_name.split("/")
                key = PRETTY.get(key.lower(), key.lower())
            else:
                key = PRETTY.get(ds_name.lower(), ds_name.lower())
                freq = PROPS[key]["frequency"]
            yield ds_name, term, f"{key}/{freq}/{term}", key


def load_dataset(ds_name, term):
    to_uni = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
    return Dataset(name=ds_name, term=term, to_univariate=to_uni)


def make_predictor(model, prediction_length, fast=False, ctx=None, mirror=None):
    """Leaderboard settings; fast=True (dev iteration only) uses 2048 context and no mirrored pass.
    ctx / mirror override either setting explicitly (for the settings sweep)."""
    return T0Predictor(
        model,
        prediction_length=prediction_length,
        quantile_levels=QUANTILE_LEVELS,
        context_length=ctx or (2048 if fast else CONTEXT_LENGTH),
        batch_size=BATCH_SIZE,
        show_progress=False,
        trim_leading_missing=True,
        symmetric_averaging=(not fast) if mirror is None else mirror,
        clip_nonnegative=True,
        snap_integers=True,
        cross_inference=CROSS_INFERENCE,
    )


def score(predictor, test_data, freq):
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
               RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS)]
    res = evaluate_model(predictor, test_data=test_data, metrics=metrics, batch_size=1024, axis=None,
                         mask_invalid_label=True, allow_nan_forecast=False, seasonality=get_seasonality(freq))
    return [res[c].iloc[0] for c in METRIC_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*", default=None, help="dataset names (e.g. us_births/D) to run")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--name", default="t0-gamma")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        done = {r["dataset"] for r in csv.DictReader(open(args.out))}
    else:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerow(["dataset", "model"] + [f"eval_metrics/{c}" for c in METRIC_COLS]
                                   + ["domain", "num_variates", "seconds"])

    model = T0Forecaster.from_pretrained(MODEL_PATH).to(args.device).eval()
    for ds_name, term, cfg, key in configs():
        if cfg in done or (args.only and ds_name not in args.only):
            continue
        ds = load_dataset(ds_name, term)
        t = time.monotonic()
        row = score(make_predictor(model, ds.prediction_length), ds.test_data, ds.freq)
        dt = time.monotonic() - t
        with open(args.out, "a", newline="") as f:
            csv.writer(f).writerow([cfg, args.name] + row + [PROPS[key]["domain"], PROPS[key]["num_variates"], f"{dt:.1f}"])
        print(f"{cfg}: MASE={row[3]:.4f} CRPS={row[-1]:.4f} ({dt:.1f}s, {len(ds.test_data)} windows)", flush=True)


if __name__ == "__main__":
    main()
