"""Backtest windows carved from the data BEFORE GIFT-Eval's test region.

Timeline of every series (pl = prediction length, W = official test windows):

    ... fit (K_FIT windows) | select (K_SEL windows) | test (W windows, never touched while tuning)

fit    -> compute edit directions (v, u)
select -> choose layers / strength k by the same CRPS and MASE the leaderboard uses
test   -> run once, at the end, with the frozen choice

Windows use gluonts' split/generate_instances exactly like gifteval_data.Dataset.test_data,
only shifted earlier, so each forecast sees all history before its window and nothing after.
Series with less than one horizon (min 8 points) of history before a window are dropped from that window set.
"""


import random

from gluonts.dataset.split import split

K_SEL = 3
K_FIT = 6




class _Filtered:
    """Iterable of entries, keeping only those long enough for the requested windows."""

    def __init__(self, entries, need):
        self.entries, self.need = entries, need

    def __iter__(self):
        for e in self.entries:
            n = e["target"].shape[-1]
            if n >= self.need:
                yield e


def _sample(entries, max_series, seed):
    entries = list(entries)
    if max_series and len(entries) > max_series:
        entries = random.Random(seed).sample(entries, max_series)
    return entries


def window_counts(ds):
    """(k_fit, k_sel) for this dataset; M4 has one test window and short series, so one of each."""
    if "m4" in ds.name:
        return 1, 1
    return min(K_FIT, ds.windows), min(K_SEL, ds.windows)


def backtest(ds, which, max_series=None, seed=0):
    """TestData for the 'fit' or 'select' windows of a gifteval_data.Dataset."""
    pl, W = ds.prediction_length, ds.windows
    k_fit, k_sel = window_counts(ds)
    if which == "select":
        skip, k = W, k_sel
    elif which == "fit":
        skip, k = W + k_sel, k_fit
    else:
        raise ValueError(which)
    offset = pl * (skip + k)
    min_ctx = max(pl, 8)  # at least one horizon of history
    entries = _sample(_Filtered(ds.gluonts_dataset, offset + min_ctx), max_series, seed)
    if not entries:
        return None
    _, template = split(entries, offset=-offset)
    return template.generate_instances(prediction_length=pl, windows=k, distance=pl)


def n_windows(test_data):
    return sum(1 for _ in test_data.input) if test_data is not None else 0
