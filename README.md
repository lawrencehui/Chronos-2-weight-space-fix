# Weight-space editing of time-series foundation models

Can you make a pretrained forecasting model better by **editing its weights directly**, with no gradient descent and no retraining? This repo follows the idea in [Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), applied to two open forecasting models on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmark, all on one 48 GB Apple M5 Pro.

## In one paragraph

A time-series foundation model reads the history of a number, such as server CPU every 5 minutes or electricity per hour, and predicts a range for what comes next. We tried three things:
- **Small, global tweaks:** nudging forecasts, widening or narrowing their ranges, steering along "bad-forecast" directions. Nothing beat a well-trained model; this is what the theory of calibrated models predicts.
- **Our closed-form head correction:** about 1% better, but on inspection its whole gain came from one hidden bug.
- **An Opus 5.5 agent with the blog's harness:** it found that bug in 30 minutes. When a series has **no observed data in its recent history**, Chronos-2 outputs absurd forecasts (ranges of ±200,000). The agent rewired one unused neuron to detect that case and output a calm, flat forecast instead.

That fix changes **1,536 of 120 million weights** and leaves every other forecast bit-identical. It scores **2.2% better CRPS** on held-out windows, better on 6 configs and worse on none.

## Results

All numbers are on held-out "select" windows from **before** GIFT-Eval's test period (see "Protocol"). Every ratio is against the unedited model on the same windows, and below 1 is better.

| Model / edit | Weights changed | Where | CRPS | MASE | Configs better / worse |
|---|---:|---|---:|---:|---|
| **Chronos-2 + agent's empty-context neuron** | **1,536** | block 11 MLP: `wi` row 11, `wo` col 11 | **×0.9779** | ×1.0000 | **6 / 0** |
| Chronos-2 + closed-form head edit (v2) | 110,736 | `output_patch_embedding.residual_layer`, 144 rows | ×0.9892 | ×0.9998 | 35 / 14 |
| Chronos-2 + both | 112,272 | | ×0.9778 | ×0.9998 | 35 / 14 |
| t0-gamma + gated q90 head edit | 32,794 | `decoder.residual_layer`, 32 rows | ×0.9992 | ×1.0000 | 15 / 9 |
| t0-gamma, blog recipe (64 writer neurons, layers 20–21, ±α) | 131,072 | MLP `down_proj` | ×1.003 to ×1.12 | | worse at every strength |
| t0-gamma, closed-form median readout | | decoder | ×1.0005 | | shrinks to 0 under CV |

The Chronos-2 rows are scored at the official notebook's settings: 8,192 context, multivariate, joint batches of 100. The t0-gamma rows are scored at fast dev settings (2,048 context, no mirrored pass).

What we learned:
- **The head edit's gain is the bug.** Split by config, all of the v2 head edit's gain is on the 8 bitbrains configs (×0.876 there, ×1.000 on the other 89). It was a blurry, 70× larger fix for the same bug the agent pinpointed.
- **Look for a pathology first.** In line with the blog's own conclusion, editing removes an obstacle; it doesn't add an ability. A probe screen (`screen.py`) shows how predictable a model's own error is from its hidden state: R² ≈ 0.1–0.25 here, so global corrections have little to work with.
- **The bug likely shows up in test too.** On the official GIFT-Eval **test** results, Chronos-2 is clearly worse than its peers on bitbrains_rnd (e.g. 5T/medium CRPS 1.00 vs 0.55–0.62 for Toto-2.0-313m, TiRex and t0-gamma).

**Not done yet:** the full GIFT-Eval test run with the edit. From the select-window gain, Chronos-2 would go from 0.4854 to about 0.475, roughly #79 to #59 of 163, but that is a projection, not a test result. Also note that a one-line preprocessing rule ("if the context is empty, forecast flat") would fix the same bug. The interesting part is that an agent **found** it from the weights and forecasts alone and fixed it **inside** the network.

## The agent run (`agent_run/`)

An Opus 5.5 agent worked through `astra_tools.py`, a harness modelled on the blog's Experiment 8.

**Rules:**
- **Budget:** 50 forward passes, where a pass is a fit-window evaluation or an activation read.
- **Free:** reading the weights and the fit windows with their answers.
- **Editable:** only `encoder.block.*`. The embeddings, output head and final norm were refused, as were scalar rescales.
- **No training:** no gradients.
- **Held out:** one final check on the select windows; the test set was never visible.

It used **11 passes and about 30 minutes**. `agent_run/notes.md` is its notebook, `agent_run/final_report.md` its report, `agent_run/state.json` the pass log, `agent_run/passes/` every pass's per-config table, and `agent_run/work/` its scripts, verbatim.

## Protocol (how we avoided overfitting the benchmark)

Every series is split in time: **fit** windows, then **select** windows, then GIFT-Eval's **test** windows. `splits.py` builds the first two from data strictly before the test period, and we verified there is no overlap.
- **Fit** windows: compute directions and edits.
- **Select** windows: choose among variants.
- **Test:** meant to run once, at the end.

`dev_cache.py` freezes a small dev set of up to 24 series × 3 windows per config, so one evaluation pass takes 30–50 s. `run_eval.py` (t0-gamma) and `c2_official.py` (Chronos-2) reproduce the official leaderboard numbers exactly on the configs we checked.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/GiftEval', repo_type='dataset', local_dir='data/gifteval')"
cp .env.example .env            # GIFT_EVAL=<absolute path to data/gifteval>
.venv/bin/python dev_cache.py   # frozen fit / select windows (~20 s)
.venv/bin/python c2_official.py select results/base.csv
.venv/bin/python -c "from chronos import Chronos2Pipeline; from apply_edits import to_full_npz; to_full_npz(Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model, 'edits/chronos2_nanfix_agent.npz', 'nanfix_full.npz')"
.venv/bin/python c2_official.py select results/nanfix.csv --edit nanfix_full.npz
```

## Files

| File | What it does |
|---|---|
| `gifteval_data.py` | GIFT-Eval's `data.py`, vendored unchanged (Apache-2.0) |
| `splits.py`, `dev_cache.py`, `dev_eval.py`, `diagnose.py`, `score.py` | splits, frozen dev set, fast evaluation, diagnostics, leaderboard scoring |
| `run_eval.py`, `c2_official.py` | official-settings evaluation for t0-gamma and Chronos-2 |
| `edits.py`, `wsp_recipe.py` | activation capture; the blog's writer-neuron recipe (t0-gamma) |
| `screen.py`, `probe_edit.py`, `q90_edit.py` | probe screen; closed-form readout edits (t0-gamma) |
| `chronos_dev.py`, `c2_edit.py` | Chronos-2 dev loop and the closed-form 9-quantile head edit |
| `calib_baseline.py`, `shape_baseline.py`, `percfg_baseline.py`, `extrapolate.py`, `settings_sweep.py` | the non-edit baselines that had to be beaten |
| `astra_tools.py` | the agent harness |
| `apply_edits.py`, `edits/` | the three edits as compact deltas, and a loader |
| `ref/` | GIFT-Eval leaderboard result files used for normalisation and comparison |

## Credits and licences

- **Code:** Apache-2.0 (see `LICENSE`).
- **Models:** Chronos-2 (Amazon, Apache-2.0) and t0-gamma (The Forecasting Company, Apache-2.0).
- **Data:** GIFT-Eval data and leaderboard files (Salesforce, Apache-2.0).
- **Method:** the harness design follows the [weight-space programming blog](https://rikiyat.github.io/blog/weight-space-programming/) and its [logs](https://github.com/RikiyaT/lm_understand-logs).
