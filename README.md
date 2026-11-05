# An AI found a bug in a forecasting model and fixed it by changing 1,536 numbers

**In short**
- We tried to improve a top open-source forecasting model **without retraining it**, by editing its internal numbers (its *weights*) directly.
- Small, general tweaks did nothing: the model was already well tuned.
- An AI agent then inspected the model and found a real **bug**: when a series has no recent data at all, the model predicts nonsense. The agent fixed it by rewiring **one unused neuron**, which changed **1,536 of the model's 120 million numbers**.
- Result on the **official GIFT-Eval test set**:
  - overall forecast error down 1.05%;
  - up to **33% better** on the tasks the bug hit;
  - **no task worse**;
  - leaderboard rank **#79 → #69** of 164.

---

## What is this model?

[Chronos-2](https://huggingface.co/amazon/chronos-2), from Amazon, is a **forecasting model**. You give it the history of a number over time, such as a server's CPU use every 5 minutes, a city's electricity demand every hour, or a shop's weekly sales. It predicts what comes next. It doesn't give just one guess; it gives a range: *"probably about 50, very likely between 40 and 60."*

It's a *foundation model*: trained once on a huge amount of data, then used on almost anything without retraining. Models like this compete on a public leaderboard called [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), which tests them on **97 real forecasting tasks** (electricity, traffic, weather, sales, cloud servers and more).

## The idea: edit the model's brain directly

A neural network is millions of numbers. Normally, the only way to change what it does is to **retrain** it, which is slow and needs lots of data and compute. This project asks: **can you open the model up and change a few of those numbers by hand, like a surgeon, to make it better?**

The idea comes from a blog post, [*Sparks of Weight Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), where an AI agent made a small language model much better at maths. It did this by finding and removing the "neurons" that made it repeat itself in a loop.

## What we tried

**1. General tune-ups: no effect.** We tried nudging forecasts up or down, widening or narrowing their ranges, and switching off parts of the model linked to bad forecasts. **Nothing helped, and most attempts made things slightly worse.** The models were already well calibrated, and maths explains why: a well-trained model sits at a sweet spot that small general tweaks can't improve.

**2. A computed correction: a small win, but for the wrong reason.** We built a correction from the model's own internal signals and wrote it into its final layer. It looked about 1% better. On closer inspection, **all of that gain came from 8 of the 97 tasks**: the correction was half-fixing a hidden bug without us knowing.

**3. An AI agent, like the blog's: it found the bug.** We gave an AI agent (Claude Opus 5.5) the same kind of setup the blog used:
- it could read every number in the model;
- it could look at the training examples and their answers;
- it could test its ideas on the model 50 times;
- it could not retrain the model.

**In 30 minutes and 11 tests, it found the problem:**

> When a series has **no data at all** in its recent history, the model panics and predicts nonsense, with ranges of **±200,000**.

We reported it to the Chronos team, with a minimal reproduction: [amazon-science/chronos-forecasting#544](https://github.com/amazon-science/chronos-forecasting/issues/544).

This happens in the **Bitbrains** data: real usage logs from a cloud-hosting company's servers. Servers get switched off and on, so sometimes a server has been off for its whole recent history and comes back on during the period being forecast. One absurd forecast like that ruins the score for the whole task.

## The fix: one neuron that was doing nothing

The agent noticed that one neuron in the model's last layer **never activated**: it was dead weight. It rewired that neuron to do one job:

- **Its input side** (768 numbers) now detects *"this history is completely empty"*.
- **Its output side** (768 numbers) then steers the forecast to *"calm, flat, close to zero"*.

For every normal series the neuron stays silent, and **the model's output is exactly the same as before**. We tested where it switches on: with no real values, or just one (where the model has nothing to judge spread from), it fires; with two or more real values in the history, the forecast is bit-identical to the original model. That's 1,536 numbers changed out of 120 million (0.001%).

## Results

**On held-out data**, a later stretch of time the agent never looked at:

| | Tasks improved | Tasks made worse | Overall error (CRPS) |
|---|---|---|---|
| The agent's fix (1,536 numbers) | 6 | **0** | **2.2% lower** |
| Our computed correction (110,736 numbers) | 35 | 14 | 1.1% lower |
| Both together | 35 | 14 | 2.2% lower, so ours adds nothing |

**On the official GIFT-Eval test set:**

Only 12 of the 97 test tasks contain a series whose history has at most one real value (`scan_empty_context.py`). The fix can only change those, so we re-ran those 12 with the edited model.

Error is measured by CRPS, which scores how good a forecast's *range* is (accurate and appropriately confident); lower is better.

| Task | Original Chronos-2 | With the fix | Change |
|---|---:|---:|---:|
| bitbrains_rnd / 5-minute / medium | 1.0046 | 0.6700 | **−33.3%** |
| bitbrains_rnd / 5-minute / long | 0.8730 | 0.6254 | **−28.4%** |
| bitbrains_rnd / hourly / short | 0.8035 | 0.6702 | **−16.6%** |
| bitbrains_fast_storage / 5-minute / long | 0.7026 | 0.6752 | −3.9% |
| bitbrains_rnd / 5-minute / short | 0.4157 | 0.4033 | −3.0% |
| bitbrains_fast_storage / 5-minute / medium | 0.6233 | 0.6107 | −2.0% |
| bitbrains_fast_storage / hourly / short | 0.6635 | 0.6586 | −0.7% |
| bitbrains_fast_storage / 5-minute / short | 0.3906 | 0.3899 | −0.2% |
| electricity / weekly / short | 0.0559 | 0.0559 | 0.0% |
| kdd_cup_2018 / hourly / short, medium, long | 0.3736 / 0.4172 / 0.4453 | 0.3736 / 0.4172 / 0.4453 | 0.0% |

**Overall leaderboard score**, normalised CRPS across all 97 tasks: **0.4854 → 0.4803**. That moves the model from **#79 to #69 of 164** on GIFT-Eval. The middle forecast (MASE) is unchanged everywhere, at 0.6978: the fix only repairs the absurd ranges.

Most of Chronos-2's gap to its peers on bitbrains_rnd turned out to be this bug. On the 5-minute medium task it scored 1.00, where comparable models score 0.55–0.62. With the fix it scores 0.67.

The held-out data had suggested a bigger jump, to about #59. The test period simply contains fewer empty-history windows than the held-out period did.

## What we learned

- **You can't tune a good model into a better one with tiny general tweaks.** It's already at its sweet spot.
- **You can find a specific thing it gets badly wrong and fix that surgically.** Editing removes obstacles; it doesn't add skills. The blog reached the same conclusion.
- **AI agents are good at finding those obstacles.** Our hand-built approach found the same bug only in a blurry, indirect way, while the agent pinpointed it and fixed it with 70× fewer changes.

## Honest caveats

- **A simpler fix exists:** a one-line rule, "if the history is empty, forecast a flat line", would fix the same bug in code. That's what we suggested upstream in [#544](https://github.com/amazon-science/chronos-forecasting/issues/544). The interesting part is that the agent **found** the bug from the model alone and fixed it **inside** the model.
- **How the test score was computed:** the fix only changes forecasts whose history has at most one real value. So we re-ran the 12 tasks that contain such cases and kept the official scores for the other 85. To confirm that's safe, we re-ran 7 unaffected tasks with the edited model, including multivariate ones. All 11 metrics matched the official scores to within 1.5 millionths (float rounding). Files: `results/nanfix_test_affected.csv`, `results/nanfix_test_unaffected_check.csv` and the merged `results/chronos2_nanfix_test_all_results.csv`.
- **Size of the win:** this is a modest improvement to one model on one benchmark, not a new state of the art.

---

## For the technically minded

<details>
<summary>Protocol, all results, files and how to reproduce</summary>

### Protocol (how we avoided fooling ourselves)
Every series is split in time into **fit → select → test** windows (`splits.py`, verified non-overlapping).
- **Fit:** where edits are designed.
- **Select:** where variants are compared.
- **Test (GIFT-Eval's official windows):** used once, at the end.

`dev_cache.py` freezes a small dev set (up to 24 series × 3 windows per config), so one evaluation takes 30–50 s. `run_eval.py` (t0-gamma) and `c2_official.py` (Chronos-2) reproduce official leaderboard numbers exactly on every config we checked.

### All edits we tried (scores on held-out select windows; below 1 is better)
| Model / edit | Weights changed | Where | CRPS | Configs better / worse |
|---|---:|---|---:|---|
| Chronos-2 + agent's empty-context neuron | 1,536 | block 11 MLP: `wi` row 11, `wo` column 11 | ×0.9779 | 6 / 0 |
| Chronos-2 + closed-form 9-quantile head edit | 110,736 | `output_patch_embedding.residual_layer`, 144 rows | ×0.9892 | 35 / 14 |
| t0-gamma + gated 90%-quantile head edit | 32,794 | `decoder.residual_layer`, 32 rows | ×0.9992 | 15 / 9 |
| t0-gamma, blog recipe (64 writer neurons in each of 2 layers, ±α) | 131,072 | MLP `down_proj`, layers 20–21 | ×1.003 to ×1.12 | worse at every strength |
| t0-gamma, closed-form median correction | — | decoder | ×1.0005 | shrinks to nothing under cross-validation |
| Post-hoc interval shrink, shape stretch, per-config fixes (no edit) | — | — | ×0.9998 to ×1.0099 | no gain, or overfits |

The Chronos-2 rows use the official notebook's settings (8,192 context, multivariate, joint batches of 100). The t0-gamma rows use fast dev settings.

`screen.py` measures how much of a model's own error its hidden state can predict on held-out windows. It found R² ≈ 0.1–0.25 here, which is why global corrections have little to work with. The edit formulas are in each script's docstring.

### The agent run (`agent_run/`)
An Opus 5.5 agent worked through `astra_tools.py`, a harness modelled on the blog's Experiment 8.
- **Budget:** 50 forward passes; reading weights and fit data was free.
- **Editable:** only `encoder.block.*` (the embeddings, output head, final norm and whole-tensor rescales were refused).
- **No training:** no gradients.
- **Held out:** one final check on the select windows; the test set was never visible.

It used 11 passes and about 30 minutes. `notes.md` is its notebook, `final_report.md` its report, `state.json` the pass log, `passes/` the per-config results of every pass, and `work/` its scripts, verbatim.

### Reproduce
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/GiftEval', repo_type='dataset', local_dir='data/gifteval')"
cp .env.example .env              # set GIFT_EVAL to the absolute path of data/gifteval
.venv/bin/python dev_cache.py     # frozen fit/select windows (~20 s)
.venv/bin/python -c "from chronos import Chronos2Pipeline; from apply_edits import to_full_npz; to_full_npz(Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model, 'edits/chronos2_nanfix_agent.npz', 'nanfix_full.npz')"
.venv/bin/python scan_empty_context.py --split test                 # which test configs the fix can change
.venv/bin/python c2_official.py test results/nanfix.csv --edit nanfix_full.npz --only bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H electricity/W kdd_cup_2018_with_missing/H
```

### Files
| File | Purpose |
|---|---|
| `edits/` + `apply_edits.py` | the three edits as compact deltas (the agent's fix is 8 KB), plus a loader |
| `astra_tools.py`, `agent_run/` | the agent harness and the complete agent log |
| `c2_official.py`, `run_eval.py`, `score.py` | official-settings evaluation and leaderboard scoring |
| `splits.py`, `dev_cache.py`, `dev_eval.py`, `diagnose.py`, `chronos_dev.py` | protocol, dev set, fast evaluation |
| `c2_edit.py`, `probe_edit.py`, `q90_edit.py`, `wsp_recipe.py`, `screen.py` | our closed-form edits, the blog recipe, the predictability screen |
| `calib_baseline.py`, `shape_baseline.py`, `percfg_baseline.py`, `extrapolate.py`, `settings_sweep.py` | the non-edit baselines that had to be beaten |
| `scan_empty_context.py` | finds windows with an all-missing context |
| `gifteval_data.py` | GIFT-Eval's data loader, vendored unchanged |
| `ref/`, `results/` | official leaderboard files and our result logs |

</details>

## Credits and licences
- **Our code:** Apache-2.0 (see `LICENSE`).
- **Models:** [Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon) and [t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma) (The Forecasting Company), both Apache-2.0.
- **Data:** the [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) benchmark data and leaderboard files (Salesforce), Apache-2.0.
- **Inspiration:** the method follows [*Sparks of Weight Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) and [its logs](https://github.com/RikiyaT/lm_understand-logs).
