# Notebook: editing Chronos-2 weights for GIFT-Eval (FIT -> SELECT)

## Step 0 (free): orientation
Plan: read harness + chronos2 source, then study the base model's errors on FIT using base_fit.pkl and the truths
(free), before spending any pass. Budget 50 passes; I will aim to keep ~10 in reserve.

Architecture facts (from source): 12 T5-style blocks (RMSNorm pre-norm, no biases), each = time self-attn (RoPE, no
1/sqrt(d) scaling) -> group self-attn (across batch; with group_ids=arange it only attends to itself, so it's a
per-token linear map v->o on the normed state) -> ReLU MLP 768->3072->768. Final RMSNorm, then output ResidualBlock
maps each future-patch token to 21 quantiles x 16 steps in normalised (arcsinh(z-score)) space.
Note: group attention with independent series = softmax over a single key = 1, so it is exactly
h += o(v(LN(h))) — a pure per-token linear residual update. Editable!

## Step 1 (free): where does the base model fail on FIT? (scripts an_base.py, an_outlier.py, an_nan.py, an_horizon.py, an_oracle.py, an_persist.py)
Findings:
- Calibration is good on average: median per-config share below q10/q50/q90 = 0.110/0.514/0.899. A global
  spread scale in arcsinh space only helps x0.9986 (s=0.9); horizon-dependent spread also ~flat.
- Oracle upper bounds (per-config, using the truths): best constant median shift x0.9795, best spread x0.9838.
  So any config-level calibration edit can win at most ~2%; realistic edits get a fraction of that.
- A clear MECHANISTIC failure: windows whose last 2048 context steps are ALL NaN. Then loc=0, scale=1 and the model
  outputs a fixed distribution with q10≈-1.9e5, q50≈0.07, q90≈2.0e5 (arcsinh ±12.9!). When labels exist this one
  window dominates the config CRPS: electricity/H/long (24% of CRPS), bitbrains_rnd/5T/long (16%), /medium (21%).
  Oracle fix (predict 0 there) = geomean x0.9929 on FIT. Caveat: in a later period (SELECT) these items probably have
  data, so the fix may be neutral there.
- Weak persistence bias: truth is slightly closer to the last observed level than the median forecast
  (pooled slope 0.035-0.06, per-config median corr 0.11). Small.
Conclusion: no big broad miscalibration. Plan: spend 1 pass on `read base` to get internal states, then look for
(a) a mechanism for the empty-context case, (b) directions that predict signed errors (probe), (c) always-on / dead
neurons usable as a bias / repurposable unit (the model has no bias terms, so a constant offset needs a constant
input feature).

## Step 2 (free): weight statistics
Many MLP neurons are dead in the weights (wi row and wo column ~0): e.g. block 11 has ~670 wo columns <10% of median
norm. Dead neurons are free slots: I can install a new ReLU feature (wi row = read direction, wo col = write
direction) without disturbing anything else. Group-attention LN gains are ~0 in blocks 0-5 (group attn mostly off).

## Step 3 (pass 1): read base activations, blocks 9,10,11 in detail
Why: need per-window internal states (window-mean over forecast tokens) to (a) find always-on neurons (usable as a
bias), (b) probe for directions that predict the signed error, (c) see the empty-context state.
Pass 1 done (read base, blocks 9/10/11). Learned:
- No block-11 neuron is on in every window; ~695 block-11 neurons are never active at forecast tokens, 441 of them
  also have ~0 weights (free slots).
- Probe screen (ridge, held-out by last window and by config): block 9-10 states predict the signed median miss
  (share of truths below q50) with R^2 ~0.08, but turning the probe into a median shift makes CRPS WORSE at every
  gain (x1.003..x1.13). Spread-miss probe R^2 ~0.01. => the model is conditionally well calibrated; no usable
  "it knows it is wrong" direction.
- Simulated output-space corrections on FIT (all worse or flat): per-quantile-level spread (best x1.0000), global
  shift ±0.02 (x1.010/x1.016), persistence pull toward last level (x1.001..x1.05), linear-trend extrapolation
  (x1.04..x1.24), clipping negative quantiles of non-negative series (x0.9995), flat-recent-context -> constant (x1.000).
  The base model sits at a sharp optimum for every simple calibration knob I tried.
- Empty-context state is extremely distinct: at block-11 MLP input (after LN) all empty windows of a given horizon
  share one deterministic state (cos 1.000), centroids of different horizons agree (cos 0.95-0.999), while normal
  windows have cos <= 0.57 with it. An LDA direction separates them by >20 sd (window-mean states).
- Caveat on transfer: the empty-context windows WITH labels are series that start/resume inside the forecast window
  (electricity items 3/11, bitbrains_rnd items 4/18). In the later SELECT period those items have data, so this fix
  may be neutral on SELECT.

## Step 4 (pass 2): empty-context fix, edit "nanfix"
Edit: repurpose dead neuron 11 of block 11 MLP (wi row and wo column were ~1e-6). wi row = LDA direction
(shrinkage 3.0) between empty-context and normal MLP inputs, with a threshold built from the Fisher 'constant'
direction (no biases exist), scaled so empty tokens give activation ~1 and normal tokens ~-0.7 (ReLU->0).
wo column = mean over horizons of (T_k - E_k): T_k = mean block-11 output of normal windows with confident (lowest
quartile spread), near-zero-level forecasts; E_k = empty-context block-11 output. So for empty contexts the final
state is moved to a 'flat, confident forecast at the context level' state (level 0 = raw 0 since loc=0, scale=1).
Prediction: only configs with empty-context windows change (electricity/H/long, bitbrains_rnd/5T/long+medium,
electricity/D/short); geomean ~x0.993; every other config exactly x1.0000.
Result pass 2 (nanfix): FIT x0.9929, MASE x1.0000, better 4 / worse 0 of 96. Exactly the four predicted configs
change (electricity/H/long x0.767, bitbrains_rnd/5T/medium x0.790, /long x0.841, electricity/D/short x0.991);
all 92 others are x1.0000 to 4 decimals. Matches the oracle bound (x0.9929). Specific, mechanistic, works.

## Step 5 (free): last attempts at a broad correction before spending more passes
- Probe for the per-window CRPS-optimal median shift (grid search per window, ridge on block 9/10/11 states):
  held-out R^2 <= 0 on last windows; corrections x1.003..x1.048. Dead.
- Probes on MLP activations / head outputs: t50 R^2 up to 0.13 (block-10 MLP), miss80 R^2 <= 0.035. t50 is
  predictable mostly via ties at exact values (intermittent/zero series, y == q50), which do not move CRPS.
- Seasonal-naive mixing (5-20%) x1.009..x1.07; long-horizon mean reversion either sign worse; trend extrapolation
  restricted to short contexts ~x1.000 (helps hospital/m4_monthly, hurts car_parts/m4_yearly/electricity W).
Conclusion: outside the empty-context bug, the base model is at a sharp optimum of every output-space correction I
can express, conditional or not. Next: controls + dose-response for nanfix, then a small blind screen of internal
edits with strict acceptance rules.

## Step 6 (passes 3-6): nanfix controls and dose-response (restricted with --only to the affected configs + a few)
- C1 "same detector, random write": wo column replaced by a random vector of the same norm. Tests whether the
  target-state direction matters.
- C2 "random detector, same write": wi row random with the same norm. Tests specificity (expect broad damage,
  since a random direction fires on ~half of all tokens).
- Dose: write vector x0.5 and x2.
Results passes 3-6:
- C1 random write (same detector): x0.8924 on the 6-config subset = identical to nanfix (x0.8924). Dose x0.5 and x2:
  also x0.8924. The affected configs' calibration columns differ slightly (below_q90 0.921-0.928) but CRPS is the
  same: once the empty-context state is pushed out of its pathological regime, the forecast lands near raw 0 and
  the loss is |y|-dominated whatever the exact write. => The write direction is not what matters; the DETECTOR is.
- C2 random detector (same norm, same write), full FIT: x1.0037, better 5 / worse 26. A random read direction fires
  on ordinary tokens and damages broadly; the LDA detector is what makes the edit surgical (92 configs exactly x1.0000).
Free analyses meanwhile: per-config CRPS-optimal shift/spread are barely stable over time (corr 0.16 / 0.14 between
early windows and the last window of FIT), quantile smoothing over horizon x1.03-1.09, mostly-NaN-context spread
x0.9989 at best, regime-specific (spiky, zeros, last-level extreme, rough/smooth, long horizon, short context)
shift/spread corrections never consistent between early and last windows. => FIT residual miscalibration is mostly
noise; any broad correction fitted on FIT would not transfer.

## Step 7 (pass 7): verify detector specificity directly
read nanfix --blocks 11: neuron 11's window-mean activation should be >0 only for the empty-context windows.
Result pass 7: neuron 11 window-mean activation = 0.84..1.06 on the 47 empty-context windows and exactly 0 on all
3941 other windows; block10/11 outputs of normal windows are bit-identical to base. Detector specificity verified.

## Step 8 (passes 8-10): one principled broad-edit hypothesis, LASER-style rank reduction
Why: in LLMs, dropping the small singular components of late MLP matrices (LASER, Sharma et al. 2023) sometimes
improves held-out accuracy ("denoising"). It is a closed-form SVD edit with no fitting. Prior for a regression
model is low, so I pre-register: adopt only if FIT < x0.997, wins >= 2x losses, and consistent on both halves of
the dataset list. Spectra are spread (95% energy at rank 219-278 in block 11, ~500 in blocks 6-10), so I start
with the mild 95%-energy truncation: (a) block 11 wi -> rank 278, (b) block 11 wo -> rank 219, (c) block 10 wi ->
rank 496.
Results passes 8-10 (LASER, mild 95%-energy truncation; scores also shown excluding the 4 empty-context configs,
which any big change to block 10/11 also "fixes" by accident):
- b11 wi r278: x1.0011 (non-empty x1.0008, better 1 / worse 9). Worse.
- b11 wo r219: x0.9996 but non-empty x1.0004 (1/3); its gains are only the empty-context configs. Worse.
- b10 wi r496: x0.9994, non-empty x0.9999 (13/6), halves A x1.0007 vs B x0.9989 — inconsistent. Not adopted.
Next (pass 11): dose check on the only non-negative one, b10 wi at 80% energy (rank 258). If it does not give a
consistent gain on both halves, LASER is closed.
Result pass 11: b10 wi r258 (80% energy): x0.9999 overall but non-empty x1.0012, better 15 / worse 25, both halves
worse (x1.0014 / x1.0010). Stronger truncation = worse: no denoising benefit. LASER closed.

## Step 9: decision
Only nanfix holds up: mechanistic diagnosis, surgical (1 dead neuron, 1536 values), verified specificity (fires on
47/47 empty windows, 0/3941 others), 4 wins / 0 losses, matches its oracle bound. Everything aimed at broad gains
(probes, regime corrections, LASER) failed on FIT or was inconsistent, and per-config optima are not stable over time
(corr ~0.15), so I stop spending passes (11/50 used) rather than search more and overfit FIT.
Expectation for SELECT: the empty-context windows with labels in FIT are series that start/resume inside the
forecast window; in a later period those items mostly have data, so SELECT is likely ~x1.000 (cannot hurt: the
detector is silent on every non-empty window).
Running `final nanfix`.

## Step 10: FINAL (SELECT) result for nanfix
SELECT x0.9780, MASE x1.0000, better 6 / worse 0 (>0.5%) of 97. Changed configs: bitbrains_fast_storage/5T/medium
x0.563, bitbrains_rnd/5T/long x0.605, bitbrains_fast_storage/5T/long x0.686, bitbrains_rnd/5T/medium x0.711,
bitbrains_rnd/H/short x0.739, bitbrains_fast_storage/H/short x0.939, electricity/W/short x1.0005 (tiny loss: likely
a large-valued series starting inside the horizon, where the old ±2e5 band happened to be closer than ~0).
My expectation (~x1.000) was too pessimistic: in the later period more bitbrains VMs come back after >2048 missing
steps, so the empty-context bug hits SELECT harder than FIT. All other 90 configs x1.0000.
