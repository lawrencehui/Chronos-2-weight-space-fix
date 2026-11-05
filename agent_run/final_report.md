# Final report: editing Chronos-2 weights for GIFT-Eval (FIT → SELECT)

**Final edit: `nanfix`.** FIT x0.9929 (4 better / 0 worse of 96). SELECT x0.9780 (6 better / 0 worse of 97,
largest loss x1.0005). I used 11 of the 50 passes, plus the one `final`.

## 1. Diagnosis
- **Average calibration is already good.** On FIT, the median per-config share of truths below q10/q50/q90 is
  0.110/0.514/0.899. I tried every simple output-space correction on base_fit.pkl against the truths, and none
  helped. Two of them were per-config oracles, so they are upper bounds:
  - global or per-quantile spread changes: best x0.9986
  - global shift of ±0.02: x1.010 / x1.016
  - pull toward the last observed level: x1.001 to x1.05
  - trend extrapolation: x1.04 to x1.24
  - mixing with seasonal naive: x1.009 to x1.07
  - clipping negative quantiles: x0.9995
  - smoothing quantiles over the horizon: x1.03 to x1.09
  - shift or spread fixes per regime (spiky, many zeros, extreme last level, rough, smooth, long horizon, short
    context)
  
  The per-config oracle bounds were x0.980 for the best shift and x0.984 for the best spread. The best shift and
  spread for a config are barely stable over time: the correlation between the early windows and the last window
  is 0.16 for shift and 0.14 for spread. So almost all of FIT's leftover miscalibration is noise.
- **Probes found nothing usable.** Ridge probes on block 9–11 states, MLP activations and heads, tested on held-out
  windows:
  - signed median miss: R² up to 0.13, but turning it into a shift made CRPS worse (x1.003 to x1.13)
  - spread miss: R² ≤ 0.035
  - CRPS-optimal shift: R² ≤ 0
- **The one real failure is the empty context.** If the last 2048 steps are all NaN, the scaling falls back to
  loc=0, scale=1. Every context token is then masked, and the future tokens see only [REG] and each other. Their
  states are deterministic, and the head returns q10 ≈ −1.9e5, q50 ≈ 0.07, q90 ≈ +2.0e5 (arcsinh ±12.9). When
  labels exist, which happens when a series starts or resumes inside the horizon, that one window dominates its
  config. For example, it is 24% of the CRPS of electricity/H/long. If those windows predicted 0 instead, FIT would
  score x0.9929.
- **The empty state is easy to separate.** At the block-11 MLP input (after the norm):
  - empty windows of a given horizon all share one state (cos 1.000)
  - the centres for different horizons agree closely (cos 0.95–0.999)
  - normal windows reach cos ≤ 0.57 with that state
  - a linear direction separates the two groups by more than 20 standard deviations

## 2. The edit (2 tensors, 1536 changed values)
I reused dead neuron j=11 in block 11's MLP. Its weights were about 1e-6, and it was never active.

- **Read side.** `encoder.block.11.layer.2.mlp.wi.weight[11,:]` = u, with 768 values and norm 0.144. It acts on
  x = g ⊙ h / rms(h), where h is block 11's MLP input. I estimated h as block11_out − W_o·mean(act), using window
  means from the read.
  - Let S be the covariance of the normal windows and Sr = S + 3·tr(S)/768·I.
  - w = Sr⁻¹(μ_empty − μ_normal) is the separating direction.
  - c = Sr⁻¹μ_normal is a near-constant direction. It stands in for a threshold, because the model has no bias
    terms.
  - u = w − t·c, where t puts the threshold midway between the largest normal projection and the smallest empty
    one. I then rescaled u so that empty windows give an activation of about 1.
- **Write side.** `encoder.block.11.layer.2.mlp.wo.weight[:,11]` = δ, with norm 33.4.
  - δ = mean over horizons k of (T_k − E_k).
  - T_k is the mean block-11 output of normal windows whose forecasts are confident (lowest-quartile spread) and
    near the context level.
  - E_k is the block-11 output for an empty context.
  - Block 11 is the last block, so the change reaches only the output head.

## 3. Evidence
- **FIT x0.9929, MASE x1.0000.** Exactly 4 configs change, and all of them improve: electricity/H/long x0.767,
  bitbrains_rnd/5T/medium x0.790, bitbrains_rnd/5T/long x0.841 and electricity/D/short x0.991. The other 92 configs
  are x1.0000. The result equals the oracle bound.
- **The detector fires only where it should.** I read the edited model with `read`. Neuron 11's activation is
  0.84–1.06 on all 47 empty windows and exactly 0 on all 3941 other windows. Block outputs for normal windows are
  bit-identical to the base model.
- **Dose-response and a random-write control.** Scaling the write by x0.5 or x2, or replacing it with a random
  vector of the same norm, gives the same CRPS (x0.8924 on the affected subset). The effect saturates: any push out
  of the broken state lands the forecast near 0. So the write direction is not what matters.
- **A random detector of the same norm is harmful.** On the full FIT set it scores x1.0037, with 5 better and
  26 worse. The specific detector is what keeps the edit surgical.

## 4. Scores
| | FIT | SELECT |
|---|---|---|
| CRPS | x0.9929 | **x0.9780** |
| MASE | x1.0000 | x1.0000 |
| better / worse | 4 / 0 | 6 / 0 |

- The SELECT gains are on bitbrains configs: fast_storage/5T/medium x0.563, rnd/5T/long x0.605, rnd/5T/medium
  x0.711 and rnd/H/short x0.739, among others.
- The only loss is electricity/W/short at x1.0005.
- I expected SELECT to be close to x1.000. More series turned out to resume after a gap of more than 2048 steps in
  the later period.

## 5. What failed
- The probe-based median and spread corrections, and all the output-space corrections listed in section 1. I
  simulated these offline with no passes.
- LASER rank reduction, which drops the small singular components of a weight matrix (passes 8–11). Scores
  excluding the 4 empty-context configs:
  - block 11 wi at rank 278: x1.0008
  - block 11 wo at rank 219: x1.0004
  - block 10 wi at rank 496: x0.9999, but its two halves of the dataset list disagree (x1.0007 and x0.9989)
  - block 10 wi at rank 258: x1.0012, with 15 better and 25 worse
  
  None met the bar I set before running them.
- The fix does nothing for contexts that are flat or mostly NaN. Those windows are already near-optimal, with an
  oracle gain of at most x0.9989.
