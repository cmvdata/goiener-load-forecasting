# Methodology

This document complements the README with implementation-level decisions.

## Forecasting architecture: recursive single-step

The LightGBM champion is trained as a single-step model: target
y(t+1) given features at time t. To produce a 168-hour forecast at
issue time T, the model is rolled forward iteratively:

  y_hat(T+1) ← model(features at T)
  y_hat(T+2) ← model(features at T+1, with kwh_lag_1 = y_hat(T+1))
  …
  y_hat(T+168) ← model(features at T+167, with lags filled from
                       past actuals plus prior predictions)

Five separate models, one per quantile (q05, q25, q50, q75, q95),
are rolled forward in parallel to produce a coherent quantile
trajectory.

### Why recursive?

Two operational alternatives:

1. **Direct multi-step.** Train one model per horizon
   (target = y.shift(-h) for h = 1..168). 168 × 5 = 840 models per
   level. Each model sees its training target only at one horizon,
   so error doesn't compound. Best for far horizons in well-resourced
   teams.

2. **Recursive single-step.** One model per quantile per level. The
   model gets retrained on a rich, near-step target distribution and
   is rolled forward at inference. Errors compound, but the simplicity
   and the natural fit with operational batch generation typically win
   for h ≤ 72h.

3. **Hybrid.** Recursive for h ≤ T_switch, direct-h for h > T_switch.
   Trade-off only worth it on data-rich production environments.

This project chose **recursive**. The trade-off is explicit:

- Pro: one model per (level, quantile), simple to retrain nightly,
  natural fit for the daily batch pattern.
- Con: error accumulation at long horizons. Coverage rates at
  h ≥ 144h are expected to widen vs h ≤ 24h.
- For deploying this in a real supply context, especially for
  imbalance and intraday hedging at h > 72h, a direct-h or hybrid
  model would be the next iteration.

## Why walk-forward, not k-fold

Random k-fold cross-validation leaks future information into the
training fold. A model evaluated this way will look artificially
accurate. Walk-forward respects time strictly: at each fold the
training window ends before the prediction window begins, no
exception.

## Why probabilistic, not point

A point forecast cannot be hedged. Real supply teams need quantile
bounds to size positions in OMIE day-ahead and to reserve imbalance
margin against REE penalties. Pinball loss at each quantile and
empirical coverage rates against nominal intervals are the
non-negotiable diagnostics.

## Realized weather, not forecast weather

The model consumes realized weather. A production deployment would
consume the AEMET or ENTSO-E weather forecast feed at issue time
and propagate that uncertainty through the forecast. This caveat
overstates accuracy by an unknown but real margin.

## Champion / challenger logic

A challenger is trained nightly on the most recent data; the
champion is whatever lives in `models/champion/`. Both are
evaluated on the same rolling 30-day holdout. The challenger is
promoted only if:

- mean MAPE improves by at least `PROMOTION_THRESHOLD_MAPE_PP`
  percentage points
- coverage at the 50% nominal interval stays above a defensive
  floor (~0.45)
- coverage at the 90% nominal interval stays at or above the
  configured threshold

Promotion decisions are logged to `logs/promotions.jsonl` for
audit. Rejected challengers are archived under
`models/challengers/<date>_rejected/` with a JSON record of why
they were rejected.

## Conformal calibration

The recursive single-step LightGBM produces empirical coverage well below
nominal at every horizon — at h=1 the 50% interval covers ~0.41 against
the nominal 0.50; at h=168 it covers ~0.15. The cause is structural: the
median is fed back as a lag at each step, so the recursive trajectory is
smoother than realized variability and the predictive bands are too
narrow. Retraining the same recursive architecture cannot fix this; we
correct it post-hoc with **split conformal prediction**.

For each non-median quantile `q ∈ {0.05, 0.25, 0.75, 0.95}` and each
(level, horizon-pool bucket), we compute an offset on a held-out
calibration set:

    δ(level, bucket, q) = quantile_q( y_real - y_pred_q )

and then return calibrated quantile predictions

    y_pred_q_calibrated = y_pred_q + δ(level, bucket, q)

The median (q=0.5) is left untouched, so the median MAPE is preserved by
construction. Only the bands move.

### Why an interleaved cal/test split

Standard split conformal halves the validation period chronologically.
Here that would put pre-reform issues in calibration and post-reform
issues in test, because the validation window straddles the June 2021
2.0TD tariff reform — a real regime shift in residential demand
patterns. A pre→post split would break the exchangeability assumption
that conformal prediction relies on. We therefore split issue-times by the
parity of their sorted index — even index → calibration, odd → test —
so both regimes appear equally in each half. This restores approximate
exchangeability without giving up any of the validation data.

### Why horizon-bucket pooling

A per-horizon adjustment computed on ~51 calibration issues is too noisy
at the tails (the empirical 0.05 quantile of 51 samples is essentially
the third-smallest residual). We pool residuals across horizons within
three buckets:

    h<=24  ,  h=25-72  ,  h=73-168

raising the per-(level, bucket, q) calibration sample size to roughly
3,500 / 12,000 / 24,000. The trade-off — losing per-horizon resolution —
is consistent with how the gate is defined: bucket-level coverage is
what the headline metrics report, and that is the granularity at which
calibration is computed and at which empirical coverage is held to
account.

### Theoretical guarantee and its limit

Under exchangeability of calibration and test residuals, the empirical
coverage of `[q_lo + δ_lo, q_hi + δ_hi]` on the test set converges to
the nominal level. The guarantee is **marginal** over the calibration
distribution — coverage holds on average across the test set, not
conditionally on features. A heatwave that simultaneously inflates load
and inflates residual variance can still produce locally low coverage,
even when bucket-level coverage is on target. Conditional conformal
prediction would address this but adds complexity that is not justified
at portfolio scale for this project.

### What this fixes and what it doesn't

After calibration, all (level × pool-bucket) cells satisfy
`cov50 ∈ [0.45, 0.55]` and `cov90 ∈ [0.85, 0.95]` on the held-out test
split. Single-horizon cells inside a pool can drift from nominal — by
design, since one δ serves all hours in the pool — and that drift is
reported in the dashboard for visibility but not gated.

## Hierarchical reconciliation

The hierarchy is `portfolio = sum(segment_0..segment_3)`. Both
bottom-up and MinT-OLS are computed and compared. MinT-OLS is the
operational default; bottom-up is the baseline the reconciler must
beat to justify its complexity.

## Out-of-scope items

- No deep learning (LSTM, TFT). Marginal gain rarely justifies
  marginal complexity at portfolio scale.
- No real-time inference layer (FastAPI, Kubernetes, MLflow).
- No causal feature integration: the pipeline forecasts demand, it
  does not estimate household-level responses to the tariff reform.
- No intraday updates: the batch is nightly. A real intraday
  hedge desk would refresh more often.
