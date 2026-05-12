# `code/_archive/`

Archived pipeline scripts that were evaluated and dropped during the
project's lifecycle. Kept here as part of the audit trail rather than
deleted, so the decision can be reviewed if the architecture is ever
revisited.

## Contents

### `07_hierarchical_reconcile.py`

Hierarchical reconciliation (BottomUp and MinT-OLS, via Nixtla
`hierarchicalforecast`) was evaluated and discarded after the project's
reframe to portfolio-only forecasting in **May 2026**. With a single
forecast level, there is no hierarchy to reconcile; the script became
unreachable from the active pipeline (`05 → 06 → 06b → 09 → 10`).

Empirical justification at the time of removal:

- BottomUp delivered marginal portfolio-MAPE improvements at
  `h = 25–72` (~0.4 pp) but degraded at `h ≤ 24` and `h = 168`.
- MinT-OLS gave a small benefit at `h = 24` (-0.1 pp vs original)
  but no benefit at the longer horizons.

Both are documented in `output/_archive_pre_reframe_20260506_*/07_*.csv`.

The companion of this script in `code/08_run_daily_batch.py` was
rewritten on the same day to drop the reconciliation block entirely.
The `hierarchicalforecast` and `utilsforecast` packages were removed
from `requirements.txt` as part of the same cleanup.
