"""
09_evaluate_challenger.py — Champion / challenger promotion logic
                            on the recursive trajectory.

The champion is whatever LightGBM artefact lives in models/champion/.
The challenger is freshly trained at script invocation time with
identical architecture (single-step + recursive) but on data extending
up to (holdout_start = max_timestamp - COMPARISON_WINDOW_DAYS).

Both models produce a 168-hour trajectory at each issue point in a
daily grid spanning the holdout. MAPE is averaged across every
(fold, target_h) pair — i.e., across every individual hourly forecast
in the holdout — so improvement at any horizon counts. Coverage is
likewise computed pooled across all pairs.

Promotion criteria (all must pass):
  1. mean MAPE improvement ≥ PROMOTION_THRESHOLD_MAPE_PP percentage points
  2. coverage at the 50% nominal interval ≥ C.MIN_COVERAGE_RATE_50
  3. coverage at the 90% nominal interval ≥ C.MIN_COVERAGE_RATE_90

Outputs:
  output/09_promotion_decision.json
  logs/promotions.jsonl
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C


_HERE = Path(__file__).parent
_l = importlib.util.spec_from_file_location("train_lgb", _HERE / "05_train_lightgbm.py")
train_lgb = importlib.util.module_from_spec(_l)
_l.loader.exec_module(train_lgb)


N_STEPS = 168
LEVEL = "portfolio"

POOL_BUCKETS = [
    ("h<=24", 1, 24),
    ("h=25-72", 25, 72),
    ("h=73-168", 73, 168),
]


def assign_pool_bucket(h: int) -> str:
    for label, lo, hi in POOL_BUCKETS:
        if lo <= h <= hi:
            return label
    raise ValueError(f"horizon_h={h} not covered by POOL_BUCKETS")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(C.CACHE_DIR / "features_portfolio.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_conformal_adjustments() -> pd.DataFrame | None:
    path = C.OUTPUT_DIR / "06b_conformal_adjustments.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def apply_conformal_to_wide(wide: pd.DataFrame, adj: pd.DataFrame, level: str) -> pd.DataFrame:
    """Add δ to non-median quantile columns. Median column is untouched.

    Both champion and challenger are evaluated with the same conformal
    adjustments because the production pipeline applies these to whatever
    model is currently deployed (08_run_daily_batch.py wires this in).
    Without this, every challenger fails the coverage gate by definition,
    since the raw recursive trajectory undercovers structurally.
    """
    if adj is None or adj.empty:
        return wide
    sub = adj[adj["level"] == level].copy()
    if sub.empty:
        return wide
    out = wide.copy()
    out["pool_bucket"] = out["horizon_h"].map(assign_pool_bucket)
    for q in C.QUANTILES:
        if q == 0.5:
            continue
        col = f"q{int(round(q*1000)):03d}"
        if col not in out.columns:
            continue
        delta_map = (
            sub[sub["quantile"] == q]
            .set_index("bucket")["adjustment"]
            .to_dict()
        )
        out[col] = out[col] + out["pool_bucket"].map(delta_map).fillna(0.0)
    out = out.drop(columns=["pool_bucket"])
    return out


def issue_points(holdout_start: pd.Timestamp, last_ts: pd.Timestamp,
                 step_h: int = 24) -> list[pd.Timestamp]:
    """Daily grid of issue points such that each issue + N_STEPS is in-range."""
    issues = []
    t = holdout_start
    cap = last_ts - pd.Timedelta(hours=N_STEPS)
    while t <= cap:
        issues.append(t)
        t = t + pd.Timedelta(hours=step_h)
    return issues


def train_quantile_suite(features: pd.DataFrame,
                         train_end: pd.Timestamp) -> dict:
    """Train one single-step model per quantile on data before train_end.

    Uses the same temporal-split + early-stopping protocol as the champion
    (script 05) so the promotion gate is an apples-to-apples comparison.
    """
    sup_full = train_lgb.make_supervised(features, train_lgb.SINGLE_STEP_HORIZON)
    target_cutoff = train_end - pd.Timedelta(hours=train_lgb.SINGLE_STEP_HORIZON)
    sup_filt = sup_full[sup_full["timestamp"] < target_cutoff]
    if sup_filt.empty:
        raise RuntimeError(f"Empty training set in 09 before {target_cutoff}")

    out = {}
    for i, q in enumerate(C.QUANTILES):
        booster, info = train_lgb.fit_with_early_stopping(
            sup=sup_filt,
            q=q,
            verify_dates=(i == 0),  # print temporal-split dates once
            label=f"09/challenger q={q:.2f}",
        )
        out[q] = booster
        print(
            f"  challenger q={q:.2f}  iters_used={info['iterations_used']:>4d}  "
            f"es_train={info['n_train']:,}  refit_full={info['n_train_full']:,}"
        )
    return out


def load_champion() -> dict:
    out = {}
    for q in C.QUANTILES:
        path = train_lgb.model_filename("portfolio", q)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing champion {path.name}; run code/05_train_lightgbm.py"
            )
        out[q] = lgb.Booster(model_file=str(path))
    return out


def evaluate_suite(boosters: dict, features: pd.DataFrame,
                   issues: list[pd.Timestamp]) -> pd.DataFrame:
    """Roll the booster suite forward N_STEPS at each issue point."""
    rows = []
    for issue in issues:
        traj = train_lgb.recursive_forecast_quantiles(
            boosters_by_q=boosters,
            features=features,
            issue_time=issue,
            n_steps=N_STEPS,
        )
        if traj.empty:
            continue
        # Pivot to wide on quantile for easier metrics
        wide = traj.pivot_table(
            index=["timestamp", "horizon_h"],
            columns="quantile",
            values="y_pred",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        wide["issue"] = issue
        # Rename quantile columns
        wide = wide.rename(columns={q: f"q{int(round(q*1000)):03d}" for q in C.QUANTILES})
        rows.append(wide)
    df = pd.concat(rows, ignore_index=True)

    # Attach actuals
    actual_lookup = features.set_index("timestamp")["kwh_total"]
    df["actual"] = df["timestamp"].map(actual_lookup)
    return df


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, mape=np.nan, rmse=np.nan,
                    coverage_50=np.nan, coverage_90=np.nan)
    actual = df["actual"].to_numpy()
    median = df["q500"].to_numpy()
    mask = (np.abs(actual) > 1e-6) & np.isfinite(median) & np.isfinite(actual)
    if not mask.any():
        return dict(n=0, mape=np.nan, rmse=np.nan,
                    coverage_50=np.nan, coverage_90=np.nan)
    mape = float(np.mean(np.abs((actual[mask] - median[mask]) / actual[mask])))
    rmse = float(np.sqrt(np.mean((actual[mask] - median[mask]) ** 2)))

    inside_50 = ((df["actual"] >= df["q250"]) & (df["actual"] <= df["q750"])).fillna(False)
    inside_90 = ((df["actual"] >= df["q050"]) & (df["actual"] <= df["q950"])).fillna(False)
    return dict(
        n=int(mask.sum()),
        mape=mape, rmse=rmse,
        coverage_50=float(inside_50.mean()),
        coverage_90=float(inside_90.mean()),
    )


def archive_models(src_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob("lgb_*.txt"):
        shutil.move(str(p), dest_dir / p.name)
        n += 1
    return n


def write_challenger(challenger: dict, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for q, booster in challenger.items():
        path = out_dir / f"lgb_portfolio_q{train_lgb.quantile_token(q)}.txt"
        booster.save_model(str(path))
        n += 1
    return n


def main():
    features = load_features()
    last_ts = pd.Timestamp(features["timestamp"].max())
    holdout_end = last_ts
    holdout_start = holdout_end - pd.Timedelta(days=C.COMPARISON_WINDOW_DAYS)
    print(f"\nHoldout window: [{holdout_start} → {holdout_end})")

    issues = issue_points(holdout_start, last_ts)
    print(f"  {len(issues)} issue points (daily, each issuing a {N_STEPS}h trajectory)")

    if len(issues) < 5:
        raise RuntimeError("Holdout too short for a meaningful comparison.")

    print("\nTraining challenger…")
    challenger = train_quantile_suite(features, holdout_start)

    print("Loading champion…")
    champion = load_champion()

    print("Evaluating both on the same trajectory grid…")
    df_champ = evaluate_suite(champion, features, issues)
    df_chall = evaluate_suite(challenger, features, issues)

    adj = load_conformal_adjustments()
    if adj is None:
        print("  [warn] 06b adjustments missing — coverage will reflect RAW bands.")
    else:
        print(f"  applying conformal adjustments ({len(adj)} rows) to both bands")
        df_champ = apply_conformal_to_wide(df_champ, adj, LEVEL)
        df_chall = apply_conformal_to_wide(df_chall, adj, LEVEL)

    m_champ = metrics(df_champ)
    m_chall = metrics(df_chall)

    delta_mape_pp = (m_champ["mape"] - m_chall["mape"]) * 100  # +ve = challenger wins

    decision_meets_mape = delta_mape_pp >= C.PROMOTION_THRESHOLD_MAPE_PP
    decision_meets_cov50 = m_chall["coverage_50"] >= C.MIN_COVERAGE_RATE_50
    decision_meets_cov90 = m_chall["coverage_90"] >= C.MIN_COVERAGE_RATE_90

    promote = decision_meets_mape and decision_meets_cov50 and decision_meets_cov90

    today = datetime.now().strftime("%Y-%m-%d")
    decision = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": holdout_end.isoformat(),
        "n_issues": len(issues),
        "n_horizon_steps": N_STEPS,
        "n_pairs": int(m_chall["n"]),
        "champion": m_champ,
        "challenger": m_chall,
        "delta_mape_pp": delta_mape_pp,
        "promotion_threshold_mape_pp": C.PROMOTION_THRESHOLD_MAPE_PP,
        "min_coverage_50": C.MIN_COVERAGE_RATE_50,
        "min_coverage_90": C.MIN_COVERAGE_RATE_90,
        "criteria": {
            "mape_improves_enough": bool(decision_meets_mape),
            "coverage_50_ok": bool(decision_meets_cov50),
            "coverage_90_ok": bool(decision_meets_cov90),
        },
        "promoted": bool(promote),
        "data_mode": C.DATA_MODE,
    }

    print("\n--- Decision summary ---")
    print(f"  champion   MAPE = {m_champ['mape']*100:5.2f}%  cov50={m_champ['coverage_50']:.3f}  cov90={m_champ['coverage_90']:.3f}")
    print(f"  challenger MAPE = {m_chall['mape']*100:5.2f}%  cov50={m_chall['coverage_50']:.3f}  cov90={m_chall['coverage_90']:.3f}")
    print(f"  Δ MAPE     = {delta_mape_pp:+.3f} pp  (need ≥ {C.PROMOTION_THRESHOLD_MAPE_PP} pp)")
    print(f"  cov50 ≥ {C.MIN_COVERAGE_RATE_50:.2f}? {decision_meets_cov50}")
    print(f"  cov90 ≥ {C.MIN_COVERAGE_RATE_90:.2f}? {decision_meets_cov90}")
    print(f"  promote?   = {promote}")

    if promote:
        archive_dir = C.MODELS_DIR / "challengers" / f"{today}_demoted"
        n_archived = archive_models(C.MODELS_DIR / "champion", archive_dir)
        n_promoted = write_challenger(challenger, C.MODELS_DIR / "champion")
        decision["archived_to"] = str(archive_dir.relative_to(C.PROJECT_ROOT))
        decision["files_archived"] = n_archived
        decision["files_promoted"] = n_promoted
        print(f"  → archived {n_archived} previous champion files to {archive_dir}")
        print(f"  → wrote {n_promoted} new champion files")
    else:
        reject_dir = C.MODELS_DIR / "challengers" / f"{today}_rejected"
        n_written = write_challenger(challenger, reject_dir)
        decision["rejected_to"] = str(reject_dir.relative_to(C.PROJECT_ROOT))
        decision["files_rejected"] = n_written
        print(f"  → wrote {n_written} challenger files to {reject_dir}")

    out_path = C.OUTPUT_DIR / "09_promotion_decision.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2, default=str)
    print(f"  → {out_path.name}")

    log_path = C.LOGS_DIR / "promotions.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision, default=str) + "\n")
    print(f"  → {log_path.relative_to(C.PROJECT_ROOT)}: appended 1 line")

    print("\n[ok] Challenger evaluation complete.")


if __name__ == "__main__":
    C._print_summary()
    main()
