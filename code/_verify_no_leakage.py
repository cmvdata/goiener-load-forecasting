"""Ad-hoc no-leakage verification for the walk-forward refit-on-full setup.

NOT part of the pipeline. Simulates the first fold of script 06 and prints
the Phase 1 (train+valid) and Phase 2 (refit-on-full) windows so we can
confirm Phase 2 stays strictly inside the fold and does NOT touch data
posterior to the fold cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `code/` importable so we get the same constants the pipeline uses.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import config as C  # noqa: E402

# Mirror the helper's split logic exactly (don't re-import the helper —
# we want to verify what _it_ does on the same data, not delegate to it).
import importlib.util  # noqa: E402

_l = importlib.util.spec_from_file_location("train_lgb", _HERE / "05_train_lightgbm.py")
train_lgb = importlib.util.module_from_spec(_l)
_l.loader.exec_module(train_lgb)


def check_one_fold(sup_full: pd.DataFrame, cutoff: pd.Timestamp,
                   train_window_days: int) -> dict:
    """Reproduce 06._train_lgb_quantile + fit_with_early_stopping's
    split logic for one fold and return its invariant flags."""
    train_start = cutoff - pd.Timedelta(days=train_window_days)
    train_end = cutoff
    target_cutoff = train_end - pd.Timedelta(hours=train_lgb.SINGLE_STEP_HORIZON)

    sup = sup_full[
        (sup_full["timestamp"] >= train_start)
        & (sup_full["timestamp"] < target_cutoff)
    ].sort_values("timestamp")
    if sup.empty:
        return {"cutoff": cutoff, "skipped": True}

    valid_cutoff = sup["timestamp"].max() - pd.Timedelta(days=C.LGB_VALID_DAYS)
    train_phase1 = sup[sup["timestamp"] <= valid_cutoff]
    valid_phase1 = sup[sup["timestamp"] > valid_cutoff]

    return {
        "cutoff": cutoff,
        "skipped": False,
        "n_train": len(train_phase1),
        "n_valid": len(valid_phase1),
        "n_refit": len(sup),
        "refit_max": sup["timestamp"].max(),
        "train_max": train_phase1["timestamp"].max() if len(train_phase1) else None,
        "valid_min": valid_phase1["timestamp"].min() if len(valid_phase1) else None,
        # Invariants:
        "cond_a_refit_max_lt_cutoff": sup["timestamp"].max() < cutoff,
        "cond_b_size_match": len(sup) == len(train_phase1) + len(valid_phase1),
        "cond_c_spans_match": (
            len(train_phase1) > 0 and len(valid_phase1) > 0
            and sup["timestamp"].min() == train_phase1["timestamp"].min()
            and sup["timestamp"].max() == valid_phase1["timestamp"].max()
        ),
        "cond_d_train_before_valid": (
            len(train_phase1) > 0 and len(valid_phase1) > 0
            and train_phase1["timestamp"].max() < valid_phase1["timestamp"].min()
        ),
    }


def main() -> int:
    feats = pd.read_parquet(C.CACHE_DIR / "features_portfolio.parquet")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    sup_full = train_lgb.make_supervised(feats, train_lgb.SINGLE_STEP_HORIZON)

    # Reproduce 06.generate_folds output for the whole walk-forward.
    start = pd.Timestamp(C.VALIDATION_START)
    end = pd.Timestamp(C.VALIDATION_END)
    step_days = C.WALK_STEP_DAYS
    train_window_days = C.TRAIN_WINDOW_DAYS
    horizon_h = 168
    cutoffs: list[pd.Timestamp] = []
    t = start
    cap = end - pd.Timedelta(hours=horizon_h)
    while t <= cap:
        cutoffs.append(t)
        t = t + pd.Timedelta(days=step_days)

    print(f"\nVerifying invariants across {len(cutoffs)} folds "
          f"({cutoffs[0]} -> {cutoffs[-1]})")
    print(f"  train_window_days = {train_window_days}, "
          f"LGB_VALID_DAYS = {C.LGB_VALID_DAYS}, "
          f"step_days = {step_days}")

    results = [check_one_fold(sup_full, c, train_window_days) for c in cutoffs]
    used = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]

    counters = {
        k: sum(1 for r in used if r[k])
        for k in ("cond_a_refit_max_lt_cutoff", "cond_b_size_match",
                  "cond_c_spans_match", "cond_d_train_before_valid")
    }

    print(f"\nFolds with non-empty sup: {len(used)} / {len(cutoffs)} "
          f"(skipped {len(skipped)})")
    print(f"  COND A passes: {counters['cond_a_refit_max_lt_cutoff']} / {len(used)}")
    print(f"  COND B passes: {counters['cond_b_size_match']} / {len(used)}")
    print(f"  COND C passes: {counters['cond_c_spans_match']} / {len(used)}")
    print(f"  COND D passes: {counters['cond_d_train_before_valid']} / {len(used)}")

    fold_first = used[0]
    fold_mid = used[len(used) // 2]
    fold_last = used[-1]
    for label, r in [("FIRST", fold_first), ("MIDDLE", fold_mid), ("LAST", fold_last)]:
        print(f"\n  [{label}] cutoff={r['cutoff']}  "
              f"refit_max={r['refit_max']}  "
              f"n_refit={r['n_refit']:,} = "
              f"{r['n_train']:,} + {r['n_valid']:,}")
        gap_h = (r['cutoff'] - r['refit_max']).total_seconds() / 3600
        print(f"           gap_to_cutoff = {gap_h:.0f}h")

    all_pass = all(v == len(used) for v in counters.values())
    if all_pass:
        print(f"\n[OK] All {len(used)} folds satisfy COND A,B,C,D. "
              "Refit-on-full stays inside the fold window everywhere.")
        return 0
    print("\n[FAIL] At least one fold violated an invariant — DO NOT EXECUTE.")
    for r in used:
        flags = [
            ("A", r["cond_a_refit_max_lt_cutoff"]),
            ("B", r["cond_b_size_match"]),
            ("C", r["cond_c_spans_match"]),
            ("D", r["cond_d_train_before_valid"]),
        ]
        bad = [name for name, ok in flags if not ok]
        if bad:
            print(f"  fold@{r['cutoff']}: failing {bad}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
