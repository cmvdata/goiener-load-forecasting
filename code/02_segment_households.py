"""
02_segment_households.py — k-means segmentation of households by daily load shape.

Each eligible household is summarised by a 24-hour profile (mean kWh per
hour-of-day) computed on **pre-2020 data only**. Profiles are standardized
per household (subtract own mean, divide by own std) so clustering captures
shape, not level. k-means with N_SEGMENTS clusters yields four interpretable
archetypes.

Why pre-2020 only: the validation window starts mid-2020, and segmentation is
a feature input downstream. Using validation-period data here would leak
forward-looking information into the train side of every walk-forward fold.

Outputs:
  output/02_household_segments.csv     (user_hash, segment, total_pre_kwh)
  output/02_segment_centroids.csv      (segment, h00..h23 standardized profile)
  output/02_segment_profiles.png       (overlay plot of the 4 archetypes)

Sample-mode validation: data/sample/metadata.csv carries a `_segment_truth`
column (debug). We compute Adjusted Rand Index between recovered labels and
truth; expect ARI > 0.5 if the synthesizer's archetypes are recoverable.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tqdm import tqdm

import config as C


# 01_build_panel starts with a digit, so import it via importlib.
_BUILD_PATH = Path(__file__).parent / "01_build_panel.py"
_spec = importlib.util.spec_from_file_location("build_panel", _BUILD_PATH)
build_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_panel)


PRE_CUTOFF = pd.Timestamp("2020-01-01")

# K values evaluated by the --ksweep diagnostic.
KSWEEP_RANGE = list(range(2, 9))

# Subsample for silhouette computation on the full population (~10k rows
# would be O(n^2) full distance matrix). Fixed seed for reproducibility.
KSWEEP_SILHOUETTE_SAMPLE = 2000

# Operational-label heuristic thresholds (see operational_label).
# Above LABEL_AMPLITUDE_THRESHOLD the daily swing is informative enough
# to label by dominant period; below it we label by consumption level
# relative to the portfolio mean — peak hour ceases to discriminate
# between archetypes when the profile is roughly flat.
LABEL_AMPLITUDE_THRESHOLD = 2.0
LABEL_HIGH_CONSUMPTION_RATIO = 1.5
LABEL_LOW_CONSUMPTION_RATIO = 0.7


def daily_profile(series: pd.DataFrame) -> np.ndarray | None:
    """Return mean kWh per hour-of-day (length 24), pre-2020 only.

    Returns None if the household has no pre-2020 observations.
    """
    pre = series[series["timestamp"] < PRE_CUTOFF]
    if pre.empty:
        return None

    by_hour = pre.assign(hour=pre["timestamp"].dt.hour).groupby("hour")["kwh"].mean()
    profile = by_hour.reindex(range(24)).to_numpy(dtype=float)

    if np.isnan(profile).any():
        return None

    return profile


def standardize(profile: np.ndarray) -> np.ndarray | None:
    """Standardize a 24-vector by its own mean/std.

    Returns None for degenerate households with zero variance across the day
    (e.g. constant readings, which can't be clustered by shape).
    """
    mu = profile.mean()
    sd = profile.std()
    if sd <= 0:
        return None
    return (profile - mu) / sd


def compute_standardized_profiles() -> tuple[np.ndarray, list[str], list[float], np.ndarray, dict]:
    """Build the standardized profile matrix shared by main() and the k-sweep.

    Returns:
        X: standardized profiles, shape (n_households, 24).
        keep_users: user_hash list aligned with X rows.
        totals: pre-2020 total kWh per kept household.
        raw_profiles: same shape as X but in the original kWh scale.
        skip_stats: dict with counts of skipped households.
    """
    eligible_path = C.CACHE_DIR / "eligible_households.csv"
    if not eligible_path.exists():
        raise FileNotFoundError(
            f"Run code/01_build_panel.py first; missing {eligible_path}"
        )
    eligible = pd.read_csv(eligible_path)

    user_col = "user" if "user" in eligible.columns else eligible.columns[0]
    user_hashes = eligible[user_col].astype(str).tolist()

    print(f"\nComputing pre-2020 daily profiles for {len(user_hashes):,} households")

    profiles: list[np.ndarray] = []
    standardized: list[np.ndarray] = []
    keep_users: list[str] = []
    totals: list[float] = []
    skipped_no_pre = 0
    skipped_zero_var = 0

    for uh in tqdm(user_hashes, desc="Profiles"):
        series = build_panel.load_household_series(uh)
        if series is None:
            skipped_no_pre += 1
            continue

        prof = daily_profile(series)
        if prof is None:
            skipped_no_pre += 1
            continue

        std_prof = standardize(prof)
        if std_prof is None:
            skipped_zero_var += 1
            continue

        keep_users.append(uh)
        profiles.append(prof)
        standardized.append(std_prof)
        totals.append(float(series.loc[series["timestamp"] < PRE_CUTOFF, "kwh"].sum()))

    if not standardized:
        raise RuntimeError("No households had usable pre-2020 profiles.")

    return (
        np.vstack(standardized),
        keep_users,
        totals,
        np.vstack(profiles),
        {"no_pre": skipped_no_pre, "zero_var": skipped_zero_var},
    )


def run_ksweep(X: np.ndarray, ks: list[int] | None = None) -> pd.DataFrame:
    """Run KMeans across `ks` and report 3 internal validation indices + inertia.

    silhouette_score is computed on a fixed-seed subsample of the population
    to keep runtime bounded (O(n^2) on the full set is wasteful for a sweep).

    Outputs go to output/02_ksweep_diagnostic/ and never overwrite the
    canonical 02_household_segments.csv / 02_segment_centroids.csv.
    """
    if ks is None:
        ks = KSWEEP_RANGE
    out_dir = C.OUTPUT_DIR / "02_ksweep_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for k in tqdm(ks, desc="k-sweep"):
        km = KMeans(n_clusters=k, random_state=C.SEED, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(
                X, labels,
                sample_size=min(KSWEEP_SILHOUETTE_SAMPLE, X.shape[0]),
                random_state=C.SEED,
            )),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "n_iter": int(km.n_iter_),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ksweep_metrics.csv", index=False)
    df[["k", "silhouette"]].to_csv(out_dir / "silhouette_scores.csv", index=False)
    df[["k", "davies_bouldin"]].to_csv(out_dir / "davies_bouldin.csv", index=False)
    df[["k", "calinski_harabasz"]].to_csv(out_dir / "calinski_harabasz.csv", index=False)
    df[["k", "inertia"]].to_csv(out_dir / "inertia_elbow.csv", index=False)

    # Elbow plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["k"], df["inertia"], marker="o", linewidth=2, color="C0")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow plot — KMeans inertia vs k")
    ax.set_xticks(df["k"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "elbow_plot.png", dpi=120)
    plt.close(fig)

    # Three-metric panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_, col, title, color in zip(
        axes,
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        ["Silhouette (higher = better)",
         "Davies-Bouldin (lower = better)",
         "Calinski-Harabasz (higher = better)"],
        ["C0", "C1", "C2"],
    ):
        ax_.plot(df["k"], df[col], marker="o", linewidth=2, color=color)
        ax_.set_title(title)
        ax_.set_xlabel("k")
        ax_.set_xticks(df["k"])
        ax_.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette_plot.png", dpi=120)
    plt.close(fig)

    return df


def operational_label(
    profile_std: np.ndarray,
    mean_hh_kwh: float,
    portfolio_mean_hh_kwh: float,
) -> str:
    """Heuristic short label that adapts to the daily amplitude.

    For high-amplitude profiles (amplitude > LABEL_AMPLITUDE_THRESHOLD σ)
    the shape itself is distinctive, so we label by the dominant 5-period
    average of the standardized centroid (night / morning / midday /
    afternoon / evening).

    For low-amplitude profiles the daily swing is small and "peak hour" is
    no longer informative — what distinguishes the cluster is its baseline
    *level*, not its shape. We label by mean per-household consumption
    relative to the portfolio mean: high-baseload (e.g. permanent HVAC,
    EV charging, large dwellings), low-flat (light, intermittent users),
    or flat-medium in between.

    The full centroid stays in 02_segment_centroids.csv for any deeper read.
    """
    amplitude = float(profile_std.max() - profile_std.min())

    if amplitude > LABEL_AMPLITUDE_THRESHOLD:
        night = np.concatenate([profile_std[23:24], profile_std[0:6]]).mean()
        morning = profile_std[6:11].mean()
        midday = profile_std[11:15].mean()
        afternoon = profile_std[15:18].mean()
        evening = profile_std[18:23].mean()
        periods = {
            "night-active": night,
            "morning-skewed": morning,
            "midday-active": midday,
            "afternoon-active": afternoon,
            "evening-peak": evening,
        }
        return max(periods, key=periods.get)

    ratio = (
        mean_hh_kwh / portfolio_mean_hh_kwh
        if portfolio_mean_hh_kwh > 0
        else 1.0
    )
    if ratio > LABEL_HIGH_CONSUMPTION_RATIO:
        return "high-baseload"
    if ratio < LABEL_LOW_CONSUMPTION_RATIO:
        return "low-flat"
    return "flat-medium"


def characterize_segments(seg_df: pd.DataFrame, cent_df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment counts, kWh share, peak hour, and operational label.

    Output: output/02_segment_characterization.csv. Designed for the
    Pipeline 2 narrative — what does each archetype look like, how many
    households does it cover, how much energy. Centroid shapes themselves
    stay in 02_segment_centroids.csv.
    """
    total_n = len(seg_df)
    total_kwh = float(seg_df["total_pre_kwh"].sum())
    portfolio_mean_hh_kwh = float(seg_df["total_pre_kwh"].mean())
    hour_cols = [f"h{h:02d}" for h in range(24)]

    rows: list[dict] = []
    for seg_id in sorted(seg_df["segment"].unique()):
        sub = seg_df[seg_df["segment"] == seg_id]
        profile = cent_df.loc[
            cent_df["segment"] == seg_id, hour_cols
        ].to_numpy().flatten()
        mean_hh_kwh = float(sub["total_pre_kwh"].mean())
        rows.append({
            "segment": int(seg_id),
            "n_households": int(len(sub)),
            "pct_households": round(100 * len(sub) / total_n, 2),
            "pct_total_kwh": round(100 * float(sub["total_pre_kwh"].sum()) / total_kwh, 2),
            "mean_hh_total_kwh": round(mean_hh_kwh, 1),
            "peak_hour": int(profile.argmax()),
            "valley_hour": int(profile.argmin()),
            "amplitude_std": round(float(profile.max() - profile.min()), 3),
            "label": operational_label(profile, mean_hh_kwh, portfolio_mean_hh_kwh),
        })

    df = pd.DataFrame(rows)
    out_path = C.OUTPUT_DIR / "02_segment_characterization.csv"
    df.to_csv(out_path, index=False)
    print(f"  → {out_path.name}")

    print("\nSegment characterization:")
    for r in rows:
        print(
            f"  segment {r['segment']}: {r['label']:<18s}  "
            f"n={r['n_households']:>5d} ({r['pct_households']:5.1f}%)  "
            f"share kWh={r['pct_total_kwh']:5.1f}%  "
            f"peak@h{r['peak_hour']:02d}  amp_std={r['amplitude_std']:.2f}"
        )
    return df


def main_recharacterize():
    """Regenerate ONLY 02_segment_characterization.csv from existing
    02_household_segments.csv + 02_segment_centroids.csv. Used when the
    label heuristic changes and we don't need to re-cluster."""
    seg_path = C.OUTPUT_DIR / "02_household_segments.csv"
    cent_path = C.OUTPUT_DIR / "02_segment_centroids.csv"
    if not seg_path.exists() or not cent_path.exists():
        raise FileNotFoundError(
            f"Need both {seg_path.name} and {cent_path.name}; "
            "run code/02_segment_households.py first."
        )
    seg_df = pd.read_csv(seg_path)
    cent_df = pd.read_csv(cent_path)
    print(f"\nRe-characterizing {len(seg_df):,} households across "
          f"{seg_df['segment'].nunique()} segments")
    characterize_segments(seg_df, cent_df)
    print("\n[ok] Recharacterization complete.")


def main_ksweep():
    """Diagnostic-only: report optimal k per metric, never touch the canonical
    clustering outputs."""
    X, _, _, _, skip_stats = compute_standardized_profiles()
    print(f"\nRunning k-sweep over k ∈ {KSWEEP_RANGE} on {X.shape[0]} households")
    print(f"  skipped (no pre-2020 data): {skip_stats['no_pre']}")
    print(f"  skipped (zero daily variance): {skip_stats['zero_var']}")
    print(f"  silhouette sample size: {min(KSWEEP_SILHOUETTE_SAMPLE, X.shape[0])}")

    df = run_ksweep(X)

    print("\nK-sweep results:")
    print(df.to_string(index=False))

    best_sil = int(df.loc[df["silhouette"].idxmax(), "k"])
    best_db = int(df.loc[df["davies_bouldin"].idxmin(), "k"])
    best_ch = int(df.loc[df["calinski_harabasz"].idxmax(), "k"])
    print("\nOptimal k per metric:")
    print(f"  silhouette        (max): k={best_sil}  -> {df.loc[df['k']==best_sil, 'silhouette'].iloc[0]:.4f}")
    print(f"  davies_bouldin    (min): k={best_db}  -> {df.loc[df['k']==best_db, 'davies_bouldin'].iloc[0]:.4f}")
    print(f"  calinski_harabasz (max): k={best_ch}  -> {df.loc[df['k']==best_ch, 'calinski_harabasz'].iloc[0]:.4f}")

    if best_sil == best_db == best_ch:
        print(f"\n[ok] All three metrics agree: k_optimal = {best_sil}")
    else:
        print("\n[warn] Metrics disagree — human decision required.")

    print(f"\n  → output/02_ksweep_diagnostic/ksweep_metrics.csv")
    print(f"  → output/02_ksweep_diagnostic/elbow_plot.png")
    print(f"  → output/02_ksweep_diagnostic/silhouette_plot.png")
    print("\n[ok] K-sweep diagnostic complete.")


def main():
    X, keep_users, totals, raw_profiles, skip_stats = compute_standardized_profiles()

    print(f"\nClustering {X.shape[0]} households into {C.N_SEGMENTS} segments")
    print(f"  skipped (no pre-2020 data): {skip_stats['no_pre']}")
    print(f"  skipped (zero daily variance): {skip_stats['zero_var']}")

    km = KMeans(
        n_clusters=C.N_SEGMENTS,
        random_state=C.SEED,
        n_init=10,
    )
    labels = km.fit_predict(X)

    seg_df = pd.DataFrame({
        "user_hash": keep_users,
        "segment": labels.astype(int),
        "total_pre_kwh": totals,
    })
    seg_path = C.OUTPUT_DIR / "02_household_segments.csv"
    seg_df.to_csv(seg_path, index=False)
    print(f"  → {seg_path.name} ({len(seg_df):,} rows)")

    cent_df = pd.DataFrame(
        km.cluster_centers_,
        columns=[f"h{h:02d}" for h in range(24)],
    )
    cent_df.insert(0, "segment", range(C.N_SEGMENTS))
    cent_df["n_households"] = (
        seg_df["segment"].value_counts().reindex(range(C.N_SEGMENTS)).fillna(0).astype(int).values
    )
    cent_path = C.OUTPUT_DIR / "02_segment_centroids.csv"
    cent_df.to_csv(cent_path, index=False)
    print(f"  → {cent_path.name}")

    characterize_segments(seg_df, cent_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(24)
    for s in range(C.N_SEGMENTS):
        mask = labels == s
        if not mask.any():
            continue
        mean_profile = raw_profiles[mask].mean(axis=0)
        ax.plot(hours, mean_profile, marker="o", linewidth=2,
                label=f"segment {s} (n={int(mask.sum())})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean kWh per household")
    ax.set_title(f"Recovered behavioral archetypes (k={C.N_SEGMENTS}, pre-2020)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = C.OUTPUT_DIR / "02_segment_profiles.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  → {plot_path.name}")

    if C.DATA_MODE == "sample":
        md = pd.read_csv(C.METADATA_PATH)
        if "_segment_truth" in md.columns:
            truth = md.set_index("user")["_segment_truth"].astype(int)
            aligned = seg_df.set_index("user_hash").join(truth, how="inner")
            ari = adjusted_rand_score(aligned["_segment_truth"], aligned["segment"])
            print(f"\n[sample] Adjusted Rand Index vs ground truth: {ari:.3f}")
            if ari < 0.5:
                print("[warn] ARI < 0.5 — segmentation may not have recovered the archetypes")
            else:
                print("[ok] ARI ≥ 0.5 — archetype recovery is acceptable")
        else:
            print("[sample] metadata.csv has no _segment_truth column; skipping ARI check")

    print("\n[ok] Segmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ksweep", action="store_true",
        help="Run k-sweep diagnostic over k ∈ {2..8}; do not overwrite "
             "02_household_segments.csv / 02_segment_centroids.csv.",
    )
    parser.add_argument(
        "--recharacterize", action="store_true",
        help="Regenerate only 02_segment_characterization.csv from existing "
             "segment/centroid files (skip the 11-min profile rebuild).",
    )
    args = parser.parse_args()

    C._print_summary()
    if args.ksweep:
        main_ksweep()
    elif args.recharacterize:
        main_recharacterize()
    else:
        main()
