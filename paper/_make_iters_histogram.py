"""Generate iters_used histogram figure for the paper.

Reads output/06_lgb_iterations.csv and writes paper/fig_iters_histogram.png.
Run from project root: python paper/_make_iters_histogram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "output" / "06_lgb_iterations.csv")
iters = df["iterations_used"].to_numpy()

p50 = float(np.percentile(iters, 50))
p95 = float(np.percentile(iters, 95))
ceiling = 2500
n_at_ceiling = int((iters == ceiling).sum())
n_under_100 = int((iters < 100).sum())

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.hist(iters, bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
ax.axvline(p50, color="#2ca02c", linestyle="--", linewidth=1.5,
           label=f"p50 = {int(p50)}")
ax.axvline(p95, color="#ff7f0e", linestyle="--", linewidth=1.5,
           label=f"p95 = {int(p95)}")
ax.axvline(ceiling, color="#d62728", linestyle=":", linewidth=1.8,
           label=f"ceiling = {ceiling}")
ax.set_xlabel("iterations used (early-stopping best_iteration)")
ax.set_ylabel("number of fits")
ax.set_title(
    f"Distribution of iterations_used across {len(iters)} walk-forward fits\n"
    f"(105 folds × 5 quantiles, portfolio level; "
    f"{n_at_ceiling} at ceiling, {n_under_100} under 100)"
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()

out_path = ROOT / "paper" / "fig_iters_histogram.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  → {out_path.relative_to(ROOT)}")
print(f"  n={len(iters)}, p50={int(p50)}, p95={int(p95)}, "
      f"at_ceiling={n_at_ceiling}, under_100={n_under_100}")
