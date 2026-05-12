# Pipeline 2 — behavioural segmentation

k-means on standardized pre-2020 daily load shapes recovers **3 archetypes** across 10,531 households. The cluster structure is weak by design (silhouette < 0.20 across k ∈ {2..8}; see output/02_ksweep_diagnostic/) — residential consumption is a continuum, not a small discrete set of types. k=3 is the metric-supported choice (silhouette peak, elbow inflection at k=3) and the labels below are operational tags assigned by amplitude/level heuristics, not statistical discoveries.

## Archetypes

| Segment | Label | Households | % hh | % kWh | mean kWh/hh | peak h | amplitude (σ) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `evening-peak` | 5,624 | 53.4% | 46.4% | 3,966 | 22:00 | 3.14 |
| 1 | `high-baseload` | 729 | 6.9% | 12.3% | 8,128 | 00:00 | 1.82 |
| 2 | `midday-active` | 4,178 | 39.7% | 41.3% | 4,747 | 11:00 | 2.12 |

Centroid 24-hour profiles live in `output/02_segment_centroids.csv`; the overlay plot is at `output/02_segment_profiles.png`.
