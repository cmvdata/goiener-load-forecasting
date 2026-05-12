# Economic interpretation of feature importance

Aggregated across 5 champion models (per horizon × quantile, portfolio level).

## Group share of total gain

| Group | Share | Total gain |
|---|---:|---:|
| lag |  48.3% | 492922 |
| calendar |  31.1% | 317568 |
| weather |  20.5% | 209260 |
| regime |   0.0% | 0 |

## Reading the table
- **lag** features explain 48.3% of the total gain. In a residential portfolio this is consistent with the intra-day and intra-week patterns dominating short-term variability once the long-term level is stable.
- **calendar** is next (31.1%); these features capture systematic shifts the model needs to track between average days and unusual ones.
- The remaining groups together provide secondary signal that lifts the model above the persistence baseline. These shares are computed on synthetic sample data; the full-mode run is expected to reweight weather upward as real Spanish heating/cooling demand kicks in.
