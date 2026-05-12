# Data

This directory holds the raw GoiEner data, the weather data, and the synthetic
sample for testing.

## Sources

### GoiEner smart meter data (required for full mode)

- **DOI**: [10.5281/zenodo.7362094](https://doi.org/10.5281/zenodo.7362094)
- **Paper**: Quesada et al. (2024), *Scientific Data* 11(59).
- **License**: Creative Commons CC-BY 4.0 (no registration, no NDA).
- **Total size**: ~1.86 GB compressed, ~14 GB decompressed.

To download programmatically, run:

```bash
python code/00_download_goiener.py
```

This fetches four files into `data/raw/`:
- `metadata.csv` (5.6 MB)
- `imp-pre.tzst` (792 MB)
- `imp-in.tzst` (530 MB)
- `imp-post.tzst` (510 MB)

The download script supports HTTP Range resume in case of connection drops, and
retries with exponential backoff. Failed downloads can be resumed by re-running
the same command.

### Weather data (required for full mode)

- **API**: [Open-Meteo Archive](https://open-meteo.com/en/docs/archive-api)
- **Locations**: Bilbao, Pamplona, Madrid (representing 100% of GoiEner customer base by weight)
- **Period**: 2018-01-01 to 2022-06-30 (matches GoiEner data span)

To download:

```bash
python code/00_download_weather.py
```

No registration required. Free tier suffices (a few hundred requests).

## Synthetic sample for testing

A small synthetic sample (50 households, ~4.5 years) is provided in
`data/sample/` so the pipeline can be tested without downloading the full
dataset. To run on the sample:

```bash
export GOIENER_DATA_MODE=sample
python code/01_build_panel.py
```

### What the sample is good for — and what it isn't

The sample is built by `code/generate_sample.py` and **does inject** a
4-segment behavioral structure with weather sensitivity, weekly cycles, and
annual seasonality. As a result:

- **Segmentation** (`02_segment_households.py`) should recover the 4 clusters
  with high accuracy. The metadata includes a `_segment_truth` column for
  ground-truth comparison.
- **Forecasting models** will have real signal to learn (load shape, weather
  response, calendar effects). Walk-forward validation produces meaningful
  metrics, though absolute numbers are smaller than what you'd get on real
  data due to the smaller portfolio (50 vs 16,500 households).
- **Hierarchical reconciliation** works correctly because segment forecasts
  sum to portfolio.

What the sample **does not** capture:
- Real weather, real holidays, real lockdowns, real tariff reform behavior
- The full distribution of consumption patterns observed in 16,500 real households
- Realistic forecast horizons under genuinely uncertain weather

In short: the sample is **suitable for validating that the pipeline works
end-to-end** and for a meaningful first look at all outputs. It is **not** a
substitute for running on real GoiEner data when interpreting results
economically.

## Citation

> Quesada, C., Astigarraga, L., Merveille, C., & Borges, C. E. (2024). An
> electricity smart meter dataset of Spanish households: insights into
> consumption patterns. *Scientific Data*, 11(59).
> https://doi.org/10.1038/s41597-023-02846-0

## Files in this directory

```
data/
├── README.md                      ← this file
├── raw/                           ← gitignored; populated by 00_download_goiener.py
│   ├── metadata.csv
│   ├── imp-pre.tzst
│   ├── imp-in.tzst
│   ├── imp-post.tzst
│   ├── imp-pre/                   ← decompressed by 01_build_panel.py
│   ├── imp-in/
│   └── imp-post/
├── weather/                       ← gitignored; populated by 00_download_weather.py
│   ├── bilbao.parquet
│   ├── pamplona.parquet
│   └── madrid.parquet
└── sample/                        ← committed; synthetic 50-household test data
    ├── metadata.csv
    ├── imp-pre/  (50 CSVs)
    ├── imp-in/   (50 CSVs)
    └── imp-post/ (50 CSVs)
```
