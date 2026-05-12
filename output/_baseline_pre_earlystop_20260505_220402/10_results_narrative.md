# Results narrative

At horizon 24h, the LightGBM median forecast achieves MAPE 4.02% versus persistence 4.97% and SARIMAX 12.09%.
At horizon 168h, the same model achieves MAPE 7.40% versus persistence 5.25% and SARIMAX 16.79%.

## Hierarchical reconciliation

- h<=24 portfolio: original 5.04% / bottom-up 4.60% / MinT-OLS 4.89%
- h=168 portfolio: original 7.40% / bottom-up 7.74% / MinT-OLS 7.42%
- h=24 portfolio: original 4.02% / bottom-up 4.27% / MinT-OLS 4.01%
- h=25-72 portfolio: original 5.21% / bottom-up 4.79% / MinT-OLS 5.06%
- h=73-168 portfolio: original 6.29% / bottom-up 5.87% / MinT-OLS 6.14%

All metrics come from walk-forward validation: at each fold the model only sees data strictly prior to the prediction window. This is the same evaluation protocol that supply-side forecasting teams use because random k-fold leaks future information into the past and inflates apparent accuracy.
