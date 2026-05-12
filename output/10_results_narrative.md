# Results narrative

At horizon 24h, the LightGBM median forecast achieves MAPE 4.02% versus persistence 4.97% and SARIMAX 12.09%.
At horizon 168h, the same model achieves MAPE 6.95% versus persistence 5.25% and SARIMAX 16.79%.

All metrics come from walk-forward validation: at each fold the model only sees data strictly prior to the prediction window. This is the same evaluation protocol that supply-side forecasting teams use because random k-fold leaks future information into the past and inflates apparent accuracy.
