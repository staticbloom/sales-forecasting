# Forecast model artifacts

`weekly_prophet.json` and `monthly_ets.joblib` are excluded from Git: their serialized objects contain the observed 204-week/47-month source series. Dataset redistribution rights are unverified.

`selected_recipes.json` contains the selected model parameters. A public checkout can run `python train.py new_daily_sales.csv --frequency monthly --output-dir runs/monthly_candidate` (or weekly) without the original models. It evaluates a chronological holdout then exports a full-history refit. Use `--model` to forecast from that new artifact.

To recreate the original models, obtain the authorized matching workbook and use `python predict.py --rebuild-study`. This reruns the full study and replaces generated exports. Alternatively restore matching authorized local artifacts. Saved-model evaluation and the default forecasting commands require the matching model files; the script explains their absence. Load only trusted joblib files.
