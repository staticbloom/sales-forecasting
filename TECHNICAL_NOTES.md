# Sales Forecasting

## Objective and Context

Forecast total Sales revenue from the supplied workbook. The [notebook](notebook/Sales-Data.ipynb) contains the analysis, saved results and figures.

## Data and Model Availability

Source data and the original Prophet/ETS files are excluded from Git because the serialized models contain observed sales history. Use the [parameter recipes](models/README.md) to train on an authorized `Date,Sales` CSV, then forecast with `predict.py --model`. Rebuilding the original study requires the matching [workbook](data/README.md). The `data/` directory retains its README and raw/processed placeholders. Source-row previews are omitted from the notebook; aggregate sales totals, evaluation tables and figures remain public.

## Dataset and Source

The source is `data/raw/Sales Data.xlsx`: 60,097 transaction lines plus product, region and salesperson lookups. The target is total recorded Sales, in source revenue units because the currency is unspecified. Source dates run from July 2021 through May 2025.

- **Weekly:** complete Sunday–Saturday weeks, a 13-week horizon and four expanding development folds.
- **Monthly:** complete calendar months, a three-month horizon and four chronological development folds.

## Methodology

Two earlier inner folds select model settings. Weekly selection uses mean development WAPE, with RMSE breaking ties. Monthly development evaluates adaptive selection across a smaller candidate pool. The notebook reports the exact configuration and family counts.

## Results

| Selected model | Evaluation | WAPE |
|---|---|---:|
| Weekly multiplicative Prophet | 13-week retrospective holdout | 23.87% |
| Monthly ETS, additive trend and multiplicative seasonality | Three-month retrospective holdout | 9.31% |

The [weekly](tables/weekly_retrospective_results.csv) and [monthly](tables/monthly_retrospective_results.csv) tables retain full precision. These are previously examined periods, not independent tests. The two aggregation levels do not support a direct accuracy ranking.

## Installation and Execution

The existing environment uses Python 3.14.7 and the pinned dependencies. Install first; the forecasting commands require locally restored or recreated original models:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python predict.py --frequency weekly --periods 13 --output outputs/weekly_predictions.csv
python predict.py --frequency monthly --periods 3 --output outputs/monthly_predictions.csv
```

These commands load the saved models and never fit or search. Without `--output`, predictions print as CSV. `--model` selects another compatible model package; `--frequency` must match it. Output files must be new. Forecast dates begin immediately after the artifact's training cutoff, which is **31 May 2025** for both submitted models. Asking for more periods does not update the model with recent sales, and horizons beyond 13 weeks/3 months have not been evaluated by the original study.

Explicit study rebuild, including optional cleanup:

```bash
python predict.py --rebuild-study
python predict.py --rebuild-study --clean-generated
```

Only the rebuild mode finds the original workbook and executes the notebook's forecasting calculation cells using their `forecast_step` metadata. It replaces generated study exports. Saved-model inference resolves models relative to the script and does not need the workbook or notebook. Keep one analysis notebook with unique step labels for rebuilding; its filename can change.

The full search can take several minutes. LightGBM and CatBoost are required at their pinned versions. To reproduce the notebook, use **Restart Kernel → Run All**.

### Reuse and Future Versions

`train.py` accepts a new CSV with exactly `Date,Sales`: timezone-free `YYYY-MM-DD` dates and finite nonnegative revenue in a consistent currency/unit. Multiple transaction rows per date are summed. Missing days raise an error; use `--missing-days zero` only when an omitted date really means no sales. Incomplete boundary weeks/months are excluded. The existing `data/processed/daily_sales.csv` illustrates the format; it is already used study data.

```bash
# Update history or use another business series with the same target definition.
python train.py new_daily_sales.csv --frequency weekly --output-dir runs/weekly_candidate_v2
python train.py new_daily_sales.csv --frequency monthly --output-dir runs/monthly_candidate_v2
python predict.py --frequency weekly --model runs/weekly_candidate_v2/weekly_prophet.json --output outputs/updated_weekly.csv

# Evaluate newly observed complete periods immediately after a saved cutoff.
python train.py later_daily_sales.csv --frequency monthly --evaluate-only --output-dir runs/monthly_evaluation_v1
```

The workflow uses the selected fixed recipes: multiplicative Prophet with annual Fourier terms for weekly totals, and additive-trend/multiplicative-seasonal ETS for monthly totals. It needs at least 117 complete weeks or 27 complete months: two seasonal cycles for fitting plus a 13-week/3-month chronological holdout. ETS requires strictly positive period totals. It compares the holdout forecast with seasonal naive, records WAPE/MAE/RMSE/bias, then refits the same recipe on all supplied history. The exported model therefore has a later origin than the fit used for evaluation; the report distinguishes them. No candidate search occurs.

New output directories contain a reload-checked model, input hash, specification, versions, aggregated series, evaluation predictions and report. Only raw-scale Prophet JSON and ETS joblib exports are supported; choosing another family in a future study requires a matching serializer and prediction implementation. New currencies, promotional patterns, structural breaks and very intermittent demand require a fresh model comparison. A successful refit is not proof that the old model family transfers well.

## Visualizations and Interpretation

`visuals/` contains revenue history, weekly/monthly retrospective forecasts, future forecasts and model comparisons. WAPE axes are percentages, revenue uses unspecified source units, and weekly/monthly charts describe distinct tasks. Forecast dates are anchored to the historical cutoff, not today.

## Repository Structure

- `notebook/Sales-Data.ipynb`: analysis, calculation cells and saved results.
- `predict.py`: saved-model inference and explicit original-study rebuild.
- `train.py`: fixed-recipe training on new daily data and subsequent-period evaluation.
- `data/raw/Sales Data.xlsx`: required original workbook.
- `requirements.txt`: pinned forecasting and notebook dependencies.

- `data/processed/`: enriched transactions and daily, weekly and monthly Sales.
- `tables/`: model comparisons, fold metrics, retrospective predictions, future forecasts, data-quality checks and timings.
- `visuals/`: EDA and forecast figures.
- `models/`: tracked parameter recipes and recreation notes; original `weekly_prophet.json` and `monthly_ets.joblib` remain local because they embed sales observations.

Rebuild cleanup is limited to generated files in these four locations. Keep future versions under `runs/`, outside those cleanup locations. Raw data, notebooks and source files are preserved. Original study candidate and backtest models stay in memory.

Use `load_selected_model(path)` and `predict_model(model, dates)` from the runner to load a trusted model and forecast dates immediately after its saved training cutoff. These models use all available history; the prediction tables preserve the retrospective evaluations.

## Limitations and Reproducibility

WAPE is the primary selection metric. Standard MAPE is undefined when an actual is zero; MASE uses period 52 weekly and 12 monthly. Runtime does not influence model selection.

The historical evaluation periods were already examined, so they are **locked retrospective holdouts**, not independent tests. Monthly aggregation smooths short-term variation; lower monthly errors do not establish lower weekly errors. Forecasts start after May 2025, the workbook's cutoff.

The unusual 2022 recording pattern, ambiguous order identifiers and unobserved business drivers limit interpretation. Broad model exploration on fewer than four annual cycles also adds selection uncertainty. Future data are needed for independent confirmation.


Package versions match the recorded local environment. Installation in a fresh environment has not been established.
