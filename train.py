"""Fit the fixed forecasting recipes to new daily sales, or evaluate a saved forecast."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cmdstanpy
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

from predict import SETTINGS, load_selected_model, predict_model

ROOT = Path(__file__).resolve().parent


def load_series(path, frequency, missing_days):
    data = pd.read_csv(path)
    if list(data.columns) != ["Date", "Sales"] or data.empty:
        raise ValueError("Supply a nonempty CSV with Date,Sales columns in that order")
    dates = pd.to_datetime(data.Date, format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be complete timezone-free calendar dates")
    amounts = pd.to_numeric(data.Sales, errors="raise")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Sales must be finite nonnegative revenue in consistent units")
    daily = pd.Series(amounts.to_numpy(), index=dates).groupby(level=0).sum().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    gaps = calendar.difference(daily.index)
    if len(gaps) and missing_days == "error":
        raise ValueError(f"{len(gaps)} missing days; supply them or explicitly choose --missing-days zero")
    daily = daily.reindex(calendar, fill_value=0.0)
    buckets = daily.resample(SETTINGS[frequency][0]).agg(["sum", "size"])
    complete = buckets["size"].eq(7 if frequency == "weekly" else buckets.index.days_in_month)
    series = buckets.loc[complete, "sum"].rename("Sales")
    if series.empty:
        raise ValueError("No complete weekly/monthly periods remain")
    return series, {"missing_days_filled_with_zero": len(gaps), "partial_boundary_periods_dropped": int((~complete).sum())}


def fit_model(series, specification, frequency):
    scale = max(float(np.std(series.to_numpy())), 1e-8)
    values = series / scale
    with threadpool_limits(limits=1):
        if specification["kind"] == "prophet":
            model = Prophet(
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                uncertainty_samples=0, seasonality_mode=specification["mode"],
                changepoint_prior_scale=specification["cps"], seasonality_prior_scale=specification["sps"],
                n_changepoints=specification["ncp"], changepoint_range=specification["cpr"],
            )
            model.add_seasonality("annual", period=365.25, fourier_order=specification["K"])
            if specification["extra"]:
                model.add_seasonality("subannual", period=7 * specification["extra"], fourier_order=1)
            with cmdstanpy.disable_logging():
                model.fit(pd.DataFrame({"ds": series.index, "y": values.to_numpy()}), seed=42, iter=1500, timeout=45, show_console=False)
        elif specification["kind"] == "ets":
            if (values <= 0).any():
                raise ValueError("The multiplicative ETS recipe needs strictly positive period totals")
            estimator = ExponentialSmoothing(
                values, trend=specification["trend"], seasonal=specification["seasonal"],
                damped_trend=specification["damped"], seasonal_periods=specification["period"],
                initialization_method="estimated",
            )
            model = estimator.fit(optimized=True, use_brute=False, method="L-BFGS-B", minimize_kwargs={"options": {"maxiter": 500}})
            if not model.mle_retvals.success:
                model = estimator.fit(optimized=True, use_brute=False, method="SLSQP", minimize_kwargs={"options": {"maxiter": 500}})
            if not model.mle_retvals.success:
                raise ValueError("ETS optimization did not converge")
        else:
            raise ValueError("Only the selected raw Prophet and ETS recipes are supported")
    return {"model": model, "specification": specification, "scale": scale, "frequency": frequency, "training_end": series.index[-1]}


def save_model(fitted, path):
    if fitted["specification"]["kind"] == "prophet":
        payload = json.loads(model_to_json(fitted["model"]))
        payload["sales_metadata"] = {k: v for k, v in fitted.items() if k != "model"}
        payload["sales_metadata"]["training_end"] = str(fitted["training_end"])
        path.write_text(json.dumps(payload) + "\n")
    else:
        joblib.dump(fitted, path)


def forecast_metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    denominator = float(np.abs(actual).sum())
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "wape_pct": float(100 * np.abs(error).sum() / denominator) if denominator else None,
            "bias": float(error.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Daily revenue or transaction rows, Date,Sales")
    parser.add_argument("--frequency", choices=SETTINGS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--missing-days", choices=["error", "zero"], default="error")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the saved model on complete periods immediately after its cutoff")
    parser.add_argument("--model", type=Path, help="Saved model; in training mode its specification supplies the fixed recipe")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    filename = "weekly_prophet.json" if args.frequency == "weekly" else "monthly_ets.joblib"
    reference_path = args.model or ROOT / "models" / filename
    # Recipes contain parameters only; serialized forecasts also contain source history.
    if not reference_path.exists() and args.model is None and not args.evaluate_only:
        recipes = json.loads((ROOT / "models/selected_recipes.json").read_text())
        reference = {"frequency": args.frequency, "specification": recipes[args.frequency]}
    else:
        reference = load_selected_model(reference_path)
    if reference["frequency"] != args.frequency or reference["specification"]["tx"] != "raw":
        raise ValueError("Use a matching frequency and raw-scale recipe")
    series, data_audit = load_series(args.input_csv, args.frequency, args.missing_days)
    _, horizon, season = SETTINGS[args.frequency]
    if args.evaluate_only:
        test = series
        fitted = reference
    else:
        if len(series) < 2 * season + horizon:
            raise ValueError(f"Need at least {2 * season + horizon} complete periods (two seasons plus holdout)")
        training, test = series.iloc[:-horizon], series.iloc[-horizon:]
        fitted = fit_model(training, reference["specification"], args.frequency)
    prediction = predict_model(fitted, test.index)
    report = {
        "status": "evaluation_only" if args.evaluate_only else "candidate_not_promoted",
        "source_filename": args.input_csv.name,
        "source_sha256": hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        "reference_model_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path.exists() else None,
        "recipe_source": str(reference_path.name) if reference_path.exists() else "models/selected_recipes.json",
        "frequency": args.frequency, "specification": fitted["specification"],
        "evaluation_training_end": str(fitted["training_end"]),
        "test_start": str(test.index[0]), "test_end": str(test.index[-1]), "test_periods": len(test),
        "test_metrics": forecast_metrics(test, prediction), "data_audit": data_audit,
        "evaluation_note": "A single chronological holdout; no search. Independence of supplied data requires source verification.",
        "versions": {name: version(name) for name in ["numpy", "pandas", "prophet", "statsmodels", "joblib"]},
    }
    if not args.evaluate_only:
        report["seasonal_naive_metrics"] = forecast_metrics(test, np.resize(training.iloc[-season:].to_numpy(), horizon))
        # Refit only after measuring the fixed recipe; this export has a later origin than the evaluated fit.
        full_fit = fit_model(series, fitted["specification"], args.frequency)
        report["export_training_end"] = str(series.index[-1])
        report["export_note"] = "Refitted on all supplied history; test metrics describe the earlier-origin fit, not unseen accuracy of this final refit."
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not args.evaluate_only:
        path = args.output_dir / filename
        save_model(full_fit, path)
        future = pd.date_range(series.index[-1], periods=horizon + 1, freq=SETTINGS[args.frequency][0])[1:]
        np.testing.assert_allclose(predict_model(load_selected_model(path), future), predict_model(full_fit, future), rtol=1e-10)
        report["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pd.DataFrame({"Date": test.index, "Actual Sales": test.to_numpy(), "Predicted Sales": prediction}).to_csv(args.output_dir / "evaluation_predictions.csv", index=False)
    series.rename_axis("Date").to_csv(args.output_dir / "period_sales.csv")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {report['status']} to {args.output_dir}")


if __name__ == "__main__":
    main()
