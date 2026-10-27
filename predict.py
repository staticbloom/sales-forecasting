#!/usr/bin/env python3
"""Forecast from a saved model, or explicitly rebuild the original notebook study.

Calculation cells are identified by their unique forecast_step metadata.
Only the two selected full-history models are saved.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from prophet.serialize import model_from_json, model_to_json
from threadpoolctl import threadpool_limits
from IPython.utils.io import capture_output

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

METRICS = ["MAPE", "WAPE", "sMAPE", "MAE", "RMSE", "MASE", "Bias", "Bias_pct"]
SETTINGS = {"weekly": ("W-SAT", 13, 52), "monthly": ("ME", 3, 12)}
OUTPUT_DIRS = ("data/processed", "tables", "visuals", "models")


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for parent in (start, *start.parents, Path(__file__).resolve().parent):
        if ((parent / "data/raw/Sales Data.xlsx").is_file()
                and (parent / "predict.py").is_file()):
            return parent
    raise FileNotFoundError("Could not find the project. Expected data/raw/Sales Data.xlsx and predict.py.")


def prepare_directories(root, clean=False):
    """Only known generated directories are eligible for cleanup; never follow links."""
    root = Path(root).resolve()
    allowed = {".csv", ".png", ".jpg", ".svg", ".json", ".joblib", ".pkl", ".pickle",
               ".ubj", ".cbm", ".bin", ".npz", ".npy", ".txt", ".log"}
    planned = []
    for relative in OUTPUT_DIRS:
        directory = root / relative
        assert directory != root and root in directory.parents
        for parent in (directory, *directory.parents):
            if parent.is_symlink():
                raise ValueError(f"Refusing symlinked output directory: {parent}")
        for path in directory.rglob("*") if directory.exists() else []:
            if path.is_symlink():
                raise ValueError(f"Refusing symlinked generated asset: {path}")
            protected = (path.name.lower().startswith("readme") or path.name in {"requirements.txt", "selected_recipes.json"}
                         or any(part.startswith(".") for part in path.relative_to(directory).parts))
            if path.is_file() and (path.name == ".DS_Store" or (path.suffix in allowed and not protected)):
                planned.append(path)
    if clean:
        for path in planned:
            print(f"Clearing {path.relative_to(root)}")
            path.unlink()
        for relative in OUTPUT_DIRS:
            directory = root / relative
            for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
    for relative in OUTPUT_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)


def load_data(root):
    """Validate the original workbook and enrich lines without deleting or correcting rows."""
    path = root / "data/raw/Sales Data.xlsx"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    required = {
        "Sales": ["SalesOrderNumber", "OrderDate", "ProductKey", "EmployeeKey",
                  "SalesTerritoryKey", "Quantity", "Unit Price", "Sales"],
        "Product": ["ProductKey", "Product", "Color", "Subcategory", "Category"],
        "Region": ["SalesTerritoryKey", "Region", "Country", "Group"],
        "Salesperson": ["EmployeeKey", "EmployeeID", "Salesperson", "Title", "UPN"],
    }
    for name, columns in required.items():
        assert name in sheets and set(columns).issubset(sheets[name]), f"Invalid {name} schema"
    sales = sheets["Sales"].copy()
    if pd.api.types.is_numeric_dtype(sales.OrderDate):
        sales["OrderDate"] = pd.to_datetime(sales.OrderDate, unit="D", origin="1899-12-30")
    else:
        sales["OrderDate"] = pd.to_datetime(sales.OrderDate, errors="raise")
    for column in ["Sales", "Quantity", "Unit Price"]:
        sales[column] = pd.to_numeric(sales[column], errors="raise").astype(float)
    assert len(sales) == 60097, "This study expects the supplied 60,097-line workbook."
    assert sales[required["Sales"]].notna().all().all()
    assert np.isfinite(sales[["Sales", "Quantity", "Unit Price"]]).all().all()
    assert sales[["Sales", "Quantity", "Unit Price"]].gt(0).all().all()
    assert sales.OrderDate.eq(sales.OrderDate.dt.normalize()).all()
    assert sales.OrderDate.min() == pd.Timestamp("2021-07-01")
    assert sales.OrderDate.max() == pd.Timestamp("2025-05-31")
    cents = np.rint(sales.Sales * 100).astype("int64")
    assert cents.sum() == 8958453785
    mismatch = ~np.isclose(sales.Sales, sales.Quantity * sales["Unit Price"], atol=.01, rtol=1e-6)
    enriched = sales.assign(source_row=np.arange(len(sales)) + 2)
    for name, key in [("Product", "ProductKey"), ("Region", "SalesTerritoryKey"),
                      ("Salesperson", "EmployeeKey")]:
        lookup = sheets[name][required[name]]
        assert lookup[key].is_unique and lookup[key].notna().all()
        assert sales[key].isin(lookup[key]).all(), f"Unmatched {key}"
        enriched = enriched.merge(lookup, on=key, how="left", sort=False, validate="many_to_one")
    pd.testing.assert_frame_equal(enriched[sales.columns], sales)
    assert enriched.source_row.tolist() == list(range(2, len(sales) + 2))
    daily = sales.groupby("OrderDate").Sales.sum().reindex(
        pd.date_range(sales.OrderDate.min(), sales.OrderDate.max(), freq="D"), fill_value=0.)
    daily.index.name = "Date"
    series, partial = {}, {}
    for name, (freq, _, _) in SETTINGS.items():
        buckets = daily.resample(freq).agg(["sum", "size"])
        complete = buckets["size"].eq(7) if name == "weekly" else buckets["size"].eq(buckets.index.days_in_month)
        series[name] = buckets.loc[complete, "sum"].rename("Sales").asfreq(freq)
        partial[name] = buckets.loc[~complete, "sum"]
        assert np.isclose(series[name].sum() + partial[name].sum(), sales.Sales.sum())
        period = "W-SAT" if name == "weekly" else "M"
        source_totals = cents.groupby(sales.OrderDate.dt.to_period(period)).sum()
        np.testing.assert_array_equal(np.rint(series[name] * 100).astype("int64"),
                                      source_totals.reindex(series[name].index.to_period(period)))
    assert series["weekly"].index.equals(pd.date_range("2021-07-10", "2025-05-31", freq="W-SAT"))
    assert series["monthly"].index.equals(pd.date_range("2021-07-31", "2025-05-31", freq="ME"))
    order_rows = []
    for label, keys in [("Order number", ["SalesOrderNumber"]),
                        ("Order number + date", ["SalesOrderNumber", "OrderDate"])]:
        orders = sales.groupby(keys).Sales.agg(["sum", "size"])
        order_rows.append({"Order definition": label, "Orders": len(orders), "AOV": orders["sum"].mean(),
                           "Lines per order": orders["size"].mean(), "Total Sales": orders["sum"].sum()})
    by_year = sales.groupby(sales.OrderDate.dt.year).agg(
        Lines=("Sales", "size"), Orders=("SalesOrderNumber", "nunique"), Sales=("Sales", "sum"))
    by_year["Lines per order"] = by_year.Lines / by_year.Orders
    positions = np.flatnonzero(sales.OrderDate.dt.year.eq(2022))
    appended = np.array_equal(positions, np.arange(len(sales)-len(positions), len(sales)))
    reused = sales.groupby("SalesOrderNumber").OrderDate.nunique().gt(1)
    checks = [
        ("Transaction lines", len(sales), "All lines retained; joins preserve values and order"),
        ("Historical Sales", cents.sum()/100, "Source revenue units; currency unspecified"),
        ("Start date", str(daily.index.min().date()), "Observed source coverage"),
        ("End date", str(daily.index.max().date()), "Observed source coverage"),
        ("Exact duplicate lines", sales.duplicated().sum(), "No automatic deletion"),
        ("Amount mismatches", mismatch.sum(), "Sales versus quantity × price; no correction"),
        ("Missing product colors", sheets["Product"].Color.isna().sum(), "Descriptive lookup field retained"),
        ("Complete weeks", len(series["weekly"]), "Sunday–Saturday, W-SAT labels"),
        ("Complete months", len(series["monthly"]), "Complete calendar months only"),
        ("Partial-week Sales", partial["weekly"].sum(), "Opening July 1–3 remains in source reconciliation"),
        ("Zero-record days", daily.eq(0).sum(), "Not confirmed zero-demand days"),
        ("Order numbers spanning dates", reused.sum(), "Neither business-key interpretation is assumed"),
        ("2022 lines", len(positions), "One line per order; source recording-regime concern"),
        ("2022 appended block", bool(appended), "Revenue retained as recorded"),
        ("Source SHA256", source_hash, "Raw workbook never modified"),
    ]
    return dict(sales=sales, enriched=enriched, daily=daily, **series, partial=partial,
                quality=pd.DataFrame(checks, columns=["Check", "Value", "Meaning"]),
                order_keys=pd.DataFrame(order_rows), recording_by_year=by_year,
                source_hash=source_hash)


def numeric_round(frame, digits=2):
    """Round numeric columns without modifying dates or labels."""
    result = frame.copy()
    columns = result.select_dtypes(include="number").columns
    result[columns] = result[columns].round(digits)
    return result


def calculation_namespace(root):
    return dict(PROJECT_ROOT=root, RAW_DATA_PATH=root / "data/raw/Sales Data.xlsx",
                capture_output=capture_output, c_round=numeric_round,
                perf_counter=perf_counter, c_cell_times=[], c_final_times={})


def run_experiments(root):
    """Execute the same Python cells used by Run All, never their saved outputs."""
    notebooks = []
    for path in sorted((root / "notebook").glob("*.ipynb")):
        document = json.loads(path.read_text())
        cells = [c for c in document["cells"]
                 if c.get("cell_type") == "code" and "forecast_step" in c.get("metadata", {})]
        if cells:
            notebooks.append((path, cells))
    if len(notebooks) != 1:
        raise RuntimeError("Expected exactly one notebook with forecasting calculation cells.")
    path, cells = notebooks[0]
    labels = [cell["metadata"]["forecast_step"] for cell in cells]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Every forecast_step label must be a nonempty string.")
    if len(labels) != len(set(labels)):
        raise ValueError("Each forecasting calculation cell must have a unique forecast_step label.")
    namespace = calculation_namespace(root)
    for cell in cells:
        label = cell["metadata"]["forecast_step"]
        print(f"Running {label} …", flush=True)
        with capture_output():
            exec(compile("".join(cell["source"]), f"{path.name}: {label}", "exec"), namespace)
        # Diagnostic figures belong to the notebook; saved presentation plots are generated below.
        plt.close("all")
    return namespace


def broad_weekly_family(spec):
    if spec["kind"] == "ml":
        return "Ridge Regression" if spec["est"] == "Ridge" else spec["est"]
    if spec["kind"] == "exposure":
        return spec["est"] + " Regression"
    return {"naive": "Seasonal Naive", "growth_naive": "Growth-adjusted Seasonal Naive",
            "trend": "Local Trend", "ets": "ETS / Holt-Winters", "arima": "ARIMA",
            "sarima": "SARIMAX", "theta": "Theta", "stl_ets": "STL + ETS",
            "stl_arima": "STL + ARIMA", "dhr": "Dynamic Harmonic Regression",
            "calendar_ridge": "Ridge Regression", "huber": "Huber Regression",
            "prophet": "Prophet"}[spec["kind"]]


def model_inventories(ns):
    weekly = pd.DataFrame(ns["r_registry"])
    weekly["Broad family"] = [broad_weekly_family(s) for s in ns["r_registry"]]
    monthly = pd.DataFrame(ns["r_month_specs"])
    monthly["Broad family"] = monthly.kind.map({
        "naive": "Seasonal Naive", "ets": "ETS / Holt-Winters", "theta": "Theta",
        "arima": "ARIMA / SARIMA", "sarima": "ARIMA / SARIMA",
        "calendar_ridge": "Harmonic / calendar regression", "dhr": "Harmonic / calendar regression",
        "prophet": "Prophet"})
    counts = pd.DataFrame([
        dict(Frequency="Weekly", Labels_or_types=weekly.family.nunique(),
             Broad_families=weekly["Broad family"].nunique(), Configurations=len(weekly)),
        dict(Frequency="Monthly", Labels_or_types=monthly.kind.nunique(),
             Broad_families=monthly["Broad family"].nunique(), Configurations=len(monthly))])
    return weekly, monthly, counts


def retrospective_detail(dates, actual, predicted):
    detail = pd.DataFrame({"Date": dates, "Actual Sales": np.asarray(actual),
                           "Predicted Sales": np.asarray(predicted)})
    detail["Absolute Error"] = np.abs(detail["Predicted Sales"] - detail["Actual Sales"])
    detail["APE"] = 100 * detail["Absolute Error"] / detail["Actual Sales"].abs().replace(0, np.nan)
    detail["WAPE contribution"] = 100 * detail["Absolute Error"] / detail["Actual Sales"].abs().sum()
    return detail


def study_results(ns):
    """Small presentation views of executed objects; no new tuning or metric definitions."""
    weekly = dict(frequency="weekly", selected_family=ns["r_champion"],
                  specification=ns["r_final_bank"][ns["r_champion"]]["spec"],
                  comparison=ns["r_primary"].copy(), fold_metrics=ns["r_dev_metrics"].copy(),
                  retrospective_metrics=ns["r_final_metrics"].loc[[ns["r_champion"]]].reset_index(),
                  retrospective_predictions=retrospective_detail(ns["r_final_dates"],
                      ns["r_final_actual"], ns["r_locked_forecasts"][ns["r_champion"]]))
    month_comparison = pd.DataFrame([{m: ns["r_strict_mean"](ns["r_month_metrics"][m]) for m in METRICS}],
                                    index=pd.Index(["Adaptive monthly selection"], name="Model"))
    month_comparison["Fold-to-fold SD"] = ns["r_month_metrics"].WAPE.std()
    month_comparison["MAPE SD"] = ns["r_month_metrics"].MAPE.std()
    monthly = dict(frequency="monthly", selected_family="Adaptive monthly selection",
                   specification=ns["r_final_month_choice"]["spec"], comparison=month_comparison,
                   fold_metrics=ns["r_month_metrics"].copy(),
                   retrospective_metrics=ns["r_final_month_metrics"].copy(),
                   retrospective_predictions=retrospective_detail(ns["r_final_month_dates"],
                       ns["r_monthly"].loc[ns["r_final_month_dates"]], ns["r_final_month_pred"]))
    return [weekly, monthly]


def timing_summary(ns, elapsed, studies):
    """Wall times around calculation stages; final selection is timed separately by task."""
    rows = list(ns["c_cell_times"])
    totals = {scope: sum(row["Seconds"] for row in rows if row["Scope"] == scope)
              + ns["c_final_times"].get(scope, 0.) for scope in ["Weekly", "Monthly", "Sensitivity"]}
    for study in studies:
        scope = study["frequency"].title()
        totals[scope] += study["production_seconds"]
        study["total_seconds"] = totals[scope]
    return pd.DataFrame([
        {"Scope": name, "Stage": "Experiment total", "Seconds": seconds} for name, seconds in totals.items()
    ] + [{"Scope": "Both", "Stage": "Shared preparation / presentation / QA",
           "Seconds": max(0., elapsed-sum(totals.values()))},
          {"Scope": "Both", "Stage": "Complete execution", "Seconds": elapsed}])


def load_selected_model(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Saved model not found: {path}. Original sales models contain historical "
            "observations and are excluded from Git; see models/README.md."
        )
    if Path(path).suffix == ".json":
        text = Path(path).read_text()
        metadata = json.loads(text)["sales_metadata"]
        return dict(model=model_from_json(text), specification=metadata["specification"],
                    scale=metadata["scale"], frequency=metadata["frequency"],
                    training_end=pd.Timestamp(metadata["training_end"]))
    return joblib.load(path)


def save_data_tables(data, root):
    for name in ["daily", "weekly", "monthly"]:
        data[name].rename("Sales").to_csv(root / "data/processed" / f"{name}_sales.csv", date_format="%Y-%m-%d")
    data["enriched"].to_csv(root / "data/processed/enriched_sales.csv", index=False)
    data["quality"].to_csv(root / "tables/data_quality_summary.csv", index=False)
    data["order_keys"].to_csv(root / "tables/order_key_comparison.csv", index=False)
    data["recording_by_year"].to_csv(root / "tables/recording_by_year.csv")


def save_study_tables(study, root):
    prefix = study["frequency"]
    for key, name in [("comparison", "model_comparison"), ("fold_metrics", "fold_metrics"),
                      ("retrospective_metrics", "retrospective_results"),
                      ("retrospective_predictions", "retrospective_predictions"), ("future", "future_forecast")]:
        study[key].to_csv(root / "tables" / f"{prefix}_{name}.csv", index=(key == "comparison"))


def finish_figure(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / "visuals" / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def revenue_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1e6:,.1f}m"))
    ax.set_ylabel("Sales (source revenue units)")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_eda(data, root):
    figures = []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["daily"].index, data["daily"], color="#a8c4db", linewidth=.6, label="Daily recorded Sales")
    ax.plot(data["daily"].index, data["daily"].rolling(28).mean(), color="#24578a", label="28-day descriptive average")
    ax.set_title("Recorded sales over time")
    ax.set_xlabel("Date")
    revenue_axis(ax)
    ax.legend()
    figures.append(finish_figure(fig, root, "sales_over_time.png"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for frequency, ax in zip(["weekly", "monthly"], axes):
        series = data[frequency]
        ax.plot(series.index, series, color="#24578a", linewidth=1)
        ax.set_title("Complete Sunday–Saturday weeks" if frequency == "weekly" else "Complete calendar months")
        ax.set_xlabel("Week ending Saturday" if frequency == "weekly" else "Month end")
        revenue_axis(ax)
    figures.append(finish_figure(fig, root, "weekly_monthly_sales.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for column, ax in zip(["Category", "Region"], axes):
        amounts = data["enriched"].groupby(column).Sales.sum().sort_values()
        ax.barh(amounts.index, amounts, color="#128579")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1e6:.1f}m"))
        ax.set(title=f"Recorded Sales by {column.lower()}", xlabel="Sales (source revenue units)")
    figures.append(finish_figure(fig, root, "category_region_sales.png"))
    return figures


def plot_study(study, series, root):
    frequency = study["frequency"]
    figures = []
    comparison = study["comparison"].drop(index="Adaptive selection", errors="ignore").sort_values("WAPE", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(4.8, .28 * len(comparison))))
    ax.barh(comparison.index, comparison.WAPE, color="#128579")
    ax.set(title=f"{frequency.title()}: four development folds", xlabel="Mean fold WAPE (%)")
    figures.append(finish_figure(fig, root, f"{frequency}_model_comparison.png"))
    path = study["retrospective_predictions"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(path.Date, path["Actual Sales"], marker="o", label="Actual")
    ax.plot(path.Date, path["Predicted Sales"], marker="o", label=(study["selected_family"] if frequency == "weekly" else "Multiplicative ETS"))
    ax.set_title(f"{frequency.title()} locked retrospective holdout — " +
                 ("13 complete weeks" if frequency == "weekly" else "three complete calendar months"))
    if frequency == "monthly":
        ax.set_xticks(path.Date, path.Date.dt.strftime("%b %Y"))
        ax.set_xlabel("Complete calendar month")
    else:
        ax.set_xticks(path.Date.iloc[::2], path.Date.iloc[::2].dt.strftime("%d %b"))
        ax.set_xlabel("Week ending Saturday (2025)")
    revenue_axis(ax)
    ax.legend()
    figures.append(finish_figure(fig, root, f"{frequency}_retrospective_forecast.png"))
    fig, ax = plt.subplots(figsize=(10, 4))
    history = series.iloc[-52:] if frequency == "weekly" else series.iloc[-24:]
    ax.plot(history.index, history, label="Recorded Sales", color="#24578a")
    future = study["future"]
    ax.plot(future.Date, future["Forecast Sales"], marker="o", label="Forecast", color="#c47a22")
    ax.axvline(series.index[-1], linestyle="--", color="gray", linewidth=.8)
    ax.set_title(f"{frequency.title()} forecast after the workbook's May 2025 cutoff")
    ax.set_xlabel("Week ending Saturday" if frequency == "weekly" else "Month end")
    revenue_axis(ax)
    ax.legend()
    figures.append(finish_figure(fig, root, f"{frequency}_future_forecast.png"))
    return figures


def persist_selected(ns, root):
    """Refit the selected specifications; save and reload two models.

    A temporary Python profiler captures the fitted result already constructed by
    the forecasting function. Its transforms and fitting options stay intact.
    This function never selects or tunes a configuration.
    """
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    weekly_spec = dict(ns["r_final_bank"][ns["r_champion"]]["spec"])
    monthly_spec = dict(ns["r_final_month_choice"]["spec"])
    selections = (
        ("weekly", ns["r_y"], pd.DatetimeIndex(ns["r_future_dates"]), weekly_spec, "prophet"),
        ("monthly", ns["r_monthly"], pd.date_range(ns["r_monthly"].index[-1], periods=4, freq="ME")[1:],
         monthly_spec, "ets"),
    )
    # An unexpected result requires an explicit serializer, not a forced selection.
    for frequency, _, _, spec, expected_kind in selections:
        if spec["kind"] != expected_kind or spec.get("tx", "raw") != "raw":
            raise ValueError(f"The selected {frequency} specification needs a reviewed serializer: {spec}")

    outputs = {}
    for frequency, train, dates, spec, kind in selections:
        start = perf_counter()
        captured = {}
        base_code = ns["r_predict_base"].__code__
        previous_profiler = sys.getprofile()

        def capture_fit(frame, event, arg):
            if event == "return" and frame.f_code is base_code and arg is not None:
                captured["model"] = frame.f_locals.get("fit")

        try:
            sys.setprofile(capture_fit)
            with warnings.catch_warnings(record=True) as fit_warnings, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                prediction, transform = ns["r_predict"](train, dates, spec)
        finally:
            sys.setprofile(previous_profiler)
        refit_seconds = perf_counter() - start
        model = captured.get("model")
        if model is None:
            raise RuntimeError(f"The original forecaster did not expose its fitted {frequency} model.")
        if frequency == "weekly":
            np.testing.assert_allclose(prediction, ns["r_future_prediction"], rtol=1e-10, atol=1e-7)
        metadata = dict(specification=spec, scale=transform["transform_scale"], frequency=frequency,
                        training_end=str(train.index[-1]), training_observations=len(train),
                        source_sha256=ns["r_source_hash"])
        path = root / "models" / f"{frequency}_{kind}{'.json' if kind == 'prophet' else '.joblib'}"
        save_start = perf_counter()
        if kind == "prophet":
            payload = json.loads(model_to_json(model))
            # All native Prophet fields remain intact; this records revenue scaling.
            payload["sales_metadata"] = metadata
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            joblib.dump(dict(model=model, **metadata), path, compress=3)
        serialization_seconds = perf_counter() - save_start

        reload_start = perf_counter()
        with threadpool_limits(limits=1):
            if kind == "prophet":
                saved_text = path.read_text(encoding="utf-8")
                restored = model_from_json(saved_text)
                restored_metadata = json.loads(saved_text)["sales_metadata"]
                restored_prediction = restored.predict(pd.DataFrame({"ds": dates})).yhat.to_numpy()
            else:
                saved = joblib.load(path)
                restored, restored_metadata = saved["model"], saved
                restored_prediction = np.asarray(restored.forecast(len(dates)))
            restored_prediction = np.maximum(restored_prediction * restored_metadata["scale"], 0.0)
        np.testing.assert_allclose(restored_prediction, prediction, rtol=1e-10, atol=1e-7)
        reload_seconds = perf_counter() - reload_start
        outputs[frequency] = dict(
            frequency=frequency, specification=spec,
            future=pd.DataFrame({"Date": dates, "Forecast Sales": prediction}),
            model_path=path.relative_to(root).as_posix(), refit_seconds=refit_seconds,
            serialization_seconds=serialization_seconds, reload_seconds=reload_seconds,
            production_seconds=perf_counter() - start,
            reload_max_difference=float(np.max(np.abs(restored_prediction - prediction))),
            warnings=[f"{type(w.message).__name__}: {w.message}" for w in fit_warnings],
        )
        print(f"Reload verified: {outputs[frequency]['model_path']}")
    return outputs


def c_results_digest(state):
    keys = [
        "development_predictions",
        "final_metrics",
        "weekly_contributions",
        "aggregation_predictions",
        "monthly_predictions",
        "production_forecast",
    ]
    payload = {key: state[key] for key in keys}
    scope_names = {
        "Final retrospective test": "Final locked retrospective weekly holdout",
        "Final aggregation test": "Final controlled aggregation holdout",
        "Final monthly test": "Final complete-month holdout",
    }
    metric_names = [
        "MAPE",
        "WAPE",
        "sMAPE",
        "MAE",
        "RMSE",
        "MASE",
        "Bias",
        "Bias_pct",
        "Fold-to-fold SD",
        "MAPE SD",
    ]
    rows = []
    for row in state["master_results"]:
        rows.append(
            dict(
                Model=row["Model"],
                Scope=scope_names.get(row["Scope"], row["Scope"]),
                **{m: row.get(m) for m in metric_names},
            )
        )
    payload["master_metrics"] = sorted(
        rows, key=lambda row: (row["Scope"], row["Model"])
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def predict_model(fitted, dates):
    dates = pd.DatetimeIndex(dates)
    frequency = fitted["frequency"]
    expected = pd.date_range(pd.Timestamp(fitted["training_end"]), periods=len(dates)+1,
                             freq=SETTINGS[frequency][0])[1:]
    if not len(dates) or not dates.equals(expected):
        raise ValueError("Forecast dates must immediately follow the saved training cutoff.")
    with threadpool_limits(limits=1):
        if fitted["specification"]["kind"] == "prophet":
            prediction = fitted["model"].predict(pd.DataFrame({"ds": dates})).yhat.to_numpy()
        elif fitted["specification"]["kind"] == "ets":
            prediction = np.asarray(fitted["model"].forecast(len(dates)))
        else:
            raise ValueError("Saved inference supports Prophet and ETS artifacts only")
    prediction = prediction * fitted["scale"]
    if prediction.shape != (len(dates),) or not np.isfinite(prediction).all():
        raise ValueError("The saved model returned an invalid forecast; check its matching environment and metadata")
    return np.maximum(prediction, 0.)


def check_results(ns, data, root):
    """Check that data, predictions and scores match the verified run."""
    assert len(data["sales"]) == len(ns["r_tx"]) == 60097
    for name, key, count in [("weekly", "r_y", 204), ("monthly", "r_monthly", 47)]:
        assert len(ns[key]) == count
        pd.testing.assert_series_equal(data[name], ns[key], check_names=False, check_freq=False)
    assert data["source_hash"] == ns["r_source_hash"] == hashlib.sha256(
        (root / "data/raw/Sales Data.xlsx").read_bytes()).hexdigest()
    for key, expected in [
        ("r_registry", "7d6e871cd3972931601448d69585d91ad9dfc45e57087acf9373eb6976391c12"),
        ("r_month_specs", "5d448181a909cf2ddaefccb82dafceefc546661e342f2456ed6ad8ef86d8bf25")]:
        assert hashlib.sha256(json.dumps(ns[key], sort_keys=True, default=ns["r_json_default"]).encode()).hexdigest() == expected
    assert ns["r_freeze_hash"] == "f47ad2dad10821f8754ee5438500826bd4676eeab6a317cf57354444dc47d797"
    assert c_results_digest(ns["r_final_state"]) == "000795f62e2acac9616be9f4d632c7a2527eaefbf4bd84fb38478a36c5154ffd", "Historical forecast/metric digest changed. Investigate before accepting results."
    assert ns["r_champion"] == ns["r_frozen"]["champion"]
    assert ns["r_plan"].validation_start.dt.strftime("%Y-%m-%d").tolist() == [
        "2024-03-09", "2024-06-08", "2024-09-07", "2024-12-07"]
    assert ns["r_plan"].validation_end.dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-01", "2024-08-31", "2024-11-30", "2025-03-01"]
    assert ns["r_final_dates"].equals(pd.date_range("2025-03-08", "2025-05-31", freq="W-SAT"))
    assert ns["r_month_origins"] == list(pd.to_datetime(["2024-02-29", "2024-05-31", "2024-08-31", "2024-11-30"]))
    assert ns["r_final_month_dates"].equals(pd.date_range("2025-03-31", "2025-05-31", freq="ME"))
    for (_, fold), group in ns["r_dev_predictions"].groupby(["Model", "fold"]):
        origin = int(group.origin.iloc[0]); train = ns["r_y"].iloc[:origin]
        assert len(group) == 13 and train.index[-1] < group.week_end.min()
        assert group.week_end.max() <= ns["r_development"].index[-1]
        np.testing.assert_array_equal(group.actual, ns["r_y"].loc[group.week_end])
        row = ns["r_dev_metrics"].loc[ns["r_dev_metrics"].Model.eq(group.Model.iloc[0]) & ns["r_dev_metrics"].Fold.eq(fold)].iloc[0]
        result = ns["r_metrics"](group.actual, group.predicted, train, period=52)
        np.testing.assert_allclose([row[m] for m in METRICS], [result[m] for m in METRICS], rtol=1e-12, equal_nan=True)
    for fold, group in pd.DataFrame(ns["r_month_predictions"]).groupby("Fold"):
        origin = ns["r_month_origins"][int(fold)-1]; train = ns["r_monthly"].loc[:origin]
        assert pd.DatetimeIndex(group.month_end).equals(pd.date_range(origin + pd.offsets.MonthEnd(1), periods=3, freq="ME"))
        assert len(group) == 3 and train.index[-1] < group.month_end.min()
        assert group.month_end.max() <= ns["r_month_final_origin"]
        row = ns["r_month_metrics"].loc[ns["r_month_metrics"].Fold.eq(fold)].iloc[0]
        result = ns["r_metrics"](group.actual, group.predicted, train, period=12)
        np.testing.assert_allclose([row[m] for m in METRICS], [result[m] for m in METRICS], rtol=1e-12, equal_nan=True)
    assert np.isnan(ns["r_metrics"]([0, 1], [1, 1], ns["r_y"])["MAPE"])
    assert ns["r_metrics"]([1, 9], [2, 9], ns["r_y"])["WAPE"] == 10
    assert "PASS: shifted rolling features" in ns["c_audit_82"].stdout
    assert pd.to_datetime([x["origin"] for x in ns["r_inner_scores"]]).max() <= ns["r_development"].index[-1]
    assert pd.to_datetime([x["origin"] for x in ns["r_month_inner"]]).max() <= ns["r_month_final_origin"]
    assert all(x["source"] == "two inner folds only" for x in ns["r_ensemble_log"])
    assert {p.name for p in (root / "models").iterdir() if p.is_file() and p.suffix in {".joblib", ".json"} and p.name != "selected_recipes.json"} == {"weekly_prophet.json", "monthly_ets.joblib"}


def executive_comparison(studies):
    rows = []
    for study in studies:
        for scope, result in [
            ("Development cross-validation", study["comparison"].loc[study["selected_family"]]),
            ("Locked retrospective holdout", study["retrospective_metrics"].iloc[0])]:
            rows.append(dict(Frequency=study["frequency"], Scope=scope,
                             **{metric: result[metric] for metric in METRICS}))
    return pd.DataFrame(rows)


def rebuild_study(clean_generated=False):
    started = perf_counter()
    plt.switch_backend("Agg")
    root = find_project_root()
    prepare_directories(root, clean=clean_generated)
    data = load_data(root)
    save_data_tables(data, root)
    plot_eda(data, root)
    ns = run_experiments(root)
    studies = study_results(ns)
    production = persist_selected(ns, root)
    for study in studies:
        study.update(production[study["frequency"]])
        save_study_tables(study, root)
        plot_study(study, data[study["frequency"]], root)
    check_results(ns, data, root)
    elapsed = perf_counter()-started
    timing = timing_summary(ns, elapsed, studies)
    timing.to_csv(root / "tables/timing_summary.csv", index=False)
    print("\nExecuted candidate inventory:")
    print(model_inventories(ns)[2].to_string(index=False))
    print("\nDevelopment and locked retrospective results:")
    print(executive_comparison(studies).round(6).to_string(index=False))
    for study in studies:
        print(f"Selected {study['frequency']} specification: {study['specification']}")
    print("\nRuntime (seconds):")
    print(timing.round(3).to_string(index=False))
    print(f"Fit attempts: {len(ns['r_fit_audit'])}; failed attempts: {ns['r_fit_audit'].error.ne('').sum()}")
    print("PASS: source, chronology, metric, registry, historical forecast hashes and model reload checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=SETTINGS, default="weekly")
    parser.add_argument("--periods", type=int, help="Forecast periods; defaults to 13 weeks or 3 months")
    parser.add_argument("--model", type=Path, help="Saved Prophet JSON or ETS joblib package")
    parser.add_argument("--output", type=Path, help="New CSV path; otherwise print CSV to stdout")
    parser.add_argument("--rebuild-study", action="store_true", help="Rerun original training/search and replace study exports")
    parser.add_argument("--clean-generated", action="store_true", help="Clear generated assets before an explicit study rebuild")
    args = parser.parse_args()
    if args.clean_generated and not args.rebuild_study:
        parser.error("--clean-generated requires --rebuild-study")
    if args.rebuild_study:
        if args.model or args.periods is not None or args.output:
            parser.error("Inference options cannot be combined with --rebuild-study")
        rebuild_study(args.clean_generated)
        return
    periods = args.periods if args.periods is not None else SETTINGS[args.frequency][1]
    if periods < 1:
        parser.error("--periods must be a positive integer")
    filename = "weekly_prophet.json" if args.frequency == "weekly" else "monthly_ets.joblib"
    path = args.model or Path(__file__).resolve().parent / "models" / filename
    fitted = load_selected_model(path)
    if fitted["frequency"] != args.frequency:
        parser.error("--frequency does not match the selected model")
    dates = pd.date_range(fitted["training_end"], periods=periods + 1, freq=SETTINGS[args.frequency][0])[1:]
    result = pd.DataFrame({"Date": dates, "Predicted Sales": predict_model(fitted, dates)})
    if args.output:
        if args.output.exists():
            parser.error("Choose a new output path; existing files are not overwritten")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Forecast saved to {args.output}")
    else:
        print(result.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
