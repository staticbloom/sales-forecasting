# Data: sales forecasting

The source dataset is not included in this repository. This directory is retained for documentation, directory placeholders, and explicitly identified synthetic examples or reproducibility metadata. Raw data and processed source records remain local.

Original input: `data/raw/Sales Data.xlsx`, an Excel workbook with 60,097 sales lines and four sheets. The checked schema in `predict.py:load_data` requires:

- `Sales`: SalesOrderNumber, OrderDate, ProductKey, EmployeeKey, SalesTerritoryKey, Quantity, Unit Price, Sales.
- `Product`: ProductKey, Product, Color, Subcategory, Category.
- `Region`: SalesTerritoryKey, Region, Country, Group.
- `Salesperson`: EmployeeKey, EmployeeID, Salesperson, Title, UPN.

No verified public acquisition URL or license accompanies this workbook. Obtain the exact workbook and permission from the supplier. Names, employee identifiers and UPN/email fields occur in the local enriched transactions; do not publish them.

`data/processed/` contains generated `enriched_sales.csv`, `daily_sales.csv`, `weekly_sales.csv` and `monthly_sales.csv`. All are excluded. The workbook's observed period is 1 July 2021–31 May 2025. Revenue currency is unspecified. Original analysis fills unrecorded days with zero recorded Sales; this does not prove zero demand. Complete Sunday–Saturday weeks and calendar months exclude incomplete boundary periods.

For new training, supply a comma-delimited CSV with exactly `Date,Sales` (ISO calendar dates and finite nonnegative amounts in consistent units). Missing days fail unless `--missing-days zero` is explicitly chosen. Weekly needs at least 117 complete weeks; monthly needs 27 months. See `train.py --help`.

The original model files also contain 204 weekly/47 monthly observed values and remain local. Public recipes permit new fixed-recipe training without those artifacts. Original-study recreation requires the exact workbook and `python predict.py --rebuild-study`; it is an expensive search that replaces generated outputs. No fresh-clone end-to-end study reproduction is claimed.
