import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import altair as alt
    import numpy as np

    import pandas as pd
    import polars as pl
    import polars.selectors as cs
    import altair as alt

    import scipy.stats as stats
    import category_encoders as ce

    import nannyml as nml

    nml.disable_usage_logging()


@app.cell(hide_code=True)
def _():
    from exploratory_data_analysis.raw_inspection.data_integrity import (
        DATA_INTEGRITY_INSIGHTS,
    )
    from exploratory_data_analysis.raw_inspection.feature_profiling import (
        FEATURE_PROFILING_INSIGHTS,
    )


    mo.callout(
        mo.vstack(
            [
                mo.md("## Extracted Insights"),
                mo.tree(FEATURE_PROFILING_INSIGHTS()),
                mo.tree(DATA_INTEGRITY_INSIGHTS()),
            ]
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Preprocessing
    """)
    return


@app.function
def DATA_PREPROCESSING_INSIGHTS():
    INSIGHTS = {
        "data preprocessing": {
            "nominal encoding": {
                "features": [
                    "Chest pain type",
                    "EKG results",
                    "Thallium",
                ],
                "task": "weight of evidence encoding",
            },
            "oridnal encoding": {
                "features": ["Slope of ST", "Number of vessels fluro"],
                "task": "keep it as integer / maybe thermometer encoding",
            },
            "binary encoding": {
                "features": ["Sex", "FBS over 120", "Exercise angina"],
                "task": "keep it as 0/1",
            },
        },
    }

    return INSIGHTS


@app.cell
def _():
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    from pathlib import Path

    PROJECT_ROOT = "playground-series-s6e2"
    CWD = Path.cwd()
    PROJECT_PATH = Path(*CWD.parts[: CWD.parts.index(PROJECT_ROOT) + 1])

    bootstrap_project(PROJECT_PATH)

    with KedroSession.create(PROJECT_PATH, False) as session:
        context = session.load_context()
        catalog = context.catalog

        ds_train: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease")
        ds_test: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease_test")
    return ds_test, ds_train


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test
    return


@app.cell
def _():
    TARGET = "Heart Disease"
    CONT_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

    NOMINAL_FEATURES = ["Chest pain type", "EKG results", "Thallium"]
    ORDINAL_FEATURES = ["Slope of ST", "Number of vessels fluro"]
    BINARY_FEATURES = ["Sex", "FBS over 120", "Exercise angina"]

    CATEGORICAL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

    FEATURES = CONT_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    return NOMINAL_FEATURES, TARGET


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nominal Encoding
    """)
    return


@app.function
def verify_log_odds_linearity(
    ds: pl.DataFrame, feature_col: str, target_col: str, n_bins: int = 10
):
    """
    Groups a continuous feature into quantiles, calculates the empirical log-odds of the target,
    and returns an interactive visualization to check the linear assumption.
    """

    # 1. Bin the continuous feature into quantiles to ensure equal statistical weight per bin
    binned_df = ds.with_columns(
        pl.col(feature_col)
        .qcut(n_bins, allow_duplicates=True)
        .alias("quantile_bin")
    )

    # 2. Calculate the mean of the feature and the probability of the target in each bin
    agg_df = binned_df.group_by("quantile_bin").agg(
        [
            pl.col(feature_col).mean().alias("mean_feature_value"),
            pl.col(target_col).mean().alias("probability"),
            pl.col(target_col).count().alias("sample_count"),
        ]
    )

    # 3. Calculate log-odds: ln(p / (1-p))
    # We clip the probability with a small epsilon to avoid math domain errors (ln(0) or division by 0)
    epsilon = 1e-5

    analyzed_df = (
        agg_df.with_columns(
            pl.col("probability")
            .clip(lower_bound=epsilon, upper_bound=1 - epsilon)
            .alias("adj_prob")
        )
        .with_columns(
            (pl.col("adj_prob") / (1 - pl.col("adj_prob")))
            .log()
            .alias("empirical_log_odds")
        )
        .sort("mean_feature_value")
        .drop("adj_prob")
    )

    # 4. Prepare data for Altair visualization
    plot_data = analyzed_df.to_pandas()

    chart = (
        alt.Chart(plot_data)
        .mark_line(point=alt.OverlayMarkDef(filled=False, size=50))
        .encode(
            x=alt.X(
                "mean_feature_value:Q", title=f"Mean of {feature_col} (Binned)"
            ),
            y=alt.Y("empirical_log_odds:Q", title="Empirical Log-Odds"),
            tooltip=[
                "mean_feature_value",
                "probability",
                "empirical_log_odds",
                "sample_count",
            ],
        )
        .properties(
            title=f"Linearity Assumption Check: Log-Odds vs {feature_col}",
            width=650,
            height=400,
        )
        .interactive()
    )

    # Render purely through marimo outputs
    return mo.vstack(
        [
            mo.md(f"**Evaluating Linear Assumption for `{feature_col}`**"),
            mo.md(
                "> **Interpretation:** If the line below is relatively straight, the continuous feature can be used directly in Logistic Regression. If it curves significantly, consider polynomial expansion or WoE encoding."
            ),
            mo.ui.altair_chart(chart),
            mo.accordion({"View Raw Bin Computations": mo.ui.table(plot_data)}),
        ]
    )


@app.cell
def _(NOMINAL_FEATURES):
    NOMINAL_FEATURES
    return


@app.cell(hide_code=True)
def _(NOMINAL_FEATURES):
    mo.md(rf"""
    ### Feature: {NOMINAL_FEATURES[0]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    _chart = (
        alt.Chart(ds_train[:1000])
        .mark_bar()
        .encode(
            x=alt.X(field="Chest pain type", type="quantitative", sort="ascending"),
            y=alt.Y(aggregate="count", type="quantitative"),
            tooltip=[
                alt.Tooltip(field="Chest pain type", format=",.0f"),
                alt.Tooltip(aggregate="count"),
            ],
        )
        .properties(height=290, width="container", config={"axis": {"grid": False}})
    )
    _chart
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.head()
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train_label_transformed = ds_train.with_columns(
        pl.col("Heart Disease").replace({"Absence": 0, "Presence": 1}).cast(pl.Int8)
    )
    return (ds_train_label_transformed,)


@app.cell
def _(ds_train_label_transformed):
    ds_train_label_transformed.head()
    return


@app.cell
def _(NOMINAL_FEATURES, ds_train_label_transformed):
    [
        verify_log_odds_linearity(
            ds_train_label_transformed, feature, "Heart Disease"
        )
        for feature in NOMINAL_FEATURES
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preprocessing Pipeline
    """)
    return


@app.cell
def _(
    NOMINAL_FEATURES,
    TARGET,
    ds_test: pl.DataFrame,
    ds_train_label_transformed,
):
    X_train = ds_train_label_transformed.drop(TARGET).to_pandas()
    y_train = ds_train_label_transformed[TARGET].to_pandas()
    X_test = ds_test.to_pandas()

    # WOE Encoding
    woe_encoder = ce.WOEEncoder(cols=[NOMINAL_FEATURES[0]])
    ds_train_processed = woe_encoder.fit_transform(X_train, y_train)
    ds_test_processed = woe_encoder.transform(X_test)

    ds_train_processed[TARGET] = y_train.values
    return ds_test_processed, ds_train_processed


@app.cell
def _(ds_train_processed):
    ds_train_processed
    return


@app.cell
def _(ds_test_processed):
    ds_test_processed
    return


if __name__ == "__main__":
    app.run()
