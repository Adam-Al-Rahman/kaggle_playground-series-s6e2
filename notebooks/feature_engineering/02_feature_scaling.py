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

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


@app.cell
def _():
    from exploratory_data_analysis.raw_inspection.univariate_analysis import (
        UNIVARIATE_ANALYSIS_INSIGHTS,
    )

    mo.callout(
        mo.vstack(
            [
                mo.md("## Extracted Insights"),
                mo.tree(UNIVARIATE_ANALYSIS_INSIGHTS()),
            ]
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Scaling
    """)
    return


@app.cell
def _(INSIGHS):
    def FEATURE_SCALING_INSIGHTS():
        INSIGHTS = {
            "caution": "scaled features not to be used in cross validation, instead go with feature engineered dataset"
        }
        return INSIGHS

    return


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

        ds_train: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_train"
        )
        ds_test: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_test"
        )
    return ds_test, ds_train


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.columns.to_list()
    return


@app.cell
def _(ds_test: pl.DataFrame, ds_train: pl.DataFrame):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), ["BP", "Cholesterol"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_scaled = preprocessor.fit_transform(
        ds_train.drop(["id", "Heart Disease"], axis=1)
    )
    X_test_scaled = preprocessor.transform(ds_test)

    feature_names = preprocessor.get_feature_names_out()

    ds_train_scaled = pl.DataFrame(X_train_scaled, schema=list(feature_names))
    ds_test_scaled = pl.DataFrame(X_test_scaled, schema=list(feature_names))
    return (ds_train_scaled,)


@app.cell
def _(ds_train_scaled):
    ds_train_scaled
    return


@app.cell
def _(ds_train: pl.DataFrame, ds_train_scaled):
    ds_train_scaled.with_columns(
        pl.Series("Heart Disease", ds_train["Heart Disease"])
    ).write_csv("data/processed/02_feature/feature_scaled_train.csv")

    ds_train_scaled.write_csv("data/processed/02_feature/feature_scaled_test.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
