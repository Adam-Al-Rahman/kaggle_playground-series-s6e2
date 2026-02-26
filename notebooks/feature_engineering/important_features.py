import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd

    import polars.selectors as cs

    return mo, pl


@app.cell
def _(mo):
    from exploratory_data_analysis.feature_engineered_inspection.scaled_feature_eda import (
        SCALED_FEATURE_EDA_INSIGHTS,
    )

    mo.callout(
        mo.vstack(
            [
                mo.md("## Extracted Insights"),
                mo.tree(SCALED_FEATURE_EDA_INSIGHTS()),
            ]
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Important Features
    """)
    return


@app.cell
def _(pl):
    from pathlib import Path

    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    PROJECT_ROOT = "playground-series-s6e2"
    CWD = Path.cwd()
    PROJECT_PATH = Path(*CWD.parts[: CWD.parts.index(PROJECT_ROOT) + 1])

    bootstrap_project(PROJECT_PATH)

    with KedroSession.create(PROJECT_PATH, False) as session:
        context = session.load_context()
        catalog = context.catalog

        ds_train: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_train_scaled"
        )
        ds_test: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_test_scaled"
        )
    return (ds_train,)


@app.cell
def _(ds_train: "pl.DataFrame", pl):
    pl.from_pandas(ds_train).select(
        ["HR_Deficit", "Thallium", "Chest pain type", "Heart Disease"]
    )
    return


@app.cell
def _(ds_train: "pl.DataFrame", pl):
    pl.from_pandas(ds_train).select(
        ["HR_Deficit", "Thallium", "Chest pain type", "Heart Disease"]
    ).write_csv("data/processed/02_feature/important_features.csv")
    return


if __name__ == "__main__":
    app.run()
