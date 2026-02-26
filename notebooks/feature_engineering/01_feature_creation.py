import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import altair as alt
    import numpy as np

    import woodwork.logical_types as ww
    import featuretools as ft

    import pandas as pd
    import polars as pl
    import polars.selectors as cs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Engineering
    """)
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

        ds_train: pl.DataFrame = catalog.load("processed.ds_heart_disease_train")
        ds_test: pl.DataFrame = catalog.load("processed.ds_heart_disease_test")
    return (ds_train,)


@app.cell
def _():
    TARGET = "Heart Disease"
    CONT_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

    NOMINAL_FEATURES = ["Chest pain type", "EKG results", "Thallium"]
    ORDINAL_FEATURES = ["Slope of ST", "Number of vessels fluro"]
    BINARY_FEATURES = ["Sex", "FBS over 120", "Exercise angina"]

    CATEGORICAL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

    FEATURES = CONT_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.sample(10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Deep Feature Synthesis
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    # Initialize the EntitySet
    ds_train_es = ft.EntitySet(id="heart_disease_entityset")

    # Map your feature lists to strict Woodwork Logical Types to optimize DFS
    # This prevents meaningless math (like multiplying "Sex" by "Exercise angina")
    logical_types_map = {
        "Sex": ww.Boolean,
        "FBS over 120": ww.Boolean,
        "Exercise angina": ww.Boolean,
        "Chest pain type": ww.Categorical,
        "EKG results": ww.Categorical,
        "Thallium": ww.Categorical,
        # Ordinal requires an order; inferring natural order for these clinical metrics
        "Slope of ST": ww.Ordinal(order=[1, 2, 3]),
        "Number of vessels fluro": ww.Ordinal(order=[0, 1, 2, 3]),
        "Heart Disease": ww.Categorical,
    }

    # Add the primary dataframe to the EntitySet
    ds_train_es = ds_train_es.add_dataframe(
        dataframe_name="ds_train_patients",
        dataframe=ds_train,
        index="id",
        logical_types=logical_types_map,
    )

    # Normalize entity to unlock aggregation primitives
    ds_train_es.normalize_dataframe(
        base_dataframe_name="ds_train_patients",
        new_dataframe_name="chest_pain_cohorts",
        index="Chest pain type",
    )
    return (ds_train_es,)


@app.cell
def _(ds_train_es):
    # Run Deep Feature Synthesis
    feature_matrix, feature_defs = ft.dfs(
        entityset=ds_train_es,
        target_dataframe_name="ds_train_patients",
        ignore_columns={"ds_train_patients": ["Heart Disease"]},
        trans_primitives=["add_numeric", "multiply_numeric", "absolute"],
        agg_primitives=["mean", "max", "min", "count"],
        max_depth=2,
        verbose=False,
    )
    return (feature_matrix,)


@app.cell
def _(feature_matrix):
    feature_matrix.iloc[:1000, :].reset_index()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Domain Feature Synthesis

    ### Maximum Target Heart Rate Deficit (HR_Deficit)

    **The Clinical Reality**: A healthy human heart has a maximum beats-per-minute (bpm) capacity that naturally declines as we age. A widely accepted clinical baseline for this is 220−Age. During a cardiovascular stress test, doctors want to see how close a patient can safely get to this maximum limit.

    **The Feature Logic**: We calculate the theoretical maximum (220−Age) and subtract the patient's actual Max HR achieved during the test.

    `Calculation: (220 - Age) - Max_HR`

    ### Ischemia Risk Interaction (High_Risk_Ischemia)

    **The Clinical Reality**: "Ischemia" means the heart muscle is being starved of oxygen-rich blood. Two massive warning signs of this during exercise are:

    **Angina**: Physical chest pain (Exercise angina).

    **ST Depression**: A specific abnormal drop in the electrical wave on an EKG.

    When a patient has both simultaneously during exercise, the probability of severe coronary artery disease skyrockets.

    **The Feature Logic**: We created a binary flag (1 or 0) that triggers only if the patient has significant ST depression (> 1.0 mm) AND experiences exercise-induced angina.

    ### Vessel Blockage Severity Score (Severe_Blockage_Age_Adjusted)

    The Clinical Reality: The Number of vessels fluro feature tells us how many major coronary arteries show blockages under fluoroscopy (0 to 3). However, atherosclerosis (plaque buildup) is a progressive disease.  Having 2 blocked vessels is bad, but having 2 blocked vessels at age 38 is vastly more alarming than having 2 blocked vessels at age 75.

    The Feature Logic: We divide the number of blocked vessels by the patient's age.

    `Calculation: Number_of_vessels / Age`
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train_domain_ = (
        pl.from_pandas(ds_train)
        .with_columns(
            HR_Deficit=(220 - pl.col("Age")) - pl.col("Max HR"),
            High_Risk_Ischemia=(
                (pl.col("ST depression") > 1.0) & (pl.col("Exercise angina") == 1.0)
            ).cast(pl.Int32),
            Severe_Blockage_Age_Adjusted=(
                pl.col("Number of vessels fluro") / pl.col("Age")
            ),
        )
        .drop(
            [
                "Age",
                "Max HR",
                "ST depression",
                "Exercise angina",
                "Number of vessels fluro",
                "id",
            ]
        )
    )

    ds_train_domain_
    return (ds_train_domain_,)


@app.cell
def _(ds_train_domain_):
    ds_train_domain_.write_csv("data/processed/02_feature/features_domain.csv")
    return


if __name__ == "__main__":
    app.run()
