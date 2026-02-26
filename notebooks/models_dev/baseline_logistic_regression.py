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

    from sklearn.model_selection import train_test_split

    from statsmodels.formula.api import logit


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Baseline: Logistic Regression
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

        ds_train_raw: pl.DataFrame = catalog.load(
            "processed.ds_heart_disease_train"
        )
        ds_test_raw: pl.DataFrame = catalog.load("processed.ds_heart_disease_test")
    return (ds_train_raw,)


@app.cell
def _(ds_train_raw: pl.DataFrame):
    ds_train, ds_val = train_test_split(
        ds_train_raw, test_size=0.2, random_state=42
    )

    ds_train.sample(10)
    return ds_train, ds_val


@app.cell
def _(ds_train):
    FORMULA = """
    Q('Heart Disease') ~ 
        Age +
        Sex +
        Q('Chest pain type') +
        BP + Cholesterol +
        Q('FBS over 120') +
        Q('EKG results') +
        Q('Max HR') +
        Q('Exercise angina') +
        Q('ST depression') +
        Q('Slope of ST') +
        Q('Number of vessels fluro') +
        Thallium
    """

    # Fit the Logistic Regression model using Maximum Likelihood Estimation
    model = logit(formula=FORMULA, data=ds_train).fit(disp=False)
    return (model,)


@app.cell
def _(model):
    print(model.summary())
    return


@app.cell
def _(model):
    PREDICTION = "data/models/predictions/"

    model_result_tables = model.summary2().tables

    df_summary_table = pl.from_pandas(model_result_tables[0])
    df_summary_table.write_csv(
        PREDICTION + "baseline_logit_v0_summary_raw.csv", include_header=False
    )

    # Extract and save the coefficients (Table 1 equivalent)
    df_coefs = pl.from_pandas(model_result_tables[1].reset_index())
    df_coefs.write_csv(PREDICTION + "baseline_logit_v0_coefficients.csv")

    return (PREDICTION,)


@app.cell
def _(ds_val):
    ds_val.head()
    return


@app.cell
def _(ds_val, model):
    y_pred_probs = model.predict(ds_val)

    # Apply the threshold to discretize into classes
    THRESHOLD = 0.5
    y_pred_class = (y_pred_probs >= THRESHOLD).astype(int)

    # Construct the auditable output table natively in Polars
    # We extract the original IDs, add our discrete predictions, and add the actual targets.
    results = pl.DataFrame(
        {
            "id": ds_val["id"],
            "y_pred": y_pred_class,
            "y_actual": ds_val["Heart Disease"],
        }
    )

    results
    return results, y_pred_probs


@app.cell
def _(PREDICTION, model, results, y_pred_probs):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    y_true = results["y_actual"].to_numpy()
    y_pred = results["y_pred"].to_numpy()

    # Calculate the core discrete metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate continuous probabilistic metric (requires the raw probabilities)
    # If ROC-AUC cannot be calculated (e.g., only one class present in a tiny sample), we handle it.
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = float("nan")

    # Extract the Confusion Matrix components
    # ravel() flattens the 2x2 matrix into its 4 distinct components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


    # Extract Macro Statistical Metrics (Statsmodels)
    # We cast everything to float to ensure schema alignment
    df_stats = pl.DataFrame(
        {
            "Metric": [
                "Log-Likelihood",
                "LL-Null",
                "Pseudo R-squared",
                "AIC",
                "BIC",
                "LLR p-value",
            ],
            "Value": [
                float(model.llf),
                float(model.llnull),
                float(model.prsquared),
                float(model.aic),
                float(model.bic),
                float(model.llr_pvalue),
            ],
        }
    )

    # 2. Extract Classification Performance Metrics (Sklearn)
    y_true = results["y_actual"].to_numpy()
    y_pred = results["y_pred"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    df_perf = pl.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC",
                "True Positives (TP)",
                "True Negatives (TN)",
                "False Positives (FP)",
                "False Negatives (FN)",
            ],
            "Value": [
                float(accuracy_score(y_true, y_pred)),
                float(precision_score(y_true, y_pred, zero_division=0)),
                float(recall_score(y_true, y_pred, zero_division=0)),
                float(f1_score(y_true, y_pred, zero_division=0)),
                float(
                    roc_auc_score(y_true, y_pred_probs)
                    if "y_pred_probs" in locals()
                    else np.nan
                ),
                float(tp),
                float(tn),
                float(fp),
                float(fn),
            ],
        }
    )

    unified_metrics = pl.concat([df_stats, df_perf])

    unified_metrics.write_csv(PREDICTION + "baseline_logit_v0_metrics.csv")

    unified_metrics
    return


@app.cell
def _(model):
    # Baseline logit post preprocess
    model.save("data/models/artifacts/baseline/baseline_logit_v0.pkl")
    return


if __name__ == "__main__":
    app.run()
