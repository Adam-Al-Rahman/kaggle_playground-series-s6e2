import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import altair as alt
    import numpy as np

    import pandas as pd
    import polars as pl
    import polars.selectors as cs

    from sklearn.model_selection import train_test_split

    from statsmodels.formula.api import logit

    return logit, mo, np, pl, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline: Logistic Regression v1

    Post Feature engineering and selecting important features
    """)
    return


@app.cell
def _(pl):
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
            "feature_engineered.ds_important_features_train"
        )
        ds_test_raw: pl.DataFrame = catalog.load(
            "feature_engineered.ds_important_features_test"
        )
    return (ds_train_raw,)


@app.cell
def _(ds_train_raw: "pl.DataFrame", train_test_split):
    ds_train, ds_val = train_test_split(
        ds_train_raw, test_size=0.2, random_state=42
    )

    ds_train.sample(10)
    return ds_train, ds_val


@app.cell
def _(ds_train, logit):
    FORMULA = """
    Q('Heart Disease') ~ 
        HR_Deficit +
        Q('Chest pain type') +
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
def _(model, pl):
    PREDICTION = "data/models/predictions/"

    model_result_tables = model.summary2().tables

    df_summary_table = pl.from_pandas(model_result_tables[0])
    df_summary_table.write_csv(
        PREDICTION + "baseline_logit_v1_summary_raw.csv", include_header=False
    )

    # Extract and save the coefficients (Table 1 equivalent)
    df_coefs = pl.from_pandas(model_result_tables[1].reset_index())
    df_coefs.write_csv(PREDICTION + "baseline_logit_v1_coefficients.csv")
    return (PREDICTION,)


@app.cell
def _(ds_val):
    ds_val.head()
    return


@app.cell
def _(ds_val, model, pl):
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
def _(PREDICTION, model, np, pl, results, y_pred_probs):
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

    unified_metrics.write_csv(PREDICTION + "baseline_logit_v1_metrics.csv")

    unified_metrics
    return


@app.cell
def _(model):
    # Baseline logit post preprocess
    model.save("data/models/artifacts/baseline/baseline_logit_v1.pkl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theory & Interpretation of Your Results

    Your logit (logistic regression) output provides a window into how well your chosen variables explain the presence of heart disease.

    * **Pseudo R-squared (0.4377):** In ordinary least squares (OLS) regression, R-squared represents the percentage of variance explained. Logistic regression uses McFadden's Pseudo R-squared instead. A value between 0.2 and 0.4 indicates an excellent model fit. At 0.4377, your model's predictive power is remarkable, especially for complex biological/medical data.
    * **LLR p-value (0.000):** The Log-Likelihood Ratio tests whether your model as a whole is better than a "null" model (a model with no predictors that just guesses the most frequent outcome). A value of 0.000 confirms that your features collectively provide a massive improvement in predictive accuracy.
    * **The Power of Large N:** You have 504,000 observations. With a sample size this massive, standard errors naturally shrink to near zero, meaning your p-values (P>|z|) will almost always be 0.000. In these cases, we stop looking at p-values to determine importance and instead look at the **coefficients** (effect sizes).

    The relationship is defined by the logistic function, mapping any real-valued number into a probability between 0 and 1:


    $$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3)}}$$

    Looking at your coefficients:

    * **Chest pain type (0.8960):** This is your strongest predictor. For every one-unit increase in the category of chest pain, the log-odds of having heart disease increase by roughly 0.89.
    * **Thallium (0.6581):** Also a massive contributor to the risk profile.
    * **HR_Deficit (0.0329):** While highly significant, its individual per-unit impact on the log-odds is relatively small compared to the other two features.
    """)
    return


if __name__ == "__main__":
    app.run()
