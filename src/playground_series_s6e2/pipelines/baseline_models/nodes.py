import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import logit


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into training and validation sets."""
    train_data, val_data = train_test_split(
        data,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )
    return train_data, val_data


def train_logit_model(train_data: pd.DataFrame, parameters: dict):
    """Fits a Logistic Regression model using statsmodels."""
    model = logit(formula=parameters["logit_formula"], data=train_data).fit(disp=False)
    return model


def evaluate_logit_model(model, val_data: pd.DataFrame, parameters: dict) -> tuple:
    """Evaluates the model and extracts metrics, summary, and coefficients."""
    y_pred_probs = model.predict(val_data)
    THRESHOLD = 0.5
    y_pred = (y_pred_probs >= THRESHOLD).astype(int)
    y_true = val_data[parameters["target_col"]]

    # Classification metrics
    metrics = {
        "accuracy": {"value": float(accuracy_score(y_true, y_pred)), "step": 0},
        "precision": {
            "value": float(precision_score(y_true, y_pred, zero_division=0)),
            "step": 0,
        },
        "recall": {
            "value": float(recall_score(y_true, y_pred, zero_division=0)),
            "step": 0,
        },
        "f1": {"value": float(f1_score(y_true, y_pred, zero_division=0)), "step": 0},
    }

    try:
        auc = float(roc_auc_score(y_true, y_pred_probs))
        metrics["roc_auc"] = {
            "value": auc,
            "step": 0,
        }
    except ValueError:
        auc = 0.0
        metrics["roc_auc"] = {"value": auc, "step": 0}

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "true_positives": {"value": float(tp), "step": 0},
            "true_negatives": {"value": float(tn), "step": 0},
            "false_positives": {"value": float(fp), "step": 0},
            "false_negatives": {"value": float(fn), "step": 0},
        }
    )

    # Statsmodels metrics
    metrics.update(
        {
            "log_likelihood": {"value": float(model.llf), "step": 0},
            "ll_null": {"value": float(model.llnull), "step": 0},
            "pseudo_r_squared": {"value": float(model.prsquared), "step": 0},
            "aic": {"value": float(model.aic), "step": 0},
            "bic": {"value": float(model.bic), "step": 0},
            "llr_pvalue": {"value": float(model.llr_pvalue), "step": 0},
        }
    )

    # Unified metrics dataframe for artifact logging
    unified_metrics_df = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": metrics["accuracy"]["value"]},
            {"Metric": "Precision", "Value": metrics["precision"]["value"]},
            {"Metric": "Recall", "Value": metrics["recall"]["value"]},
            {"Metric": "F1-Score", "Value": metrics["f1"]["value"]},
            {"Metric": "ROC-AUC", "Value": auc},
            {"Metric": "True Positives (TP)", "Value": float(tp)},
            {"Metric": "True Negatives (TN)", "Value": float(tn)},
            {"Metric": "False Positives (FP)", "Value": float(fp)},
            {"Metric": "False Negatives (FN)", "Value": float(fn)},
            {"Metric": "Log-Likelihood", "Value": float(model.llf)},
            {"Metric": "LL-Null", "Value": float(model.llnull)},
            {"Metric": "Pseudo R-squared", "Value": float(model.prsquared)},
            {"Metric": "AIC", "Value": float(model.aic)},
            {"Metric": "BIC", "Value": float(model.bic)},
            {"Metric": "LLR p-value", "Value": float(model.llr_pvalue)},
        ]
    )

    # Summary and Coefficients
    model_result_tables = model.summary2().tables
    summary_df = model_result_tables[0]
    coefficients_df = model_result_tables[1].reset_index()

    return metrics, summary_df, coefficients_df, unified_metrics_df


def create_submission(model, test_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Creates a Kaggle submission file."""
    y_pred_probs = model.predict(test_data)
    THRESHOLD = 0.5
    y_pred = (y_pred_probs >= THRESHOLD).astype(int)

    # Map back to string labels
    submission = pd.DataFrame({"id": test_data["id"], parameters["target_col"]: y_pred})

    return submission


def baseline_model_v1_submission(
    model, test_data: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Creates a Kaggle submission file for V1 with proper label mapping."""
    y_pred_probs = model.predict(test_data)
    THRESHOLD = 0.5
    y_pred = (y_pred_probs >= THRESHOLD).astype(int)

    submission = pd.DataFrame({"id": test_data["id"], parameters["target_col"]: y_pred})

    # Map back to string labels for Kaggle
    # submission[parameters["target_col"]] = submission[parameters["target_col"]].map(
    #     {0: "Absence", 1: "Presence"}
    # )

    return submission
