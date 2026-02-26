import marimo

__generated_with = "0.20.1"
app = marimo.App(
    width="full",
    app_title="Univariate Analysis",
    sql_output="polars",
)

with app.setup:
    import marimo as mo

    import altair as alt
    import numpy as np

    import polars as pl
    import polars.selectors as cs
    import pandas as pd

    import structlog
    import warnings

    from enum import Enum, auto
    from dataclasses import dataclass, asdict
    from typing import Dict, Any, Optional, List

    from visions.functional import infer_type
    from visions.typesets import StandardSet
    from visions import (
        Integer,
        Float,
        String,
        Boolean,
        Categorical,
        Date,
        Time,
        DateTime,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # <span style="color: #2c3e50">Feature Profiling</span>
    """)
    return


@app.function
def FEATURE_PROFILING_INSIGHTS():
    INSIGHTS = {
        "features datatype": {
            "nominal encoding": ["Chest pain type", "EKG results", "Thallium"],
            "ordinal encoding": ["Slope of ST", "Number of vessels fluro"],
            "binary encoding": [
                "Sex",
                "FBS over 120",
                "Exercise angina",
                "Heart Disease",
            ],
            "continuous": [
                "Age",
                "BP",
                "Cholesterol",
                "Max HR",
                "ST depression",
            ],
        }
    }

    return INSIGHTS


@app.cell
def _():
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    log = structlog.get_logger()
    return


@app.cell
def _():
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

        ds_train: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease")
        ds_test: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease_test")
    return ds_test, ds_train


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.glimpse()
    return


@app.cell
def _(ds_test: pl.DataFrame, ds_train: pl.DataFrame):
    TRAIN_FEATURES = ds_train.columns
    TEST_FEATURES = ds_test.columns
    return TEST_FEATURES, TRAIN_FEATURES


@app.cell
def _():
    mo.md("""
    Train Features
    """)
    return


@app.cell
def _(TRAIN_FEATURES):
    TRAIN_FEATURES
    return


@app.cell
def _():
    mo.md("""
    Test Features
    """)
    return


@app.cell
def _(TEST_FEATURES):
    TEST_FEATURES
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train
    return


@app.class_definition
class StatType:
    UNSUPPORTED = "UNSUPPORTED"
    CONSTANT = "CONSTANT"
    BINARY = "BINARY"
    NOMINAL_ID = "NOMINAL_ID"
    NOMINAL_CAT = "NOMINAL_CAT"
    ORDINAL = "ORDINAL"
    INTERVAL_DISCRETE = "INTERVAL_DISCRETE"
    INTERVAL_CONTINUOUS = "INTERVAL_CONTINUOUS"
    RATIO_DISCRETE = "RATIO_DISCRETE"
    RATIO_CONTINUOUS = "RATIO_CONTINUOUS"
    TEMPORAL = "TEMPORAL"
    TEXT = "TEXT"


@app.class_definition
@dataclass
class ColumnProfile:
    col_name: str
    stat_type: str
    python_type: str
    n_unique: int
    sparsity: float
    entropy: float
    inference_reason: str


@app.class_definition
class VariableTypeEngine:
    def __init__(self, sample_size: int = 50_000):
        self.sample_size = sample_size
        self.typeset = StandardSet()

    def _calculate_entropy(self, series: pd.Series) -> float:
        counts = series.value_counts(normalize=True, sort=False)
        if len(counts) <= 1:
            return 0.0
        # Normalize entropy to [0,1] range to make it comparable across columns with different cardinality
        entropy = -np.sum(counts * np.log2(counts))
        return entropy / np.log2(len(counts))

    def _is_equidistant(self, series: pd.Series) -> bool:
        uniques = np.sort(series.dropna().unique())
        if len(uniques) < 3:
            return True
        diffs = np.diff(uniques)
        # We check for constant step size (e.g., 1,2,3 vs 1,10,100) to distinguish true Ordinal logic from arbitrary integers
        return np.allclose(diffs, diffs[0], atol=1e-9)

    def _resolve_topology(
        self,
        col: str,
        series: pd.Series,
        v_type,
        n_unique: int,
        sparsity: float,
        entropy: float,
    ):
        clean_s = series.dropna()

        if n_unique <= 1:
            return StatType.CONSTANT, "Zero Variance"
        if n_unique == 2:
            return StatType.BINARY, "Binary"

        if v_type == Integer:
            # High entropy + high sparsity implies these integers are likely IDs (e.g. UserID), not features
            if sparsity > 0.90 and entropy > 0.95:
                return StatType.NOMINAL_ID, "High Card Integer (ID)"

            if n_unique < 20 or (sparsity < 0.05 and n_unique < 50):
                if self._is_equidistant(clean_s):
                    return StatType.ORDINAL, "Low Card, Equidistant"
                else:
                    return StatType.NOMINAL_CAT, "Low Card, Irregular Spacing"

            # Check for negatives to distinguish Interval (arbitrary zero, e.g. Year) vs Ratio (natural zero, e.g. Age)
            if clean_s.min() < 0:
                return StatType.INTERVAL_DISCRETE, "Integer with Negatives"
            else:
                return StatType.RATIO_DISCRETE, "Integer, Natural Zero"

        elif v_type == Float:
            if clean_s.min() < 0:
                return StatType.INTERVAL_CONTINUOUS, "Float with Negatives"
            else:
                return StatType.RATIO_CONTINUOUS, "Float, Natural Zero"

        elif v_type in [String, Categorical]:
            if sparsity > 0.9:
                return StatType.NOMINAL_ID, "High Card String"
            elif sparsity < 0.2:
                return StatType.NOMINAL_CAT, "Low Card String"
            else:
                return StatType.TEXT, "Unstructured Text"

        elif v_type in [Date, DateTime, Time]:
            return StatType.TEMPORAL, "Native Temporal"

        return StatType.UNSUPPORTED, f"Visions Fallback: {v_type}"

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        # We sample here because converting massive Polars frames to Pandas for Visions inference is a performance bottleneck
        if df.height > self.sample_size:
            df_pandas = df.sample(self.sample_size, seed=42).to_pandas()
        else:
            df_pandas = df.to_pandas()

        # Visions aggressively tries to parse dates from messy strings, causing log spam. We suppress this specific warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            inferred_types = infer_type(df_pandas, self.typeset)

        results = []

        for col in df_pandas.columns:
            series = df_pandas[col]
            v_type = inferred_types[col]

            n_unique = series.nunique()
            sparsity = n_unique / len(series) if len(series) > 0 else 0
            entropy = self._calculate_entropy(series)

            stat_type, reason = self._resolve_topology(
                col, series, v_type, n_unique, sparsity, entropy
            )

            results.append(
                ColumnProfile(
                    col_name=col,
                    stat_type=stat_type,
                    python_type=str(df[col].dtype),
                    n_unique=n_unique,
                    sparsity=round(sparsity, 5),
                    entropy=round(entropy, 4),
                    inference_reason=reason,
                )
            )

        return pl.DataFrame([asdict(r) for r in results])


@app.cell
def _(ds_train: pl.DataFrame):
    engine = VariableTypeEngine()
    ds_train_results = engine.run(ds_train)

    # Calculate KPI Cards
    total_cols = ds_train_results.height
    dominant_type = ds_train_results["stat_type"].mode().first()
    avg_sparsity = round(ds_train_results["sparsity"].mean(), 2)

    # Using on_select to allow future drill-down capabilities
    table_ui = mo.ui.table(
        ds_train_results,
        selection="multi",
        label="Feature Type Registry",
        page_size=10,
    )

    # D. Assemble Dashboard
    dashboard = mo.vstack(
        [
            mo.md(f"## Statistical Type Inference Engine"),
            # KPI Grid
            mo.hstack(
                [
                    mo.stat(
                        value=str(total_cols), label="Total Features", bordered=True
                    ),
                    mo.stat(
                        value=str(dominant_type),
                        label="Dominant Type",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{avg_sparsity:.2f}",
                        label="Avg Sparsity",
                        caption="1.0 = Unique, 0.0 = Constant",
                        bordered=True,
                    ),
                ],
                gap=1,
            ),
            mo.md("---"),
            # The Main Table
            mo.md("### Feature Topology Map"),
            mo.md("Select a row to inspect (Future Implementation)"),
            table_ui,
        ]
    )

    dashboard
    return (table_ui,)


@app.function
def create_feature_card(row, source_df):
    col = row["col_name"]
    stat_type = row["stat_type"]

    # Data Sample for Plotting (Max 5k rows for speed)
    plot_data = (
        source_df.select(col).sample(n=min(5000, len(source_df))).to_pandas()
    )

    # Dynamic Charting
    base = (
        alt.Chart(plot_data)
        .mark_bar()
        .encode(y="count()", tooltip=[col, "count()"])
    )

    if "CONTINUOUS" in stat_type or "DISCRETE" in stat_type:
        chart = base.encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col)
        )
    else:
        chart = base.encode(x=alt.X(f"{col}:N", sort="-y", title=col))

    chart = chart.properties(width=600, height=250, title=f"Dist: {col}")

    # Quality Alerts
    alerts = []
    if row["sparsity"] > 0.95 and "ID" not in stat_type:
        alerts.append("⚠️ **Near Unique**")
    if stat_type == "CONSTANT":
        alerts.append("⛔ **Zero Variance**")

    alert_md = " | ".join(alerts) if alerts else "✅ **Healthy**"

    # Card Layout
    return (
        mo.ui.table(pl.DataFrame([row]), pagination=False)
        if False
        else mo.vstack(
            [
                mo.md(f"### `{col}` ({stat_type})"),
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.md(f"**Unique:** {row['n_unique']}"),
                                mo.md(f"**Dtype:** `{row['python_type']}`"),
                                mo.md(f"**Status:** {alert_md}"),
                            ]
                        ),
                        mo.ui.altair_chart(chart),
                    ],
                    widths=[1, 2],
                    gap=2,
                ),
                mo.md("---"),
            ]
        )
    )


@app.cell
def _(ds_train: pl.DataFrame, table_ui):
    # Fetch Selection
    selected_df = table_ui.value


    # Render Output
    if selected_df.is_empty():
        report_content = mo.md(
            "*Select one or more rows in the table above to visualize.*"
        )
    else:
        # List comprehension to generate a report for EVERY selected row
        reports = [
            create_feature_card(row, ds_train) for row in selected_df.to_dicts()
        ]
        report_content = mo.vstack(reports, gap=1)

    report_content
    return


@app.cell(hide_code=True)
def domain_type_inference():
    mo.md(r"""
    ## Domain Type Inference

    Using domain knowledge and level of measurement to categories features.

    ### Continuous Features (Magnitude Matters)
    *Treatment: These measure physical quantities. We must Scale them because Logistic Regression thinks "bigger number = more important."*

    | Feature | Domain Label | Why is it **Continuous**? | Logistic Reg. Action (Baseline Model)|
    | :--- | :--- | :--- | :--- |
    | **Age** | Vascular Age | **Time is a continuum.** A 55-year-old has exactly 5 more years of arterial stiffening than a 50-year-old. The risk accumulates progressively. | **Standard Scale**<br>(Z-score normalization). |
    | **BP** | Resting BP | **It measures Pressure in mmHg.** 160 mmHg exerts physically more force on the artery walls than 120 mmHg. It is a physics measurement. | **Robust Scale**<br>(Use Median/IQR because BP often has extreme spikes/outliers). |
    | **Cholesterol** | Lipid Burden | **It measures Concentration in mg/dL.** It represents the *amount* of plaque-building material in the blood. More is linearly (or logarithmically) worse. | **Log Transform $\to$ Standard Scale**<br>(Fixes the right-skewed distribution). |
    | **Max HR** | Max Heart Rate | **It measures Rate in BPM.** It captures the exact physical limit of the heart engine. 150 bpm is a specific frequency. | **Standard Scale** |
    | **ST depression** | Ischemic Burden | **It measures Distance in mm.** A 2mm depression indicates physically more tissue suffocation than a 1mm depression. | **Standard Scale** |

    ---

    ### Nominal Features (Names, Not Numbers)
    *Treatment: These are categories disguised as numbers. We MUST "One-Hot Encode" them to prevent the model from finding false mathematical patterns.*

    | Feature | Domain Label | Why is it **Nominal**? | Logistic Reg. Action (Baseline Model)|
    | :--- | :--- | :--- | :--- |
    | **Chest pain type** | Symptom Type | **Identity Mistake.**<br>1=Typical, 2=Atypical, 3=Non-Pain, 4=Asymptomatic.<br>Is "Non-Pain" (3) three times worse than "Typical" (1)? No. They are just different descriptions. | **One-Hot Encode (drop='first')**<br>*Critical: Without this, the model fails.* |
    | **EKG results** | Resting EKG | **Pathology Mismatch.**<br>0=Normal, 1=Wave Issue, 2=Hypertrophy (Large Heart).<br>A "Large Heart" (2) is not double a "Wave Issue" (1). They are completely different biological problems. | **One-Hot Encode (drop='first')** |
    | **Thallium** | Thallium Defect | **Arbitrary Labels.**<br>3=Normal, 6=Fixed Scar, 7=Reversible Ischemia.<br>The numbers 3, 6, 7 are legacy codes from the machine. Treating 6 as "twice as bad" as 3 is mathematically absurd. | **One-Hot Encode (drop='first')** |

    ---

    ### Ordinal Features (Rank Matters)
    *Treatment: There is a clear "Ladder of Risk." We can treat these as Integers (1, 2, 3) because the order correlates with disease severity.*

    | Feature | Domain Label | Why is it **Ordinal**? | Logistic Reg. Action (Baseline Model)|
    | :--- | :--- | :--- | :--- |
    | **Slope of ST** | ST Slope | **Geometric Severity.**<br>1=Upsloping (Best), 2=Flat (Warning), 3=Downsloping (Critical).<br>The physiology moves from Healthy $\to$ Struggling $\to$ Failing. The order $1 < 2 < 3$ reflects reality. | **Keep as Integer**<br>(Preserve the monotonicity). |
    | **Number of vessels fluro** | Calcified Vessels | **Count of Damage.**<br>0, 1, 2, 3 vessels colored.<br>Having 3 blocked vessels is objectively worse than having 1. The "amount" of disease scales with the number. | **Keep as Integer** |

    ---

    ### Binary Features (Yes/No Switches)
    *Treatment: These are already perfect. 0 means "Off", 1 means "On".*

    | Feature | Domain Label | Why is it **Binary**? | Logistic Reg. Action (Baseline Model)|
    | :--- | :--- | :--- | :--- |
    | **Sex** | Biological Sex | **Biological Switch.**<br>In this dataset, there are only two states: Male (1) or Female (0). | **None** (Passthrough). |
    | **FBS over 120** | Diabetes Flag | **Threshold Switch.**<br>While sugar is continuous, this dataset *already converted it*. You are either High (>120) or Normal. It is a specific flag for Diabetes risk. | **None** (Passthrough). |
    | **Exercise angina** | Exertional Pain | **Symptom Switch.**<br>Did it hurt? Yes (1) or No (0). | **None** (Passthrough). |
    | **Heart Disease** | Target | **Outcome Switch.**<br>Presence (1) or Absence (0). | **Label Encode** |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
