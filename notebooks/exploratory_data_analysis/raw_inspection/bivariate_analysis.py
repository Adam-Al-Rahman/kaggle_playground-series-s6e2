import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd

    import scipy.stats as stats
    import polars.selectors as cs

    from sklearn.metrics import mutual_info_score
    from sklearn.feature_selection import mutual_info_regression, f_classif

    import nannyml as nml

    nml.disable_usage_logging()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bivariate Analysis
    """)
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
    return (ds_train,)


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Based on [Feature Profiling](http://localhost:2718/?file=notebooks%2Fexploratory_data_analysis%2Fraw_inspection%2Ffeature_profiling.py.py#scrollTo=domain_type_inference) we got to know about feature types from
    """)
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
    return CATEGORICAL_FEATURES, CONT_FEATURES, FEATURES


@app.class_definition
class BivariateAnalyzer:
    """Modular suite for performing bivariate analysis and visualization."""

    @staticmethod
    def num_vs_num(df: pl.DataFrame, col_x: str, col_y: str) -> mo.Html:
        pearson_corr = df.select(pl.corr(col_x, col_y)).item()
        spearman_corr = df.select(
            pl.corr(col_x, col_y, method="spearman")
        ).item()

        x_array = df[col_x].to_numpy().reshape(-1, 1)
        y_array = df[col_y].to_numpy()
        mi_score = mutual_info_regression(x_array, y_array)[0]

        metrics_md = mo.md(f"""
        **Numerical vs Numerical Metrics ({col_x} vs {col_y}):**
        - **Pearson (Linear):** {pearson_corr:.4f}
        - **Spearman (Monotonic):** {spearman_corr:.4f}
        - **Mutual Info (Non-Linear):** {mi_score:.4f}
        """)

        sample_df = df.sample(n=min(5000, df.height)).to_pandas()

        scatter = (
            alt.Chart(sample_df)
            .mark_circle(opacity=0.5)
            .encode(
                x=alt.X(col_x, type="quantitative"),
                y=alt.Y(col_y, type="quantitative"),
            )
            .properties(title="Scatter Plot", width=300, height=300)
        )

        binned_rect = (
            alt.Chart(sample_df)
            .mark_rect()
            .encode(
                x=alt.X(col_x, type="quantitative", bin=alt.Bin(maxbins=40)),
                y=alt.Y(col_y, type="quantitative", bin=alt.Bin(maxbins=40)),
                color=alt.Color("count()", scale=alt.Scale(scheme="viridis")),
            )
            .properties(title="2D Density (Binned)", width=300, height=300)
        )

        # FIX: Directly pass the concatenated chart object to vstack.
        # Marimo will natively render the raw Altair grammar of graphics.
        charts = scatter | binned_rect
        return mo.vstack([metrics_md, charts])

    @staticmethod
    def cat_vs_cat(df: pl.DataFrame, col_x: str, col_y: str) -> mo.Html:
        x_array = df[col_x].to_numpy()
        y_array = df[col_y].to_numpy()
        mi_score = mutual_info_score(x_array, y_array)

        metrics_md = mo.md(f"""
        **Categorical vs Categorical Metrics ({col_x} vs {col_y}):**
        - **Mutual Information Score:** {mi_score:.4f}
        """)

        contingency = (
            df.group_by([col_x, col_y]).agg(pl.len().alias("count")).to_pandas()
        )

        heatmap = (
            alt.Chart(contingency)
            .mark_rect()
            .encode(
                x=alt.X(f"{col_x}:N"),
                y=alt.Y(f"{col_y}:N"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=[col_x, col_y, "count"],
            )
            .properties(title="Contingency Heatmap", width=400, height=300)
        )

        return mo.vstack([metrics_md, heatmap])

    @staticmethod
    def num_vs_cat(df: pl.DataFrame, num_col: str, cat_col: str) -> mo.Html:
        x_array = df[num_col].to_numpy().reshape(-1, 1)
        y_array = df[cat_col].to_numpy()
        f_val, p_val = f_classif(x_array, y_array)

        metrics_md = mo.md(f"""
        **Numerical vs Categorical Metrics ({num_col} over {cat_col}):**
        - **ANOVA F-value:** {f_val[0]:.4f}
        - **ANOVA p-value:** {p_val[0]:.4e}
        """)

        sample_df = df.sample(n=min(5000, df.height)).to_pandas()

        boxplot = (
            alt.Chart(sample_df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X(f"{cat_col}:N"),
                y=alt.Y(f"{num_col}:Q"),
                color=alt.Color(f"{cat_col}:N", legend=None),
            )
            .properties(title="Grouped Boxplot", width=300, height=300)
        )

        point_plot = alt.Chart(sample_df).mark_errorband(extent="ci").encode(
            x=alt.X(f"{cat_col}:N"), y=alt.Y(f"{num_col}:Q")
        ) + alt.Chart(sample_df).mark_point(color="black", size=50).encode(
            x=alt.X(f"{cat_col}:N"), y=alt.Y(f"{num_col}:Q", aggregate="mean")
        ).properties(title="Mean + 95% CI", width=300, height=300)

        charts = boxplot | point_plot
        return mo.vstack([metrics_md, charts])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NUMERICAL vs. NUMERICAL
    """)
    return


@app.cell
def _(CONT_FEATURES):
    CONT_FEATURES
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[0]} Vs. {CONT_FEATURES[1]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    age_bp = BivariateAnalyzer.num_vs_num(ds_train, "Age", "BP")
    age_bp
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pearson (r): Measures the linear relationship. It asks: "If X goes up by 1 unit, does Y go up by a constant amount?" A value of −0.0040 is effectively zero. There is no straight-line relationship.

    Spearman (ρ): Measures the monotonic relationship using ranks. It asks: "If X increases, does Y generally increase, even if it's not a straight line?" A value of −0.0027 means there isn't even a consistent "upward" or "downward" trend.

    Mutual Information (MI): Measures dependency (including non-linear). It uses entropy to ask: "How much does knowing X reduce my uncertainty about Y?" A value of 0.0005 is extremely low, suggesting that X provides almost no information about Y.
    """)
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[0]} vs. {CONT_FEATURES[2]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    age_cholesterol = BivariateAnalyzer.num_vs_num(ds_train, "Age", "Cholesterol")
    age_cholesterol
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[0]} vs. {CONT_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    age_max_hr = BivariateAnalyzer.num_vs_num(ds_train, "Age", "Max HR")
    age_max_hr
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[0]} vs. {FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    age_st_depression = BivariateAnalyzer.num_vs_num(
        ds_train, "Age", "ST depression"
    )
    age_st_depression
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    These specific values—Pearson at 0.1031, Spearman at 0.1065, and Mutual Information (MI) at 0.0056 --- tell a very specific, interlocking mathematical story.

    We can test if your data represents a simple, noisy linear relationship without any hidden non-linear secrets by looking at the relationship between Pearson's r and Mutual Information $I(X;Y)$ for bivariate normal distributions:

    $$
    I(X;Y) = -\frac{1}{2} ln(1 - r^2)
    $$

    Let's plug Pearson value into the derivation:

    - Square the Pearson correlation: $0.1031^2 = 0.0106$
    - Calculate the natural log of the remainder: $ln(1−0.0106) \approx −0.0106$
    - Multiply by $−0.5: −\frac{1}{2} \times −0.0106 \approx 0.0053$

    Your actual empirical Mutual Information is 0.0056, which almost perfectly matches the theoretical Gaussian limit of 0.0053.

    **The Conclusion**: This proves there are no hidden, complex, or non-linear patterns. The slight 10% correlation is entirely linear, and the remaining 90% is purely normally distributed noise.
    """)
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[1]} vs. {CONT_FEATURES[2]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    bp_cholesterol = BivariateAnalyzer.num_vs_num(ds_train, "BP", "Cholesterol")
    bp_cholesterol
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[1]} vs. {CONT_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    bp_max_hr = BivariateAnalyzer.num_vs_num(ds_train, "BP", "Max HR")

    bp_max_hr
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[1]} vs. {CONT_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    bp_st_depression = BivariateAnalyzer.num_vs_num(ds_train, "BP", "ST depression")

    bp_st_depression
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[2]} vs. {CONT_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    cholesterol_max_hr = BivariateAnalyzer.num_vs_num(
        ds_train, "Cholesterol", "Max HR"
    )

    cholesterol_max_hr
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[2]} vs. {CONT_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    cholesterol_st_depression = BivariateAnalyzer.num_vs_num(
        ds_train, "Cholesterol", "ST depression"
    )
    cholesterol_st_depression
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[3]} vs. {CONT_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    max_hr_st_depression = BivariateAnalyzer.num_vs_num(
        ds_train, "Max HR", "ST depression"
    )
    max_hr_st_depression
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CATEGORICAL vs. CATEGORICAL
    """)
    return


@app.cell
def _(CATEGORICAL_FEATURES):
    CATEGORICAL_FEATURES
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[1]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_ekg_results = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "EKG results"
    )
    chest_pain_ekg_results
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[2]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_thallium = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "Thallium"
    )
    chest_pain_thallium
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_slope_st = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "Slope of ST"
    )

    chest_pain_slope_st
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_vessels = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "Number of vessels fluro"
    )

    chest_pain_vessels
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[5]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_sex = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "Sex"
    )

    chest_pain_sex
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_fbs = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "FBS over 120"
    )

    chest_pain_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[0]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    chest_pain_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Chest pain type", "Exercise angina"
    )

    chest_pain_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs. {CATEGORICAL_FEATURES[2]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_thallium = BivariateAnalyzer.cat_vs_cat(
        ds_train, "EKG results", "Thallium"
    )

    ekg_results_thallium
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs. {CATEGORICAL_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_slope_st = BivariateAnalyzer.cat_vs_cat(
        ds_train, "EKG results", "Slope of ST"
    )

    ekg_results_slope_st
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs {CATEGORICAL_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_vessels = BivariateAnalyzer.cat_vs_cat(
        ds_train, "EKG results", "Number of vessels fluro"
    )

    ekg_results_vessels
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs. {CATEGORICAL_FEATURES[5]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_sex = BivariateAnalyzer.cat_vs_cat(ds_train, "EKG results", "Sex")

    ekg_results_sex
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_fbs = BivariateAnalyzer.cat_vs_cat(
        ds_train, "EKG results", "FBS over 120"
    )

    ekg_results_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[1]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ekg_results_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "EKG results", "Exercise angina"
    )

    ekg_results_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[2]} vs. {CATEGORICAL_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    thallium_slope_st = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Thallium", "Slope of ST"
    )

    thallium_slope_st
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[2]} vs. {CATEGORICAL_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    thallium_vessels = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Thallium", "Number of vessels fluro"
    )

    thallium_vessels
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[2]} vs {CATEGORICAL_FEATURES[5]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    thallium_sex = BivariateAnalyzer.cat_vs_cat(ds_train, "Thallium", "Sex")

    thallium_sex
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[2]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    thallium_fbs = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Thallium", "FBS over 120"
    )

    thallium_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[2]} vs {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    thallium_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Thallium", "Exercise angina"
    )

    thallium_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[3]} vs. {CATEGORICAL_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    slope_st_vessels = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Slope of ST", "Number of vessels fluro"
    )

    slope_st_vessels
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[3]} vs. {CATEGORICAL_FEATURES[5]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    slope_st_sex = BivariateAnalyzer.cat_vs_cat(ds_train, "Slope of ST", "Sex")

    slope_st_sex
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[3]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    slope_st_fbs = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Slope of ST", "FBS over 120"
    )

    slope_st_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[3]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    slope_st_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Slope of ST", "Exercise angina"
    )

    slope_st_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[4]} vs. {CATEGORICAL_FEATURES[5]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    vessels_sex = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Number of vessels fluro", "Sex"
    )

    vessels_sex
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[4]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    vessels_fbs = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Number of vessels fluro", "FBS over 120"
    )

    vessels_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[4]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    vessels_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "Number of vessels fluro", "Exercise angina"
    )

    vessels_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[5]} vs. {CATEGORICAL_FEATURES[6]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    sex_fbs = BivariateAnalyzer.cat_vs_cat(ds_train, "Sex", "FBS over 120")

    sex_fbs
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[5]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    sex_angina = BivariateAnalyzer.cat_vs_cat(ds_train, "Sex", "Exercise angina")

    sex_angina
    return


@app.cell(hide_code=True)
def _(CATEGORICAL_FEATURES):
    mo.md(rf"""
    ### Feature: {CATEGORICAL_FEATURES[6]} vs. {CATEGORICAL_FEATURES[7]}
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    fbs_angina = BivariateAnalyzer.cat_vs_cat(
        ds_train, "FBS over 120", "Exercise angina"
    )

    fbs_angina
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NUMERICAL vs. CATEGORICAL
    """)
    return


@app.cell
def _(CATEGORICAL_FEATURES, CONT_FEATURES, ds_train: pl.DataFrame):
    dashboard_elements = {}

    for cont in CONT_FEATURES:
        for cat in CATEGORICAL_FEATURES:
            pair_analysis_ui = BivariateAnalyzer.num_vs_cat(ds_train, cont, cat)

            ui_label = f"### Distribution: {cont} segmented by {cat}"

            dashboard_elements[ui_label] = pair_analysis_ui

    interactive_report = mo.accordion(dashboard_elements)

    mo.vstack([interactive_report])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
