import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd

    import scipy.stats as stats
    import polars.selectors as cs

    import nannyml as nml

    nml.disable_usage_logging()


@app.cell
def _():
    from exploratory_data_analysis.raw_inspection.feature_profiling import (
        FEATURE_PROFILING_INSIGHTS,
    )

    mo.callout(
        mo.vstack(
            [
                mo.md("## Extracted Insights"),
                mo.tree(FEATURE_PROFILING_INSIGHTS()),
            ]
        ),
        kind="success",
    )
    return


@app.cell
def _():
    mo.md("""
    # Univariate Analysis
    """)
    return


@app.function
def UNIVARIATE_ANALYSIS_INSIGHTS():
    INSIGHTS = {
        "exist outliers": ["BP", "Cholesterol", "Max HR", "ST depression"],
        "skewed": ["BP", "Max HR", "ST depression"]
    }

    return INSIGHTS


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

        ds_train_raw: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease")
        ds_test_raw: pl.DataFrame = catalog.load(
            "raw_ingestion.ds_heart_disease_test"
        )
    return ds_test_raw, ds_train_raw


@app.cell
def _(ds_train_raw: pl.DataFrame):
    ds_train_raw
    return


@app.cell
def _():
    TARGET = "Heart Disease"
    CONT_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    NOMINAL_FEATURES = ["Chest pain type", "EKG results", "Thallium"]
    ORDINAL_FEATURES = ["Slope of ST", "Number of vessels fluro"]
    BINARY_FEATURES = ["Sex", "FBS over 120", "Exercise angina"]

    FEATURES = CONT_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    return (
        BINARY_FEATURES,
        CONT_FEATURES,
        FEATURES,
        NOMINAL_FEATURES,
        ORDINAL_FEATURES,
        TARGET,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## NUMERICAL (Continuous/Discrete)
    """)
    return


@app.cell
def _(CONT_FEATURES):
    CONT_FEATURES
    return


@app.function
def numerical_univariate_dashboard(df: pl.DataFrame, col_name: str):
    """
    Creates an interactive Altair dashboard.
    Args:
    - df: cleaned dataframe (removed null, duplicates)
    - col_name: numerical column name to analyze
    """
    # We drop nulls and sort to calculate sample quantiles
    df = df.sort(col_name)
    n_total = df.height
    # Subsample if large to keep browser fast
    if n_total > 5000:
        viz_df = df.sample(5000, with_replacement=False).sort(col_name)
        subtitle = f"(Subsampled 5k / {n_total:,})"
    else:
        viz_df = df
        subtitle = f"(N = {n_total:,})"

    # 2. Add Q-Q Plot Coordinates
    n_viz = viz_df.height
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, n_viz))
    viz_df = viz_df.with_columns(
        pl.Series(name="theoretical_q", values=theoretical_q)
    )

    # 3. Visualization Components
    brush = alt.selection_interval(encodings=["x"], name="brush")

    base = (
        alt.Chart(viz_df)
        .encode(x=alt.X(f"{col_name}:Q", title=f"{col_name} {subtitle}"))
        .properties(width=400, height=220)
    )

    # A: Histogram (Driver)
    hist = (
        base.mark_bar()
        .encode(
            y=alt.Y("count()", title="Frequency"),
            color=alt.condition(
                brush, alt.value("#4c78a8"), alt.value("lightgray")
            ),
            tooltip=[col_name, "count()"],
        )
        .add_params(brush)
        .properties(title="1. Distribution (Drag to Filter)")
    )

    # B: Density
    kde = (
        base.transform_density(col_name, as_=[col_name, "density"])
        .mark_area(opacity=0.5, color="red")
        .encode(y="density:Q", x=alt.X(f"{col_name}:Q", title=None))
        .transform_filter(brush)
        .properties(title="2. Density Estimate")
    )

    # C: Q-Q Plot
    qq_points = (
        base.mark_circle(size=20)
        .encode(
            x=alt.X("theoretical_q:Q", title="Theoretical Q"),
            y=alt.Y(f"{col_name}:Q", title="Sample Q"),
            color=alt.condition(
                brush, alt.value("blue"), alt.value("lightgray")
            ),
        )
        .transform_filter(brush)
    )

    qq_line = (
        base.mark_line(color="red", strokeDash=[5, 5])
        .encode(x="theoretical_q:Q", y=f"{col_name}:Q")
        .transform_regression("theoretical_q", col_name)
        .transform_filter(brush)
    )

    # D: Box Plot
    box = (
        base.mark_boxplot(extent=1.5)
        .encode(x=f"{col_name}:Q")
        .transform_filter(brush)
        .properties(height=220, title="4. Box Plot")
    )

    # Layout
    chart = (hist | kde) & (
        (qq_points + qq_line).properties(title="3. Q-Q Plot") | box
    )
    return mo.ui.altair_chart(chart.properties(title=f"Analysis: {col_name}"))


@app.function
def display_reactive_stats(selection_data, full_df, col_name):
    # 1. Determine which data to use (Full vs Selection)
    use_selection = False

    # Check if selection_data has content
    if selection_data is not None:
        if (
            isinstance(selection_data, pl.DataFrame)
            and selection_data.height > 0
        ):
            use_selection = True
        elif isinstance(selection_data, list) and len(selection_data) > 0:
            use_selection = True

    if use_selection:
        # --- CASE A: User Selected a Region ---
        title = "📊 Statistics (Selected Region)"

        # Extract data from selection object
        try:
            if isinstance(selection_data, pl.DataFrame):
                data = selection_data[col_name].drop_nulls().to_numpy()
            else:
                # Handle list of dicts (JSON)
                data = np.array([d[col_name] for d in selection_data])
        except Exception:
            return mo.md("⚠️ Error parsing selection data.")

    else:
        # --- CASE B: Default / No Selection ---
        title = "📊 Statistics (Full Dataset)"
        data = full_df[col_name].drop_nulls().to_numpy()

    # 2. Calculate Statistics
    n = len(data)
    if n < 2:
        return mo.md(f"⚠️ **Not enough data** (N={n})")

    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)

    # Normality
    if n <= 5000:
        _, p = stats.shapiro(data)
        norm_str = f"p={p:.2e} ({'Normal' if p > 0.05 else 'Not Normal'})"
    else:
        norm_str = "Skipped (N > 5k)"

    # Outliers (Robust MAD)
    mad = np.median(np.abs(data - median_val))
    mad = mad if mad > 1e-9 else 1e-9
    mod_z = 0.6745 * (data - median_val) / mad
    n_outliers = np.sum(np.abs(mod_z) > 3.5)

    # 3. Render
    return mo.md(
        f"""
        {title}

        | Metric | Value | Interpretation |
        | :--- | :--- | :--- |
        | **Count** | `{n:,}` | Samples |
        | **Mean** | `{mean_val:.3f}` | Average |
        | **Median** | `{median_val:.3f}` | Center |
        | **Std Dev** | `{std_val:.3f}` | Spread |
        | **Skew** | `{skew:.3f}` | {">0.5 (Right)" if skew > 0.5 else ("<-0.5 (Left)" if skew < -0.5 else "Symmetric")} |
        | **Kurtosis** | `{kurt:.3f}` | {">3 (Heavy)" if kurt > 3 else "Normal-ish"} |
        | **Outliers** | `{n_outliers}` | Mod-Z > 3.5 |
        | **Normality** | `{norm_str}` | Shapiro-Wilk |
        """
    )


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[0]}
    """)
    return


@app.cell
def _(ds_train_raw: pl.DataFrame):
    age_dashboard = numerical_univariate_dashboard(ds_train_raw, "Age")
    age_dashboard
    return (age_dashboard,)


@app.cell
def _(age_dashboard, ds_train_raw: pl.DataFrame):
    display_reactive_stats(age_dashboard.value, ds_train_raw, "Age")
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Features: {CONT_FEATURES[1]}
    """)
    return


@app.cell
def _(ds_train_raw: pl.DataFrame):
    bp_dashboard = numerical_univariate_dashboard(ds_train_raw, "BP")
    bp_dashboard
    return (bp_dashboard,)


@app.cell
def _(bp_dashboard, ds_train_raw: pl.DataFrame):
    display_reactive_stats(bp_dashboard.value, ds_train_raw, "BP")
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[2]}
    """)
    return


@app.cell
def _(ds_train_raw: pl.DataFrame):
    cholesterol_dashboard = numerical_univariate_dashboard(
        ds_train_raw, "Cholesterol"
    )
    cholesterol_dashboard
    return (cholesterol_dashboard,)


@app.cell
def _(cholesterol_dashboard, ds_train_raw: pl.DataFrame):
    display_reactive_stats(cholesterol_dashboard.value, ds_train_raw, "Cholesterol")
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[3]}
    """)
    return


@app.cell
def _(ds_train_raw: pl.DataFrame):
    max_hr_dashboard = numerical_univariate_dashboard(ds_train_raw, "Max HR")
    max_hr_dashboard
    return (max_hr_dashboard,)


@app.cell
def _(ds_train_raw: pl.DataFrame, max_hr_dashboard):
    display_reactive_stats(max_hr_dashboard.value, ds_train_raw, "Max HR")
    return


@app.cell(hide_code=True)
def _(CONT_FEATURES):
    mo.md(rf"""
    ### Feature: {CONT_FEATURES[4]}
    """)
    return


@app.cell
def _(ds_train_raw: pl.DataFrame):
    st_depression_dashboard = numerical_univariate_dashboard(
        ds_train_raw, "ST depression"
    )
    st_depression_dashboard
    return (st_depression_dashboard,)


@app.cell
def _(ds_train_raw: pl.DataFrame, st_depression_dashboard):
    display_reactive_stats(
        st_depression_dashboard.value, ds_train_raw, "ST depression"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## CATEGORICAL (Nominal/Ordinal)
    """)
    return


@app.cell
def _(BINARY_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES):
    NOMINAL_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES
    return


@app.function
def categorical_univariate_analysis(df: pl.DataFrame, col_name: str, target: str):
    """
    Generates visual and statistical univariate analysis for a categorical feature,
    including cardinality and categorical skewness (Normalized Shannon Entropy).
    """
    if col_name == target:
        mo.output.append(mo.md(f"### Target: **{col_name}**"))
    else:
        mo.output.append(mo.md(f"### Feature: **{col_name}**"))

    # Calculate frequencies and proportions
    freq_df = (
        df.group_by(col_name)
        .len()
        .sort("len", descending=True)
        .with_columns((pl.col("len") / pl.sum("len")).alias("proportion"))
        .to_pandas()
    )

    # CHART 1: Bar Chart
    bar_chart = (
        alt.Chart(freq_df)
        .mark_bar(color="teal")
        .encode(
            x=alt.X(col_name, sort="-y", title="Category"),
            y=alt.Y("len", title="Count"),
            tooltip=[col_name, "len", alt.Tooltip("proportion", format=".1%")],
        )
        .properties(title="Frequency Bar Chart", width=400, height=300)
        .interactive()
    )

    # CHART 2: Donut Chart
    donut_chart = (
        alt.Chart(freq_df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="len", type="quantitative"),
            color=alt.Color(field=col_name, type="nominal"),
            tooltip=[col_name, "len", alt.Tooltip("proportion", format=".1%")],
        )
        .properties(title="Proportional Distribution", width=400, height=300)
    )

    # Combine and Display Charts
    combined_chart = alt.hconcat(bar_chart, donut_chart)
    mo.output.append(mo.ui.altair_chart(combined_chart))

    # --- CATEGORICAL SKEWNESS CALCULATION ---
    # Extract proportions as a numpy array for vectorized math
    p_i = freq_df["proportion"].to_numpy()
    
    # Filter out absolute zeros to avoid log2(0) runtime warnings
    p_i = p_i[p_i > 0] 
    k = len(p_i)
    
    if k > 1:
        entropy = -np.sum(p_i * np.log2(p_i))
        max_entropy = np.log2(k)
        skewness_index = 1.0 - (entropy / max_entropy)
    else:
        # If there is only 1 category, the distribution is completely skewed
        skewness_index = 1.0 

    # --- CARDINALITY METRICS ---
    unique_count = df[col_name].n_unique()
    uniqueness_ratio = unique_count / df.height

    # Display Statistics via Marimo UI
    mo.output.append(
        mo.hstack(
            [
                mo.stat(label="Unique Categories", value=str(unique_count)),
                mo.stat(label="Uniqueness Ratio", value=f"{uniqueness_ratio:.4%}"),
                mo.stat(label="Categorical Skewness", value=f"{skewness_index:.4f}"),
            ]
        )
    )
    mo.output.append(mo.md("---"))


@app.cell
def _(
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    TARGET,
    ds_train_raw: pl.DataFrame,
):
    for variable in (
        NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + [TARGET]
    ):
        categorical_univariate_analysis(ds_train_raw, variable, TARGET)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train Vs. Test Univariate Distribution

    Check the distribution of each train feature against test variable

    Question: Is the data in the real world (Test) physically similar to the data I am learning from (Train)?
    """)
    return


@app.cell
def _(ds_train):
    ds_train.columns
    return


@app.function
def plot_reference_analysis_dist(
    reference: pl.DataFrame,
    analysis: pl.DataFrame,
    feature: str,
    is_categorical: bool = False,
):
    df_plot = pl.concat(
        [
            reference.select(
                pl.col(feature), pl.lit("Reference").alias("Source")
            ),
            analysis.select(
                pl.col(feature), pl.lit("Analysis").alias("Source")
            ),
        ]
    )

    color_scale = alt.Scale(
        domain=["Reference", "Analysis"], range=["#1f77b4", "#ff7f0e"]
    )

    if is_categorical:
        # Route: Probability Mass Function (Normalized Proportions)
        data = (
            df_plot.group_by(["Source", feature])
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("Source")).alias(
                    "proportion"
                )
            )
        )

        chart = (
            alt.Chart(data)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X(f"{feature}:N", title=f"{feature} Category"),
                y=alt.Y(
                    "proportion:Q",
                    title="Relative Frequency",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("Source:N", scale=color_scale),
                xOffset=alt.XOffset("Source:N"),
            )
        )
    else:
        # Route: Probability Density Function (KDE)
        chart = (
            alt.Chart(df_plot)
            .transform_density(
                feature, groupby=["Source"], as_=[feature, "density"]
            )
            .mark_area(opacity=0.5)
            .encode(
                x=alt.X(f"{feature}:Q", title=f"{feature} Value"),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color("Source:N", scale=color_scale),
            )
        )

    # Apply universal formatting
    return chart.properties(
        title=f"Distribution Shift: {feature}", width=600, height=400
    )


@app.cell
def _(
    BINARY_FEATURES,
    CONT_FEATURES,
    FEATURES,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    ds_test_raw: pl.DataFrame,
    ds_train_raw: pl.DataFrame,
):
    # Intializing univariate drift

    ds_train = ds_train_raw
    ds_test = ds_test_raw

    calc = nml.UnivariateDriftCalculator(
        column_names=FEATURES,
        treat_as_numerical=CONT_FEATURES,
        treat_as_categorical=NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES,
        continuous_methods=["jensen_shannon"],
        categorical_methods=["jensen_shannon"],
        timestamp_column_name="id",
    )

    calc.fit(reference_data=ds_train.to_pandas())
    results = calc.calculate(data=ds_test.to_pandas())

    results.to_df()
    return ds_test, ds_train, results


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[0]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    age_drift_chart = plot_reference_analysis_dist(ds_train, ds_test, "Age")

    mo.ui.altair_chart(age_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Age"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Age"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Age"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[1]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    bp_drift_chart = plot_reference_analysis_dist(ds_train, ds_test, "BP")

    mo.ui.altair_chart(bp_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["BP"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["BP"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Age"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[2]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    cholesterol_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Cholesterol"
    )

    mo.ui.altair_chart(cholesterol_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Cholesterol"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Cholesterol"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Cholesterol"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[3]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    max_hr_drift_chart = plot_reference_analysis_dist(ds_train, ds_test, "Max HR")

    mo.ui.altair_chart(max_hr_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Max HR"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Max HR"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Max HR"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[4]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    drift_chart = plot_reference_analysis_dist(ds_train, ds_test, "ST depression")

    mo.ui.altair_chart(drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["ST depression"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["ST depression"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["ST depression"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[5]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    chest_pain_type_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Chest pain type", True
    )

    mo.ui.altair_chart(chest_pain_type_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Chest pain type"]).to_df()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Chest pain type"], methods=["jensen_shannon"]
    ).plot(kind="drift").show()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Chest pain type"], methods=["jensen_shannon"]
    ).plot(kind="distribution").show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[6]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    ekg_results_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "EKG results", True
    )

    mo.ui.altair_chart(ekg_results_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["EKG results"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["EKG results"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["EKG results"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[7]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    thallium_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Thallium", True
    )

    mo.ui.altair_chart(thallium_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Thallium"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Thallium"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Thallium"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[8]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    slope_of_st_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Slope of ST", True
    )

    mo.ui.altair_chart(slope_of_st_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Slope of ST"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Slope of ST"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Slope of ST"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[9]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    number_vessels_fluro_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Number of vessels fluro", True
    )

    mo.ui.altair_chart(number_vessels_fluro_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Number of vessels fluro"]).to_df()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Number of vessels fluro"], methods=["jensen_shannon"]
    ).plot(kind="drift").show()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Number of vessels fluro"], methods=["jensen_shannon"]
    ).plot(kind="distribution").show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[10]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    sex_drift_chart = plot_reference_analysis_dist(ds_train, ds_test, "Sex", True)

    mo.ui.altair_chart(sex_drift_chart)
    return


@app.cell(hide_code=True)
def _(results):
    results.filter(column_names=["Sex"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["Sex"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["Sex"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[11]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    fbs_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "FBS over 120", True
    )

    mo.ui.altair_chart(fbs_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["FBS over 120"]).to_df()
    return


@app.cell
def _(results):
    results.filter(column_names=["FBS over 120"], methods=["jensen_shannon"]).plot(
        kind="drift"
    ).show()
    return


@app.cell
def _(results):
    results.filter(column_names=["FBS over 120"], methods=["jensen_shannon"]).plot(
        kind="distribution"
    ).show()
    return


@app.cell(hide_code=True)
def _(FEATURES):
    mo.md(rf"""
    ### Feature: {FEATURES[12]}
    """)
    return


@app.cell
def _(ds_test, ds_train):
    exercise_angina_drift_chart = plot_reference_analysis_dist(
        ds_train, ds_test, "Exercise angina", True
    )

    mo.ui.altair_chart(exercise_angina_drift_chart)
    return


@app.cell
def _(results):
    results.filter(column_names=["Exercise angina"]).to_df()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Exercise angina"], methods=["jensen_shannon"]
    ).plot(kind="drift").show()
    return


@app.cell
def _(results):
    results.filter(
        column_names=["Exercise angina"], methods=["jensen_shannon"]
    ).plot(kind="distribution").show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusion

    Almost all the features are similarly distribution in train and test, except in if we view in chunk-wise we get a small drift (measured in jensen-shannon).
    """)
    return


if __name__ == "__main__":
    app.run()
