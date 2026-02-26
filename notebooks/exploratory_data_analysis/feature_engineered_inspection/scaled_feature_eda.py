import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd

    import polars.selectors as cs

    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaled Feature EDA
    """)
    return


@app.function
def SCALED_FEATURE_EDA_INSIGHTS():
    INSIGHTS = {
        "important features": ["HR_Deficit", "Thallium", "Chest pain type"]
    }

    return INSIGHTS


@app.cell
def _():
    from exploratory_data_analysis.raw_inspection.bivariate_analysis import (
        BivariateAnalyzer,
    )

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

        ds_train: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_train_scaled"
        )
        ds_test: pl.DataFrame = catalog.load(
            "feature_engineered.ds_heart_disease_test_scaled"
        )
    return (ds_train,)


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.columns.to_list()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Principal Component Analysis
    """)
    return


@app.function
def run_scaled_space_eda(
    ds_train_scaled: pl.DataFrame, target_col: str
) -> mo.Html:
    """
    Executes post-scaling Exploratory Data Analysis focusing on geometry,
    intrinsic dimensionality, feature influence (loadings), and multivariate outliers.
    """
    features = [col for col in ds_train_scaled.columns if col != target_col]
    X_scaled = ds_train_scaled.select(features).to_numpy()
    y = ds_train_scaled.select(target_col).to_series().to_numpy()

    # 1. PCA for Intrinsic Dimensionality (Scree Plot) and Global Structure
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)

    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # --- SCREE PLOT ---
    n_components_plot = min(len(explained_var), 20)
    scree_df = pl.DataFrame(
        {
            "Component": np.arange(1, n_components_plot + 1),
            "Explained_Variance": explained_var[:n_components_plot],
            "Cumulative_Variance": cumulative_var[:n_components_plot],
        }
    )

    base = alt.Chart(scree_df.to_pandas()).encode(
        x=alt.X("Component:O", title="Principal Component")
    )
    bar = base.mark_bar(opacity=0.6, color="#4c78a8").encode(
        y=alt.Y("Explained_Variance:Q", title="Variance Explained")
    )
    line = base.mark_line(color="#e45756", point=True).encode(
        y=alt.Y(
            "Cumulative_Variance:Q",
            title="Cumulative Variance",
            scale=alt.Scale(domain=[0, 1.05]),
        )
    )
    scree_chart = (
        (bar + line)
        .resolve_scale(y="independent")
        .properties(
            title=f"Scree Plot (Top {n_components_plot} Components)",
            width=500,
            height=300,
        )
    )

    # --- 2D PROJECTION SCATTER PLOT ---
    sample_idx = np.random.choice(
        len(X_scaled), min(5000, len(X_scaled)), replace=False
    )
    pca_df = pl.DataFrame(
        {
            "PC1": X_pca_full[sample_idx, 0],
            "PC2": X_pca_full[sample_idx, 1],
            "Target": y[sample_idx],
        }
    )

    pca_scatter = (
        alt.Chart(pca_df.to_pandas())
        .mark_circle(size=45, opacity=0.7)
        .encode(
            x=alt.X(
                "PC1:Q", title=f"PC1 ({explained_var[0] * 100:.1f}% Variance)"
            ),
            y=alt.Y(
                "PC2:Q", title=f"PC2 ({explained_var[1] * 100:.1f}% Variance)"
            ),
            color=alt.Color("Target:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Target"],
        )
        .properties(
            title="Scaled Space Global Structure (PCA 2D Projection)",
            width=500,
            height=400,
        )
        .interactive()
    )

    # --- PCA LOADING CHART (FEATURE INFLUENCE) ---
    pc1_loadings = pca_full.components_[0, :]
    pc2_loadings = pca_full.components_[1, :]

    # Calculate vector magnitude to find the most influential features
    loading_magnitudes = np.sqrt(pc1_loadings**2 + pc2_loadings**2)

    loadings_df = pl.DataFrame(
        {
            "Feature": features,
            "PC1_Weight": pc1_loadings,
            "PC2_Weight": pc2_loadings,
            "Magnitude": loading_magnitudes,
        }
    ).sort("Magnitude", descending=True)

    # Highlight only the top 10 features to keep the plot readable
    top_loadings = loadings_df.head(10)

    base_loadings = (
        alt.Chart(loadings_df.to_pandas())
        .mark_circle(color="lightgray", size=40, opacity=0.5)
        .encode(
            x=alt.X("PC1_Weight:Q", title=f"PC1 Loading"),
            y=alt.Y("PC2_Weight:Q", title=f"PC2 Loading"),
            tooltip=["Feature", "PC1_Weight", "PC2_Weight"],
        )
    )

    highlight_loadings = (
        alt.Chart(top_loadings.to_pandas())
        .mark_circle(color="#d62728", size=70)
        .encode(
            x="PC1_Weight:Q",
            y="PC2_Weight:Q",
            tooltip=["Feature", "PC1_Weight", "PC2_Weight"],
        )
    )

    text_loadings = highlight_loadings.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        dy=-2,
        fontSize=11,
        fontWeight="bold",
        color="black",
    ).encode(text="Feature:N")

    # Add crosshairs for the origin (0,0)
    hline = (
        alt.Chart(pl.DataFrame({"y": [0]}).to_pandas())
        .mark_rule(color="black", strokeDash=[2, 2], opacity=0.3)
        .encode(y="y:Q")
    )
    vline = (
        alt.Chart(pl.DataFrame({"x": [0]}).to_pandas())
        .mark_rule(color="black", strokeDash=[2, 2], opacity=0.3)
        .encode(x="x:Q")
    )

    loading_chart = (
        hline + vline + base_loadings + highlight_loadings + text_loadings
    ).properties(
        title="PCA Loadings (Top 10 Feature Contributions)",
        width=800,
        height=400,
    )

    # 2. Multivariate Outlier Detection using Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    outlier_labels = iso.fit_predict(X_scaled)

    outlier_status = pl.Series(
        "Status",
        ["Anomaly" if label == -1 else "Normal" for label in outlier_labels],
    )
    outlier_summary = outlier_status.value_counts().sort(
        "count", descending=True
    )

    # 3. Scaled Feature Distribution Sanity Check (Boxplots)
    variances = np.var(X_scaled, axis=0)
    top_5_idx = np.argsort(variances)[-5:]
    top_5_features = [features[i] for i in top_5_idx]

    box_data = ds_train_scaled.select(top_5_features).unpivot(
        variable_name="Feature", value_name="Scaled_Value"
    )

    box_chart = (
        alt.Chart(box_data.to_pandas())
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("Feature:N", title="Top 5 High-Variance Features"),
            y=alt.Y("Scaled_Value:Q", title="Standardized Value"),
            color="Feature:N",
        )
        .properties(
            title="Distribution Sanity Check (Post-Scaling)",
            width=500,
            height=300,
        )
    )

    # Compile outputs using marimo
    return mo.vstack(
        [
            mo.md("### 1. Intrinsic Dimensionality (Scree Plot)"),
            mo.md(
                "Does our engineered feature space hold unique geometric information, or is it mathematically redundant?"
            ),
            mo.ui.altair_chart(scree_chart),
            mo.md("### 2. Feature Influence (PCA Loadings)"),
            mo.md(
                "Which original features are mathematically driving the variance in our top two components? *(Top 10 highlighted)*"
            ),
            mo.ui.altair_chart(loading_chart),
            mo.md("### 3. Global Structure & Target Separability"),
            mo.md(
                "In standard scaled space, do the target classes form distinct clusters along the axes of highest variance?"
            ),
            mo.ui.altair_chart(pca_scatter),
            mo.md("### 4. Multivariate Anomaly Detection"),
            mo.md(
                "Using Isolation Forests to detect points that break the multidimensional correlation structure."
            ),
            mo.ui.table(outlier_summary.to_dicts()),
            mo.md("### 5. Scaled Distribution Sanity Check"),
            mo.md(
                "Ensuring extreme values haven't compressed the normal operating range into a tiny band near zero."
            ),
            mo.ui.altair_chart(box_chart),
        ]
    )


@app.cell
def _(ds_train: pl.DataFrame):
    scaled_eda_report = run_scaled_space_eda(
        pl.from_pandas(ds_train).drop("id"), target_col="Heart Disease"
    )
    scaled_eda_report
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Variables with a correlation are groupeed together
    - Observations with similar overall profiles are clustered together
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    pl.from_pandas(ds_train).select(
        ["HR_Deficit", "Thallium", "Chest pain type", "Heart Disease"]
    ).write_csv("data/processed/02_feature/important_features.csv")
    return


if __name__ == "__main__":
    app.run()
