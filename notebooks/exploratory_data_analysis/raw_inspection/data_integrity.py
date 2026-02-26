import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Integrity & Quality Check
    """)
    return


@app.function
def DATA_INTEGRITY_INSIGHTS():
    INSIGHTS = {
        "data integrity": {
            "missing data": "zero missing in each rows and columns",
            "duplication": "zero duplication",
            "data type consistent": "each columns has correct datatype according to the domain",
        },
    }

    return INSIGHTS


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

        ds_train: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease")
        ds_test: pl.DataFrame = catalog.load("raw_ingestion.ds_heart_disease_test")
    return ds_test, ds_train


@app.cell
def _(ds_test: pl.DataFrame, ds_train: pl.DataFrame):
    ds_train.estimated_size(unit="mb"), ds_test.estimated_size(unit="mb")
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.head()
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.glimpse()
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test.glimpse()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is there missing data?
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.null_count()
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test.null_count()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusions

    - No missing values
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Are there duplicates or near-duplicates?
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.filter(ds_train.is_duplicated())
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test.filter(ds_test.is_duplicated())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusions

    - No duplicates
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Are data types consistent with domain
    """)
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.glimpse()
    return


@app.cell
def _(ds_train: pl.DataFrame):
    ds_train.describe()
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test.glimpse()
    return


@app.cell
def _(ds_test: pl.DataFrame):
    ds_test.describe()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conclusion

    - Yes, each columns has correct datatype according to the domain
    """)
    return


if __name__ == "__main__":
    app.run()
