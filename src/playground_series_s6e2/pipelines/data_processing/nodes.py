import category_encoders as ce
import pandas as pd
import polars as pl


def preprocess_labels(ds_train: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Transforms target labels from string to integer (Absence: 0, Presence: 1)."""
    return ds_train.with_columns(
        pl.col(target_col).replace({"Absence": 0, "Presence": 1}).cast(pl.Int8)
    )


def encode_features(
    ds_train: pl.DataFrame,
    ds_test: pl.DataFrame,
    nominal_features: list[str],
    binary_features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolidated encoding using category encoders:
    - WOE Encoding for the first nominal feature (Chest pain type).
    - Binary Encoding for all binary features.
    - Passthrough for all other features.
    """
    # Prepare training data (Pandas)
    train_pd = ds_train.to_pandas()
    y_train = train_pd[target_col]
    X_train = train_pd.drop(columns=[target_col])

    # Prepare test data (Pandas)
    X_test = ds_test.to_pandas()

    # WOE Encoding: "Chest pain type"
    woe_encoder = ce.WOEEncoder(cols=[nominal_features[0]])
    ds_train_processed = woe_encoder.fit_transform(X_train, y_train)
    ds_test_processed = woe_encoder.transform(X_test)

    # Add target back
    ds_train_processed[target_col] = y_train.values

    return ds_train_processed, ds_test_processed
