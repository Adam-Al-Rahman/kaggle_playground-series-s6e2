import pandas as pd
import polars as pl


def create_domain_features(
    ds_train: pd.DataFrame, ds_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates domain-specific features for heart disease prediction.
    - HR_Deficit: (220 - Age) - Max HR
    - High_Risk_Ischemia: (ST depression > 1.0) & (Exercise angina == 1.0)
    - Severe_Blockage_Age_Adjusted: Number of vessels fluro / Age

    Args:
        ds_train: Training data.
        ds_test: Test data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Dataframes with domain features.
    """

    def _create_features(df: pd.DataFrame) -> pd.DataFrame:
        pl_df = pl.from_pandas(df)

        pl_df = pl_df.with_columns(
            HR_Deficit=(220 - pl.col("Age")) - pl.col("Max HR"),
            High_Risk_Ischemia=(
                (pl.col("ST depression") > 1.0) & (pl.col("Exercise angina") == 1.0)
            ).cast(pl.Int32),
            Severe_Blockage_Age_Adjusted=(
                pl.col("Number of vessels fluro") / pl.col("Age")
            ),
        )

        columns_to_drop = [
            "Age",
            "Max HR",
            "ST depression",
            "Exercise angina",
            "Number of vessels fluro",
        ]

        pl_df = pl_df.drop([c for c in columns_to_drop if c in pl_df.columns])

        return pl_df.to_pandas()

    return _create_features(ds_train), _create_features(ds_test)


def scale_features(
    ds_train: pd.DataFrame,
    ds_test: pd.DataFrame,
    target_col: str,
    cont_features_to_scale: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales continuous features using StandardScaler.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # Identify non-feature columns
    non_feature_cols = [target_col]
    if "id" in ds_train.columns:
        non_feature_cols.append("id")

    # Split features and non-features
    X_train = ds_train.drop(columns=non_feature_cols)
    X_test = ds_test.drop(columns=["id"]) if "id" in ds_test.columns else ds_test

    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), cont_features_to_scale),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrame
    ds_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    ds_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Add back target and id
    ds_train_scaled[target_col] = ds_train[target_col].values
    if "id" in ds_train.columns:
        ds_train_scaled["id"] = ds_train["id"].values

    if "id" in ds_test.columns:
        ds_test_scaled["id"] = ds_test["id"].values

    return ds_train_scaled, ds_test_scaled


def select_important_features(
    ds_train: pd.DataFrame,
    ds_test: pd.DataFrame,
    important_features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects only the most important features.
    """
    # Filter columns that actually exist in the dataframe
    train_cols = [c for c in important_features if c in ds_train.columns] + [target_col]
    if "id" in ds_train.columns:
        train_cols.append("id")

    test_cols = [c for c in important_features if c in ds_test.columns]
    if "id" in ds_test.columns:
        test_cols.append("id")

    return ds_train[train_cols], ds_test[test_cols]
