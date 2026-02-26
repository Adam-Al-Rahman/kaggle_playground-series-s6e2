from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_domain_features, scale_features, select_important_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_domain_features,
                inputs=[
                    "processed.ds_heart_disease_train",
                    "processed.ds_heart_disease_test",
                ],
                outputs=[
                    "feature_engineered.ds_heart_disease_train",
                    "feature_engineered.ds_heart_disease_test",
                ],
                name="create_domain_features_node",
            ),
            node(
                func=scale_features,
                inputs=[
                    "feature_engineered.ds_heart_disease_train",
                    "feature_engineered.ds_heart_disease_test",
                    "params:target_col",
                    "params:features_to_scale",
                ],
                outputs=[
                    "feature_engineered.ds_heart_disease_train_scaled",
                    "feature_engineered.ds_heart_disease_test_scaled",
                ],
                name="scale_features_node",
            ),
            node(
                func=select_important_features,
                inputs=[
                    "feature_engineered.ds_heart_disease_train_scaled",
                    "feature_engineered.ds_heart_disease_test_scaled",
                    "params:important_features",
                    "params:target_col",
                ],
                outputs=[
                    "feature_engineered.ds_important_features_train",
                    "feature_engineered.ds_important_features_test",
                ],
                name="select_important_features_node",
            ),
        ]
    )
