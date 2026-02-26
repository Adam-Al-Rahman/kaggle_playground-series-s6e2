from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_labels, encode_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_labels,
                inputs=["raw_ingestion.ds_heart_disease", "params:target_col"],
                outputs="ds_train_labels_processed",
                name="preprocess_labels_node",
            ),
            node(
                func=encode_features,
                inputs=[
                    "ds_train_labels_processed",
                    "raw_ingestion.ds_heart_disease_test",
                    "params:nominal_features",
                    "params:binary_features",
                    "params:target_col",
                ],
                outputs=[
                    "processed.ds_heart_disease_train",
                    "processed.ds_heart_disease_test",
                ],
                name="encode_features_node",
            ),
        ]
    )
