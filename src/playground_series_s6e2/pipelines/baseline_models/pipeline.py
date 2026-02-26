from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    baseline_model_v1_submission,
    create_submission,
    evaluate_logit_model,
    split_data,
    train_logit_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_v0 = pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "processed.ds_heart_disease_train",
                    "params:baseline_models",
                ],
                outputs=["train_data_v0", "val_data_v0"],
                name="split_data_v0_node",
            ),
            node(
                func=train_logit_model,
                inputs=["train_data_v0", "params:baseline_models"],
                outputs="baseline_models.model_v0",
                name="train_logit_model_v0_node",
            ),
            node(
                func=evaluate_logit_model,
                inputs=[
                    "baseline_models.model_v0",
                    "val_data_v0",
                    "params:baseline_models",
                ],
                outputs=[
                    "baseline_models.model_v0_metrics",
                    "baseline_models.model_v0_summary",
                    "baseline_models.model_v0_coefficients",
                    "baseline_models.model_v0_metrics_df",
                ],
                name="evaluate_logit_model_v0_node",
            ),
            node(
                func=create_submission,
                inputs=[
                    "baseline_models.model_v0",
                    "processed.ds_heart_disease_test",
                    "params:baseline_models",
                ],
                outputs="baseline_models.model_v0_submission",
                name="create_baseline_model_v0_submission_node",
            ),
        ]
    )

    pipeline_v1 = pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "feature_engineered.ds_important_features_train",
                    "params:baseline_models_v1",
                ],
                outputs=["train_data_v1", "val_data_v1"],
                name="split_data_v1_node",
            ),
            node(
                func=train_logit_model,
                inputs=["train_data_v1", "params:baseline_models_v1"],
                outputs="baseline_models.model_v1",
                name="train_logit_model_v1_node",
            ),
            node(
                func=evaluate_logit_model,
                inputs=[
                    "baseline_models.model_v1",
                    "val_data_v1",
                    "params:baseline_models_v1",
                ],
                outputs=[
                    "baseline_models.model_v1_metrics",
                    "baseline_models.model_v1_summary",
                    "baseline_models.model_v1_coefficients",
                    "baseline_models.model_v1_metrics_df",
                ],
                name="evaluate_logit_model_v1_node",
            ),
            node(
                func=baseline_model_v1_submission,
                inputs=[
                    "baseline_models.model_v1",
                    "feature_engineered.ds_important_features_test",
                    "params:baseline_models_v1",
                ],
                outputs="baseline_models.model_v1_submission",
                name="create_baseline_model_v1_submission_node",
            ),
        ]
    )

    return pipeline_v0 + pipeline_v1
