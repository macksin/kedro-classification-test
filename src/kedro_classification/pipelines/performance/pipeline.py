"""
This is a boilerplate pipeline 'performance'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import performance

performance_node = node(
    performance, 
    inputs=["X_test", "y_test", "model"], 
    outputs="my_model_metrics")


def create_pipeline_srf(**kwargs):
    return Pipeline([
        performance_node
    ])

def create_pipeline_cat(**kwargs):
    return Pipeline([
        node(
            performance, 
            inputs=["X_test", "y_test", "catboost.model"], 
            outputs="my_model_metrics")
    ])

performance_models_pipeline = Pipeline([
    node(
        performance,
        inputs=["X_test", "y_test", "model"],
        outputs="my_model_metrics"
    )
])

def create_pipeline_linear(**kwargs):
    return pipeline(
        performance_models_pipeline,
        inputs=["X_test", "y_test"],
        outputs="my_model_metrics",
        namespace="linear"
    )

def create_pipeline_svm(**kwargs):
    return pipeline(
        performance_models_pipeline,
        inputs=["X_test", "y_test"],
        outputs="my_model_metrics",
        namespace="svm"
    )

def create_pipeline_naive(**kwargs):
    return pipeline(
        performance_models_pipeline,
        inputs=["X_test", "y_test"],
        outputs="my_model_metrics",
        namespace="naive"
    )