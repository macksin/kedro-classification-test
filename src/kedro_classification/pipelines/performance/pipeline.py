"""
This is a boilerplate pipeline 'performance'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import performance

def create_pipeline_srf(**kwargs):
    return Pipeline([
        node(
            performance, 
            inputs=["X_test", "y_test", "model"], 
            outputs="my_model_metrics")
    ])

def create_pipeline_cat(**kwargs):
    return Pipeline([
        node(
            performance, 
            inputs=["X_test", "y_test", "catboost.model"], 
            outputs="my_model_metrics")
    ])
