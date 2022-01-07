"""
This is a boilerplate pipeline 'simple_random_forest'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import random_forest_model, catboostc

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            random_forest_model, 
            inputs=["X_train", "y_train", "params:cv_splits"], 
            outputs=["model", "srf_results_train"])
    ])


def create_pipeline_catboost(**kwargs):
    return Pipeline([
        node(
            catboostc, 
            inputs=["X_train", "y_train"], 
            outputs="catboost.model") 
    ])
