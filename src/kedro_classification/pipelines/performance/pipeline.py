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
            outputs="srf_results_test")
    ])
