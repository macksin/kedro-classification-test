"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import score


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            score,
            inputs=["X_test", "model", "y_test"],
            outputs="scored_test"
        )
    ])



