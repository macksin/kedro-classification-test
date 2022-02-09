"""
This is a boilerplate pipeline 'learning_curves'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import learning_curves


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            learning_curves,
            inputs=["X_train", "y_train"],
            outputs="learning_curve"
        )
    ])
