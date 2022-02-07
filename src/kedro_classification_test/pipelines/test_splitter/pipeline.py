"""
This is a boilerplate pipeline 'test_splitter'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import train_test_splitter


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            train_test_splitter, 
            inputs=[
                "raw_data", 
                "params:random_state", 
                "params:test_size"], 
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="train_test_split")
    ])
