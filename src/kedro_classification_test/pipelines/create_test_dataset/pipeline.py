"""
This is a boilerplate pipeline 'create_test_dataset'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import create_dataset


def create_pipeline(**kwargs):
    return Pipeline([
        node(create_dataset, 
             inputs="params:random_state", 
             outputs="raw_data", 
             name="create_imbalanced_dataset")
    ])
