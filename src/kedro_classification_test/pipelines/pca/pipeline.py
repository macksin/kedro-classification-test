"""
This is a boilerplate pipeline 'pca'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import pca_train, pca_use


pca_train_pipeline = Pipeline([
    node(
        pca_train, 
        inputs=["X_train", "params:n_components"], 
        outputs="pca_model",
        name="fitting_pca_algorithm"
    )
])

pca_use_pipeline = Pipeline([
    node(
        pca_use, 
        inputs=["X_train", "pca_model"], 
        outputs="pca.X_train",
    )
])

def create_pipeline_train(**kwargs):
    return pca_train_pipeline

def create_pipeline_use(**kwargs):
    return pca_use_pipeline

def create_pipeline_use_test(**kwargs):
    return pipeline(
        pca_use_pipeline,
        inputs={"X_train": "X_test", "pca_model": "pca_model"},
        outputs={"pca.X_train": "pca.X_test"}
    )