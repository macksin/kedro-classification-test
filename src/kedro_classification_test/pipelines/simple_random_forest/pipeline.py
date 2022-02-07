"""
This is a boilerplate pipeline 'simple_random_forest'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import random_forest_model, different_models

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            random_forest_model, 
            inputs=["X_train", "y_train", "params:cv_splits"], 
            outputs=["model", "srf_results_train"])
    ])


various_models_pipeline = Pipeline([
    node(
        different_models,
        inputs=["X_train", "y_train", "params:linear"],
        outputs="model",
        name="treinando_modelo"
    )
])

def create_pipeline_linear(**kwargs):
    return pipeline(
                various_models_pipeline,
                inputs=["X_train", "y_train"],
                namespace='linear',
                parameters={"params:linear":"params:linear"}
        )

def create_pipeline_svm(**kwargs):
    return pipeline(
                various_models_pipeline,
                inputs=["X_train", "y_train"],
                namespace='svm',
                parameters={"params:linear":"params:svm"}
        )

def create_pipeline_naive(**kwargs):
    return pipeline(
                various_models_pipeline,
                inputs=["X_train", "y_train"],
                namespace='naive',
                parameters={"params:linear":"params:naive"}
        )

def create_pipeline_linear_pca(**kwargs):
    return pipeline(
                various_models_pipeline,
                #inputs=["pca.X_train", "y_train"],
                inputs= {"X_train": "pca.X_train", "y_train": 'y_train'},
                namespace='linear',
                parameters={"params:linear":"params:linear"}
        )
