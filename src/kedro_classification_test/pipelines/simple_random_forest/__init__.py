"""
This is a boilerplate pipeline 'simple_random_forest'
generated using Kedro 0.17.6
"""

from .pipeline import (
    create_pipeline, 
    create_pipeline_linear, 
    create_pipeline_naive, 
    create_pipeline_svm, 
    create_pipeline_linear_pca)

__all__ = ["create_pipeline"]
