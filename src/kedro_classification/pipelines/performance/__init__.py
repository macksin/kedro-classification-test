"""
This is a boilerplate pipeline 'performance'
generated using Kedro 0.17.6
"""

from .pipeline import (
    create_pipeline_srf, 
    performance, 
    create_pipeline_cat, 
    create_pipeline_linear, 
    create_pipeline_svm, 
    create_pipeline_naive, 
    create_pipeline_linear_pca
    )

__all__ = ["create_pipeline"]
