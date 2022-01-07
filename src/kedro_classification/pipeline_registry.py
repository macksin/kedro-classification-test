"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro_classification.pipelines import create_test_dataset as cd
from kedro_classification.pipelines import test_splitter as ts
from kedro_classification.pipelines import simple_random_forest as srf
from kedro_classification.pipelines import performance as pf
from kedro_classification.pipelines import scoring as sc


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    
    create_dataset_pipeline = cd.create_pipeline()
    create_testing_pipeline = ts.create_pipeline()
    model_simple_rf_pipeline = srf.create_pipeline()
    performance_pipeline = pf.create_pipeline_srf()
    scoring_pipeline = sc.create_pipeline()
    cat_train = srf.create_pipeline_catboost()
    cat_score = sc.create_pipeline_cat()

    return {
        "__default__": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_simple_rf_pipeline +\
            performance_pipeline +\
            scoring_pipeline +\
            cat_train +\
            cat_score
        ]),
        "Create Dataset": create_dataset_pipeline,
        "Create Testing": create_testing_pipeline,
        "Simple Random Forest": model_simple_rf_pipeline,
        "Performance": performance_pipeline,
        "Scoring": scoring_pipeline
    }
