"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro_classification.pipelines import create_test_dataset as cd
from kedro_classification.pipelines import test_splitter as ts
from kedro_classification.pipelines import simple_random_forest as srf
from kedro_classification.pipelines import performance as pf
from kedro_classification.pipelines import scoring as sc
from kedro_classification.pipelines import pca as pca


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
    performance_pipeline_catboost = pf.create_pipeline_cat()

    model_linear_pipe = srf.create_pipeline_linear()
    model_svm_pipe = srf.create_pipeline_svm()
    model_naive_pipe = srf.create_pipeline_naive()
    model_linear_pca_pipe = srf.create_pipeline_linear_pca()

    perf_linear_pipe = pf.create_pipeline_linear()
    perf_svm_pipe = pf.create_pipeline_svm()
    perf_naive_pipe = pf.create_pipeline_naive()
    perf_linear_pca_pipe = pf.create_pipeline_linear_pca()

    pca_train_pipe = pca.create_pipeline_train()
    pca_use_pipe = pca.create_pipeline_use()
    pca_use_test_pipe = pca.create_pipeline_use_test()

    return {
        "__default__": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_simple_rf_pipeline +\
            performance_pipeline +\
            scoring_pipeline
        ]),
        "Catboost": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            performance_pipeline_catboost +\
            cat_train +\
            cat_score   
        ]),

        "Linear": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_linear_pipe +\
            perf_linear_pipe +\
            scoring_pipeline
        ]),

        "Linear_PCA": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            pca_train_pipe +\
            pca_use_pipe +\
            pca_use_test_pipe +\
            model_linear_pca_pipe +\

            perf_linear_pca_pipe
        ]),

        "SVM": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_svm_pipe +\
            perf_svm_pipe +\
            scoring_pipeline
        ]),

        "Naive": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_naive_pipe +\
            perf_naive_pipe +\
            scoring_pipeline
        ]),

        "RandomForest": Pipeline([
            create_dataset_pipeline +\
            create_testing_pipeline +\
            model_simple_rf_pipeline +\
            performance_pipeline +\
            scoring_pipeline
        ]),
        
        "Create Dataset": create_dataset_pipeline,
        "Create Testing": create_testing_pipeline,
        "Simple Random Forest": model_simple_rf_pipeline,
        "Performance": performance_pipeline,
        "Scoring": scoring_pipeline
    }
