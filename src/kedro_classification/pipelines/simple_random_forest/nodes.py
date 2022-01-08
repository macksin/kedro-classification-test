"""
This is a boilerplate pipeline 'simple_random_forest'
generated using Kedro 0.17.6
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from pandas import DataFrame, Series
from logging import getLogger
from numpy import mean, std
from typing import Dict, List

from catboost import CatBoostClassifier, Pool


def random_forest_model(
    train_x: DataFrame, train_y: Series, cv_splits: int
):
    """Modelo simples Random Forest"""
    model = RandomForestClassifier(max_depth=3)

    cv_scores = cross_val_score(
        model, 
        X=train_x, 
        y=train_y, 
        scoring="roc_auc", 
        cv=cv_splits)

    model.fit(train_x, train_y)
    # results
    log = getLogger(__name__)
    log.info("\n\tROC AUC SCORE: %.4f%% +/- %.4f" % (
        mean(cv_scores), std(cv_scores)
    ))

    return model, {
        'roc_auc': mean(cv_scores),
        'std(roc_auc)': std(cv_scores)
    }


def catboostc(
    train_x: DataFrame, train_y: Series
):

    model = CatBoostClassifier(iterations=150,
                               depth=10,
                               learning_rate=1,
                               loss_function='Logloss',
                               verbose=False)
    # train the model
    model.fit(train_x, train_y)

    return model