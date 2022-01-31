"""
Performance node where we will develop all the performance functions.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from pandas import DataFrame, Series
from logging import getLogger
from typing import Dict, Union, List


def performance(
    test_x: DataFrame, test_y: Series, model: RandomForestClassifier
):
    """Measure the test data with the metrics:
        - F1 Score
        - AUROC"""
    y_proba = model.predict_proba(test_x)[:, 1]
    y_pred = model.predict(test_x)
    sc = roc_auc_score(test_y, y_proba)

    # f1 score
    f1 = f1_score(test_y, y_pred)

    # log
    log = getLogger(__name__)
    log.info("Testing roc_auc: %.4f" % sc)

    return {
        "roc_auc": {"value": float(sc), "step": 1},
        "f1_score": {"value": float(f1), "step": 1}
    }
