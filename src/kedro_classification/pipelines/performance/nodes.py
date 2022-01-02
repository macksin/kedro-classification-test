"""
This is a boilerplate pipeline 'performance'
generated using Kedro 0.17.6
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from pandas import DataFrame, Series
from logging import getLogger
from typing import Dict


def performance(
    test_x: DataFrame, test_y: Series, model: RandomForestClassifier
) -> Dict:
    """Modelo simples Random Forest"""
    y_proba = model.predict_proba(test_x)[:, 1]
    # y_pred = model.predict(test_x)
    # sc = classification_report(test_y, y_pred)
    sc = roc_auc_score(test_y, y_proba)

    # log
    log = getLogger(__name__)
    log.info("\n\tTESTING ROC_AUC: %.4f" % sc)
    return {
        'roc_auc': sc
    }
