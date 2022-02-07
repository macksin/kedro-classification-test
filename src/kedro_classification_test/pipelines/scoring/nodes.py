"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.17.6
"""

from pandas import DataFrame


def score(
    data: DataFrame, model, y_true
) -> DataFrame:
    proba = model.predict_proba(data)[:, 1]
    pred = model.predict(data)
    data = data.copy()
    data['prob'] = proba
    data['pred'] = pred
    data['target'] = y_true.values
    return data