"""
This is a boilerplate pipeline 'test_splitter'
generated using Kedro 0.17.6
"""

from sklearn.model_selection import train_test_split
from pandas import DataFrame
from typing import List

def train_test_splitter(
    data: DataFrame,
    random_state: int,
    test_size: float
) -> List:
    """Divide between Training and Testing data."""
    X = data.drop(columns="target").copy()
    Y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    return [
        X_train.reset_index(drop=True), 
        X_test.reset_index(drop=True), 
        y_train.reset_index(drop=True), 
        y_test.reset_index(drop=True)]