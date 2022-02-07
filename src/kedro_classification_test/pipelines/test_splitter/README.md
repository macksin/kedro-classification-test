# Pipeline test_splitter

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.17.6`.

## Overview

Generate train and test data, separated by X (features) and y (targets), we reset the indexes the format `feather` does not work with non-consective indices.

## Pipeline inputs

data: DataFrame
	The dataframe, with the target named `target`.
random_state: int
	The random state that will be used with `train_test_split`.
test_size: float
	From 0 to 1, the fraction of samples of the test data.

## Pipeline outputs

A list that contains X_train, X_test, y_train, y_test data in DataFrame type.

