"""
This is a boilerplate pipeline 'pca'
generated using Kedro 0.17.6
"""

from sklearn.decomposition import PCA
from pandas import DataFrame

def pca_train(
    train_x: DataFrame, n_components: int 
) -> PCA:
    """Train a PCA process."""
    pca = PCA(n_components=n_components)
    pca.fit(train_x)
    return pca

def pca_use(
    input_x: DataFrame, pca: PCA
) -> DataFrame:
    return pca.transform(input_x)