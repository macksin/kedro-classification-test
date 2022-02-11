"""
This is a boilerplate pipeline 'create_test_dataset'
generated using Kedro 0.17.6
"""

from pandas import DataFrame
from sklearn.datasets import make_classification
from numpy import bincount, mean
from logging import getLogger

def create_dataset(random_state: int) -> DataFrame:
    """Make inbalanced classification dataset."""
    args = dict(
        n_samples=1000,
        n_features=50,
        n_informative=10, 
        n_redundant=10, 
        n_repeated=5, 
        n_classes=2, 
        n_clusters_per_class=6,
        weights=(0.70,),
        random_state=random_state
    )

    data = make_classification(**args)

    f = DataFrame(data[0])
    f.columns = [f"col_{i}" for i in range(50)]
    f['target'] = data[1]

    log = getLogger(__name__)
    log.info(
        "\n\tBincount: %s\n\tTarget Mean: %.2f\n\tShape: %s" % (
            bincount(data[1]), mean(data[1]), f.shape
        )
    )
    
    return f

