import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

class Perceptron(BaseEstimator, ClassifierMixin, object):
    """Perceptron classifier.
    """

    def __init__(self, eta=0.001, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Traning vectors.
        y : array-like, shape = [n_examples]


        Returns
        -------
        self: object

        """
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=1e-2,
                            size = 1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return the class label."""
        return np.where(self.net_input(X) >= 0.5, 1, 0)

    def predict_proba(self, X):
        prob = self.net_input(X)
        return np.vstack((1-prob, prob)).T


if __name__ == '__main__':

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

    X, y = load_iris(return_X_y=True)
    X = X[:100]
    y = y[:100]

    # plotting
    plt.scatter(X[:50, 0], X[:50, 1], color='red', label='class=0')
    plt.scatter(X[50:, 0], X[50:, 1], color='red', label='class=0')
    plt.savefig('fig.png')

    model = Perceptron(n_iter=1000, eta=0.01)
    #model = LogisticRegression()
    model.fit(X, y)
    print(model.predict(X))
    print(model.predict_proba(X))
    print(model.predict_proba(X).shape)