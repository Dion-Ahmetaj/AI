import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter

    def fit(self, X, y):
        m, n = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize log-probabilities
        self.log_prior = np.zeros(n_classes)
        self.log_likelihood = np.zeros((n_classes, n))

        # Compute log-probabilities
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.log_prior[idx] = np.log(X_cls.shape[0] / m)
            self.log_likelihood[idx] = np.log((X_cls.sum(axis=0) + self.alpha) / (X_cls.shape[0] + 2 * self.alpha))

    def predict(self, X):
        return np.argmax(self._predict_log_proba(X), axis=1)

    def _predict_log_proba(self, X):
        return X @ self.log_likelihood.T + self.log_prior

    def predict_proba(self, X):
        return np.exp(self._predict_log_proba(X))