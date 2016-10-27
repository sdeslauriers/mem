import numpy as np


class GaussianState(object):
    def __init__(self, mean, variance):
        """Sources in a Gaussian state"""

        if len(mean) != len(variance):
            raise ValueError(
                'The size of the mean and variance is not consistent. '
                'The mean has a size of {} and the variance has '
                'a size of {}.'
                .format(len(mean), len(variance)))

        if len(variance) != len(variance.T):
            raise ValueError('The variance matrix is not square.')

        if not np.all(np.linalg.eigvals(variance) >= 0):
            raise ValueError(
                'The variance matrix is not positive semi-definite.')

        self._mean = mean
        self._variance = variance

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def partition(self, lagrange):
        """Evaluate the states partition function and its jacobian"""
        temp = self.variance @ lagrange
        value = np.exp(self.mean.T @ lagrange + 0.5 * lagrange.T @ temp)
        value = value.item(0)
        jacobian = value * (self.mean + temp)
        return value, jacobian

    def project(self, projector):
        """Project the state to another space"""
        return GaussianState(
            projector @ self.mean,
            projector @ self.variance @ projector.T)
