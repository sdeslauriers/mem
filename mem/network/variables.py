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
    def nb_sources(self):
        return len(self._mean)

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


class Cluster(object):
    def __init__(self, source_numbers, states):
        """A variable that represent a cluster of sources"""

        # The number of source numbers must match the number of source in the
        # states.
        for i, state in enumerate(states):
            if state.nb_sources != len(source_numbers):
                raise ValueError(
                    'The number of sources of the state {} does not match '
                    'the number of sources numbers ({} != {}).'
                    .format(i, state.nb_sources, len(source_numbers)))

        self._source_numbers = source_numbers
        self._states = states

    @property
    def nb_sources(self):
        return len(self._source_numbers)

    @property
    def nb_states(self):
        return len(self._states)

    @property
    def states(self):
        return self._states


class Connection(object):
    """A variable that represents a connection between clusters"""
    pass


class ZeroOneCluster(Cluster):
    def __init__(self, source_numbers):
        """A cluster with sources intensities in [0, 1]"""
        nb_sources = len(source_numbers)
        variance = np.matrix(np.zeros((nb_sources, nb_sources)))
        zero_state = GaussianState(
            np.matrix(np.zeros((nb_sources, 1))), variance)
        one_state = GaussianState(
            np.matrix(np.ones((nb_sources, 1))), variance)
        super().__init__(source_numbers, [zero_state, one_state])
