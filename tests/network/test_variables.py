import numpy as np
import unittest

import mem.network.variables


class GaussianState(unittest.TestCase):

    def test_init(self):
        mean = np.matrix(np.zeros((2, 1)))
        variance = np.matrix(np.identity(2))
        state = mem.network.variables.GaussianState(mean, variance)

    def test_partition(self):

        # Test normal behavior.
        mean = np.matrix(np.zeros((2, 1)))
        variance = np.matrix(np.identity(2))
        state = mem.network.variables.GaussianState(mean, variance)

        lagrange = np.matrix(np.ones((2, 1)))
        partition, jacobian = state.partition(lagrange)

        self.assertAlmostEqual(partition, np.exp(1))
        np.testing.assert_array_almost_equal(jacobian, np.exp([[1], [1]]))

        # Inconsistent dimensions should raise an exception.
        mean = np.matrix(np.zeros((3, 1)))
        variance = np.matrix(np.identity(2))
        self.assertRaises(
            ValueError,
            mem.network.variables.GaussianState,
            mean, variance)

        # The variance matrix must be square.
        mean = np.matrix(np.zeros((2, 1)))
        variance = np.matrix(np.eye(2, 3))
        self.assertRaises(
            ValueError,
            mem.network.variables.GaussianState,
            mean, variance)

        # A variance matrix that is not positive semi-definite should raise an
        # exception.
        variance[0, 0] = -1
        self.assertRaises(
            ValueError,
            mem.network.variables.GaussianState,
            mean, variance)

    def test_projector(self):

        # Test normal behavior.
        mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        state = mem.network.variables.GaussianState(mean, variance)

        projector = np.matrix([[1, -1], [1, 1]])
        projected_state = state.project(projector)

        np.testing.assert_array_almost_equal(
            projected_state.mean, np.matrix([[0], [2]]))
        np.testing.assert_array_almost_equal(
            projected_state.variance, np.matrix([[2, 0], [0, 2]]))
