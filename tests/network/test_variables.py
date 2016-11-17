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


class Cluster(unittest.TestCase):

    def test_init(self):

        # Test normal behavior.
        off_mean = np.matrix(np.zeros((2, 1)))
        on_mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))

        on_state = mem.network.variables.GaussianState(on_mean, variance)
        off_state = mem.network.variables.GaussianState(off_mean, variance)

        cluster = mem.network.variables.Cluster([0, 1], [off_state, on_state])

        self.assertEqual(cluster.nb_states, 2)
        self.assertEqual(cluster.nb_sources, 2)
        self.assertEqual(cluster.states[0], off_state)
        self.assertEqual(cluster.states[1], on_state)

        # The number of sources of the cluster should match the number of
        # sources of the states.
        self.assertRaises(
            ValueError,
            mem.network.variables.Cluster,
            [0], [off_state, on_state])


class ZeroOneCluster(unittest.TestCase):

    def test_init(self):
        cluster = mem.network.variables.ZeroOneCluster([0, 1])
        self.assertEqual(cluster.nb_states, 2)
        self.assertEqual(cluster.nb_sources, 2)
