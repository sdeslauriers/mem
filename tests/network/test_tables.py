
import unittest

import numpy as np

import mem.network.tables
import mem.network.variables


class Table(unittest.TestCase):

    def test_init(self):

        # Test normal behavior.
        off_mean = np.matrix(np.zeros((2, 1)))
        on_mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        on_state = mem.network.variables.GaussianState(on_mean, variance)
        off_state = mem.network.variables.GaussianState(off_mean, variance)
        cluster = mem.network.variables.Cluster([0, 1], [off_state, on_state])
        table = mem.network.tables.ConditionalProbability(cluster, [0.5, 0.5])

        self.assertEqual(table.variables[0], cluster)
        self.assertEqual(table.nb_states, 2)
        np.testing.assert_array_equal(table.probabilities, [0.5, 0.5])

        # A list of variables is also acceptable.
        other_cluster = mem.network.variables.Cluster(
            [0, 1], [off_state, on_state])
        table = mem.network.tables.ConditionalProbability(
            [cluster, other_cluster], [0.5, 0.2, 0.1, 0.2])

        # The number of probabilities must match the number of possible
        # states.
        self.assertRaises(
            ValueError,
            mem.network.tables.ConditionalProbability,
            cluster, [0.5, 0.4, 0.05, 0.05])
