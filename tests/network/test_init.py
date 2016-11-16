
import unittest

import numpy as np

import mem.network


class Bayesian(unittest.TestCase):

    def test_init(self):

        # Test normal behavior.
        network = mem.network.Bayesian()

        mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        state = mem.network.variables.GaussianState(mean, variance)
        cluster = mem.network.variables.Cluster([0, 1], [state])
        table = mem.network.tables.ConditionalProbability(cluster, [1.0])

        network.add_variable(cluster)
        network.add_table(table)

        # The variables of a table must be part of the network.
        network = mem.network.Bayesian()
        self.assertRaises(ValueError, network.add_table, table)

        # The variables must appear in the same order.
        mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        other_state = mem.network.variables.GaussianState(mean, variance)
        other_cluster = mem.network.variables.Cluster(
            [0, 1], [state, other_state])
        other_table = mem.network.tables.ConditionalProbability(
            [cluster, other_cluster], [0.50, 0.50])
        network = mem.network.Bayesian()
        network.add_variable(other_cluster)
        network.add_variable(cluster)
        self.assertRaises(ValueError, network.add_table, other_table)

        # The same variable cannot be added twice.
        network = mem.network.Bayesian()
        network.add_variable(cluster)
        self.assertRaises(ValueError, network.add_variable, cluster)
