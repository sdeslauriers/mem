
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


class Evidence(unittest.TestCase):

    def test_init(self):

        # Test normal behavior.
        off_mean = np.matrix(np.zeros((2, 1)))
        on_mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        on_state = mem.network.variables.GaussianState(on_mean, variance)
        off_state = mem.network.variables.GaussianState(off_mean, variance)
        cluster = mem.network.variables.Cluster([0, 1], [off_state, on_state])
        table = mem.network.tables.Evidence(cluster, [0.5, 0.5])

        self.assertEqual(table.variables[0], cluster)
        self.assertEqual(table.nb_states, 2)
        np.testing.assert_array_equal(table.probabilities, [0.5, 0.5])

        # The number of probabilities must match the number of states.
        self.assertRaises(
            ValueError,
            mem.network.tables.Evidence,
            cluster, [0.5, 0.3, 0.1, 0.1])

        # Evidence tables can only depend on a single variable.
        other_cluster = mem.network.variables.Cluster(
            [0, 1], [off_state, on_state])
        self.assertRaises(
            ValueError,
            mem.network.tables.Evidence,
            [cluster, other_cluster], [0.5, 0.2, 0.1, 0.2])

    def test_update(self):

        off_mean = np.matrix(np.zeros((2, 1)))
        on_mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        on_state = mem.network.variables.GaussianState(on_mean, variance)
        off_state = mem.network.variables.GaussianState(off_mean, variance)
        cluster = mem.network.variables.Cluster([0, 1], [off_state, on_state])
        table = mem.network.tables.Evidence(cluster, [0.5, 0.5])

        # The probabilities of an evidence table can be updated.
        table.update([0.2, 0.8])
        np.testing.assert_array_equal(table.probabilities, [0.2, 0.8])

        # The number of probabilities must match the number of possible
        # states.
        self.assertRaises(
            ValueError,
            table.update,
            [1.0])


class Marginal(unittest.TestCase):

    def test_init(self):

        # Test normal behavior.
        off_mean = np.matrix(np.zeros((2, 1)))
        on_mean = np.matrix(np.ones((2, 1)))
        variance = np.matrix(np.identity(2))
        on_state = mem.network.variables.GaussianState(on_mean, variance)
        off_state = mem.network.variables.GaussianState(off_mean, variance)
        cluster = mem.network.variables.Cluster([0, 1], [off_state, on_state])
        other_cluster = mem.network.variables.Cluster(
            [0, 1], [off_state, on_state])
        table = mem.network.tables.ConditionalProbability(
            [cluster, other_cluster], [0.5, 0.2, 0.1, 0.2])

        cluster_marginal = mem.network.tables.Marginal(table, cluster)
        np.testing.assert_array_almost_equal(
            cluster_marginal.probabilities, [0.6, 0.4])

        other_cluster_marginal = mem.network.tables.Marginal(
            table, other_cluster)
        np.testing.assert_array_almost_equal(
            other_cluster_marginal.probabilities, [0.7, 0.3])

        # The variable to marginalize must be part of the tables domain.
        missing_cluster = mem.network.variables.Cluster(
            [2, 3], [off_state, on_state])

        self.assertRaises(
            ValueError,
            mem.network.tables.Marginal,
            table, missing_cluster)

        # The output probabilities have to be in linear form.
        table = mem.network.tables.ConditionalProbability(
            [cluster, missing_cluster, other_cluster],
            [0.05, 0.2, 0.1, 0.2, 0.05, 0.03, 0.02, 0.1])
        marginal = mem.network.tables.Marginal(table, cluster)
        np.testing.assert_array_almost_equal(
            marginal.probabilities,
            [0.1, 0.23, 0.12, 0.3])
        marginal = mem.network.tables.Marginal(table, missing_cluster)
        np.testing.assert_array_almost_equal(
            marginal.probabilities,
            [0.15, 0.4, 0.07, 0.13])
        marginal = mem.network.tables.Marginal(table, other_cluster)
        np.testing.assert_array_almost_equal(
            marginal.probabilities,
            [0.25, 0.3, 0.08, 0.12])

        # The order of marginalization should not matter.
        marginal_left = mem.network.tables.Marginal(
            mem.network.tables.Marginal(table, cluster),
            other_cluster)
        marginal_right = mem.network.tables.Marginal(
            mem.network.tables.Marginal(table, other_cluster),
            cluster)
        np.testing.assert_array_almost_equal(
            marginal_left.probabilities,
            marginal_right.probabilities)
