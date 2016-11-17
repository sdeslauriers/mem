
from functools import reduce

import numpy as np

import mem.network.tables


class Bayesian(object):
    def __init__(self):
        """MEM Bayesian network"""

        self._variables = []
        self._tables = []
        self._interface = []

    def add_variable(self, variable):

        # The variables must be unique.
        if variable in self._variables:
            raise ValueError(
                'The variable {} is already part of the network.'
                .format(variable))

        self._variables.append(variable)

    def add_table(self, table):

        # The variables of the table all have to be part of the domain
        # and appear in the same order.
        past_index = -1
        for variable in table.variables:
            if variable not in self._variables:
                raise ValueError(
                    'The variable {} is part of the table but '
                    'not of the network.'
                    .format(variable))

            index = self._variables.index(variable)
            if index <= past_index:
                raise ValueError(
                    'The variables in the table do not appear in the '
                    'same order as in the network.')
            past_index = index

        self._tables.append(table)

    def marginalize(self, variable):
        """Marginalizes a variable out of the Bayesian network

        Removes a variable from the domain of the Bayesian network by summing
        it out. The result is a new Bayesian network.

        Args:
            variable (bayesian.Variable) : The variable to be removed.

        """

        # Find all tables with the variable in their domain.
        tables = []
        for table in self._interface:
            if variable in table.domain:
                tables.append(table)

        # Compute the product of all tables and marginalize the variable out
        # of the result.
        def mul(left, right):
            return mem.network.tables.Product(self._variables, left, right)
        new_table = reduce(mul, tables)
        new_table = new_table.marginalize(variable)

        # Remove all table with the variable in their domain and add the
        # new one.
        for table in tables:
            self._interface.remove(table)
        self._interface.append(new_table)

    def log_partition(self, lagrange):
        """Evaluate the log partition of the network"""

        # Update all evidence tables.
        for table in self._tables:
            if isinstance(table, mem.network.tables.Evidence):
                table.update(lagrange)

        # Marginalize all variables except the first one.
        self._interface = self._tables
        for variable in self._variables[1:]:
            self.marginalize(variable)

        # Compute the product of all remaining tables.
        def mul(left, right):
            return mem.network.tables.Product(self._variables, left, right)
        new_table = reduce(mul, self._interface)

        return np.log(np.sum(new_table.probabilities))
