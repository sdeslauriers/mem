

class Bayesian(object):
    def __init__(self):
        """MEM Bayesian network"""

        self._variables = []
        self._tables = []

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
