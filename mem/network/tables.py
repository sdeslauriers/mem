
import collections
from functools import reduce
import itertools
import operator


class BaseTable(object):
    def __init__(self, variables):
        """Base class for tables"""

        # Support both a single variable and an iterable of variables.
        if isinstance(variables, collections.abc.Iterable):
            variables = list(variables)
        else:
            variables = [variables]
        self._variables = variables

    def __str__(self):
        """String representation of the conditional probability table"""

        out = ''
        states = [range(v.nb_states) for v in self.variables]
        for state, p in zip(itertools.product(*states), self.probabilities):
            out += ' '.join((str(s) for s in state))
            out += ' | {}\n'.format(p)

        return out

    @property
    def nb_states(self):
        return reduce(operator.mul, [v.nb_states for v in self.variables])

    @property
    def probabilities(self):
        raise NotImplementedError(
            'The BaseTable class does not implement the probabilities '
            'property. It must be implemented by a subclass.')

    @property
    def variables(self):
        return self._variables

    def validate_probabilities(self, probabilities):

        # The number of probabilities must match the number of possible
        # states.
        if len(probabilities) != self.nb_states:
            raise ValueError(
                'The number of probabilities does not match the number of '
                'possible states ({} != {}).'
                .format(len(probabilities), self.nb_states))


class ConditionalProbability(BaseTable):
    def __init__(self, variables, probabilities):
        """Conditional probability table"""

        super().__init__(variables)
        self.validate_probabilities(probabilities)
        self._probabilities = probabilities

    @property
    def probabilities(self):
        return self._probabilities

    def update(self, probabilities):
        self.validate_probabilities(probabilities)
        self._probabilities = probabilities


class Evidence(BaseTable):
    def __init__(self, variable, probabilities):
        """Evidence probability table"""

        # Evidence tables can only depend on a single variable.
        if isinstance(variable, collections.abc.Iterable):
            if len(variable) != 1:
                raise ValueError(
                    'Evidence tables can only depend on a single variable.')

        super().__init__(variable)
        self.validate_probabilities(probabilities)
        self._probabilities = probabilities

    @property
    def probabilities(self):
        return self._probabilities

    def update(self, probabilities):
        self.validate_probabilities(probabilities)
        self._probabilities = probabilities
