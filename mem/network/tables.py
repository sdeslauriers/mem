
import collections
from functools import reduce
import itertools
import operator


class ConditionalProbability(object):
    def __init__(self, variables, probabilities):
        """Conditional probability table"""

        # Support both a single variable and an iterable of variables.
        if isinstance(variables, collections.abc.Iterable):
            variables = list(variables)
        else:
            variables = [variables]

        # The number of probabilities must match the number of possible
        # states.
        nb_states = reduce(operator.mul, [v.nb_states for v in variables])
        if len(probabilities) != nb_states:
            raise ValueError(
                'The number of probabilities does not match the number of '
                'possible states ({} != {}).'
                .format(len(probabilities), nb_states))

        self._variables = variables
        self._probabilities = probabilities

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
        return len(self.probabilities)

    @property
    def probabilities(self):
        return self._probabilities

    @property
    def variables(self):
        return self._variables
