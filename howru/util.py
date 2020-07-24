"""
Collection of utility functions.

"""

import itertools

__date__ = "2020-07-23"
__version__ = "0.1"


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    xs = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(xs, n) for n in range(len(xs)+1))
