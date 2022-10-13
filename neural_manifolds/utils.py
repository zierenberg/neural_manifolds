from functools import wraps
from pprint import pformat
import numpy as np


def to_string(object):
    return pformat(
        object=object,
        indent=1,
        width=80,
        depth=2,
        compact=False,
        sort_dicts=True,
        underscore_numbers=True,
    )


default_seed = 0


def set_seed(seed=None):
    """
    Decorator to set the seed of the random number generator.
    To change the seed globaly set the variable `default_seed` to the desired.
    """
    if seed is None:
        seed = default_seed

    def decorator(func):
        def wrapper(*args, **kwargs):
            np.random.seed(seed)
            return func(*args, **kwargs)

        return wrapper

    return decorator
