import numpy as np

__all__ = ["choice"]


def choice(l):
    item = np.random.randint(0, len(l))
    return l[item]
