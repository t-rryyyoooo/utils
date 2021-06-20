import numpy as np
from itertools import product

class DummyGenerator:
    def __init__(self, length=10**9):
        self.length = length

    def __call__(self):
        """ Slice image on axis and yield it. """
        for l in range(self.length):
            yield l, "dummy"
