import numpy as np


class Constant(object):
    def __init__(self, constant):
        self.constant = constant

    def generate(self, shape):
        return self.constant * np.ones(shape, dtype=np.float32)
