import numpy as np


class Normal(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def generate(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape).astype(np.float32)
