import numpy as np


class Uniform(object):
    def __init__(self, mean=0.0, width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, but not both")
        if std is not None:
            self.width = np.sqrt(12) * std
        else:
            self.width = width
        self.mean = mean

    def generate(self, shape):
        half = self.width / 2
        return np.random.uniform(low=self.mean - half, high=self.mean + half, size=shape).astype(np.float32)
