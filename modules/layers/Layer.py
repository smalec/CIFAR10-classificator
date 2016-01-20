import numpy as np
from theano import shared


class Layer(object):
    def __init__(self, weights_initializer, weights_shape, biases_initializer, biases_shape):
        self.weights = shared(np.zeros(weights_shape, dtype=np.float32), name='W')
        self.weights.tag.initializer = weights_initializer
        self.biases = shared(np.zeros(biases_shape, dtype=np.float32), name='b')
        self.biases.tag.initializer = biases_initializer
        self.inputs, self.outputs = None, None

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def parameter_names(self):
        return ['W', 'b']
