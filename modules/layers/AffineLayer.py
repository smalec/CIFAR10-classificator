import theano.tensor as T
import numpy as np
from theano import shared
from Layer import Layer
from ..initializators.Normal import Normal


class AffineLayer(Layer):
    def __init__(self, in_size, out_size, activation=T.nnet.sigmoid, weights_initializer=None, biases_initializer=None):
        super(AffineLayer, self).__init__(weights_initializer, biases_initializer)
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        if not self.weights:
            self.weights = self.init_weights()
        if not self.biases:
            self.biases = self.init_biases()

    def propagate(self, inputs):
        self.inputs = inputs.reshape((inputs.shape[0], self.in_size))
        self.outputs = self.activation(T.dot(self.inputs, self.weights) + self.biases.dimshuffle('x', 0))

    def init_weights(self):
        weights = shared(np.zeros((self.in_size, self.out_size), dtype=np.float32), name='W')
        weights.tag.initializer = Normal(std=np.sqrt(1.0/self.out_size))
        return weights

    def init_biases(self):
        biases = shared(np.zeros((self.out_size,), dtype=np.float32), name='b')
        biases.tag.initializer = Normal(std=1.0)
        return biases
