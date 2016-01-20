import theano.tensor as T
import numpy as np
from Layer import Layer
from ..initializers.Normal import Normal
from ..initializers.Constant import Constant


class AffineLayer(Layer):
    def __init__(self, in_size, out_size, activation=T.nnet.sigmoid, weights_initializer=None, biases_initializer=None):
        super(AffineLayer, self).__init__(weights_initializer, (in_size, out_size), biases_initializer, (out_size,))
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        if not self.weights.tag.initializer:
            self.init_weights()
        if not self.biases.tag.initializer:
            self.init_biases()

    def propagate(self, inputs):
        self.inputs = inputs.reshape((inputs.shape[0], self.in_size))
        self.outputs = self.activation(T.dot(self.inputs, self.weights) + self.biases.dimshuffle('x', 0))

    def init_weights(self):
        self.weights.tag.initializer = Normal(std=np.sqrt(1.0/self.out_size))

    def init_biases(self):
        self.biases.tag.initializer = Constant(0.0)
