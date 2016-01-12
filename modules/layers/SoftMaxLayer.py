import theano.tensor as T
from theano import shared
import numpy as np
from Layer import Layer


class SoftMaxLayer(Layer):
    def __init__(self, in_size, out_size):
        super(SoftMaxLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.inputs, self.outputs = None, None

    def propagate(self, inputs):
        self.inputs = inputs
        self.outputs = T.nnet.softmax(T.dot(self.inputs, self.weights) + self.biases.dimshuffle('x', 0))

    def init_weights(self):
        return shared(np.zeros((self.in_size, self.out_size), dtype=np.float32), name='W')

    def init_biases(self):
        return shared(np.zeros((self.out_size,), dtype=np.float32), name='b')

    def cost(self, Y):
        return - T.log(self.outputs[T.arange(Y.shape[0]), Y]).mean()

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def parameter_names(self):
        return ['W', 'b']
