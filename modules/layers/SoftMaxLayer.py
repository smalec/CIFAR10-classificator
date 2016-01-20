import theano.tensor as T
from Layer import Layer
from ..initializers.Constant import Constant


class SoftMaxLayer(Layer):
    def __init__(self, in_size, out_size, weights_initializer=None, biases_initializer=None):
        super(SoftMaxLayer, self).__init__(weights_initializer, (in_size, out_size), biases_initializer, (out_size,))
        self.in_size = in_size
        self.out_size = out_size
        if not self.weights.tag.initializer:
            self.init_weights()
        if not self.biases.tag.initializer:
            self.init_biases()

    def propagate(self, inputs):
        self.inputs = inputs.reshape((inputs.shape[0], self.in_size))
        self.outputs = T.nnet.softmax(T.dot(self.inputs, self.weights) + self.biases.dimshuffle('x', 0))

    def init_weights(self):
        self.weights.tag.initializer = Constant(0.0)

    def init_biases(self):
        self.biases.tag.initializer = Constant(0.0)

    def cost(self, Y):
        return - T.log(self.outputs[T.arange(Y.shape[0]), Y]).mean()
