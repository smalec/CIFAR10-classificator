import theano.tensor as T
import numpy as np
from Layer import Layer
from theano import shared
from theano.tensor.signal.downsample import max_pool_2d
from ..initializators.Normal import Normal

ReLU = lambda x: T.maximum(0.0, x)


class ConvolutionalPoolLayer(Layer):
    def __init__(self, image_shape, filter_shape, pool_shape, activation=ReLU):
        super(ConvolutionalPoolLayer, self).__init__()
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.activation = activation
        self.weights = self.init_weights()
        self.biases = self.init_biases()

    def propagate(self, inputs):
        images_shape = (inputs.shape[0],) + self.image_shape
        self.inputs = T.reshape(inputs, images_shape, ndim=4)
        convolution = T.nnet.conv2d(self.inputs, self.weights) + self.biases.dimshuffle('x', 0, 'x', 'x')
        self.outputs = max_pool_2d(self.activation(convolution), ds=self.pool_shape, ignore_border=True)

    def init_weights(self):
        weights = shared(np.zeros(self.filter_shape, dtype=np.float32), name='W')
        weights.tag.initializer = Normal(np.sqrt(1.0/np.prod(self.filter_shape[2:])))
        return weights

    def init_biases(self):
        biases = shared(np.zeros((self.filter_shape[0],), dtype=np.float32), name='b')
        biases.tag.initializer = Normal(1.0)
        return biases
