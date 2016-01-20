import theano.tensor as T
import numpy as np
from Layer import Layer
from theano.tensor.signal.downsample import max_pool_2d
from ..initializers.Normal import Normal
from ..initializers.Constant import Constant

ReLU = lambda x: T.maximum(0.0, x)


class ConvolutionalPoolLayer(Layer):
    def __init__(self, image_shape, filter_shape, pool_shape, activation=ReLU,
                 weights_initializer=None, biases_initializer=None, conv_mode='valid'):
        super(ConvolutionalPoolLayer, self).__init__(weights_initializer, filter_shape,
                                                     biases_initializer, filter_shape[0])
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.activation = activation
        self.conv_mode = conv_mode
        if not self.weights.tag.initializer:
            self.init_weights()
        if not self.biases.tag.initializer:
            self.init_biases()

    def propagate(self, inputs):
        images_shape = (inputs.shape[0],) + self.image_shape
        self.inputs = T.reshape(inputs, images_shape, ndim=4)
        if self.conv_mode == 'same':
            pads = ((self.filter_shape[2]/2, (self.filter_shape[2] - 1)/2),
                    (self.filter_shape[2]/2, (self.filter_shape[3] - 1)/2))
            img_rng = (pads[0][0], images_shape[2] + pads[0][1]), (pads[1][0], images_shape[3] + pads[1][1])
            full_conv_no_bias = T.nnet.conv2d(self.inputs, self.weights, border_mode='full')
            conv_no_bias = full_conv_no_bias[:, :, img_rng[0][0]:img_rng[0][1], img_rng[1][0]:img_rng[1][1]]
        else:
            conv_no_bias = T.nnet.conv2d(self.inputs, self.weights, border_mode=self.conv_mode)
        convolution = conv_no_bias + self.biases.dimshuffle('x', 0, 'x', 'x')
        self.outputs = max_pool_2d(self.activation(convolution), ds=self.pool_shape, ignore_border=True)

    def init_weights(self):
        out_size = np.prod(self.filter_shape) / self.filter_shape[1]
        self.weights.tag.initializer = Normal(std=np.sqrt(1.0/out_size))

    def init_biases(self):
        self.biases.tag.initializer = Constant(0.0)
