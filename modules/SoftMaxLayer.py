import theano.tensor as T


class SoftMaxLayer(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weights = self.init_weights()
        self.biases = self.init_biases()

    def no_name(self, inputs):
        self.inputs = inputs
        self.outputs = T.nnet.softmax(self.inputs.dot(self.weights) + self.biases)
        self.result = T.argmax(self.outputs)

    def init_weights(self):
        return None

    def init_biases(self):
        return None

    def cost(self):
        return None#T.log(self.outputs[])
