class Layer(object):
    def __init__(self, weights_initializer, biases_initializer):
        self.weights, self.biases = weights_initializer, biases_initializer
        self.inputs, self.outputs = None, None

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def parameter_names(self):
        return ['W', 'b']
