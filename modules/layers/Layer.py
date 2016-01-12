class Layer(object):
    def __init__(self):
        self.weights, self.biases = None, None
        self.inputs, self.outputs = None, None

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def parameter_names(self):
        return ['W', 'b']
