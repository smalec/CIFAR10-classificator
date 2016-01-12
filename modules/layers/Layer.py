class Layer(object):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weights, self.biases = None, None
        self.inputs, self.outputs = None, None

    @property
    def parameters(self):
        return [self.weights, self.biases]

    @property
    def parameter_names(self):
        return ['W', 'b']
