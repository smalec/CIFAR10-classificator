class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.params = [param for layer in layers for param in [layer.weights, layer.biases]]
        #self.connect_layers()

    """def connect_layers(self):
        for i, layer in enumerate(self.layers[1:]):
            layer.no_name(self.layers[i].output)"""

    def train(self, ):
        pass

    def predict(self, inputs):
        pass
