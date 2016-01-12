import theano.tensor as T
import numpy as np
from theano import function


class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.X = T.matrix('X', dtype='float32')
        self.Y = T.ivector('Y')
        self.output = self.connect_layers()
        self.run = function([self.X], T.argmax(self.output, axis=1))

    def connect_layers(self):
        self.layers[0].propagate(self.X)
        for i, layer in enumerate(self.layers[1:]):
            layer.propagate(self.layers[i].outputs)
        return self.layers[-1].outputs

    @property
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters]

    def train(self, train_stream, validation_stream, epochs, rate):
        self.init_training()
        for epoch in range(epochs):
            batch_counter = 1
            for X_batch, Y_batch in train_stream.get_epoch_iterator():
                cost = self.train_step(X_batch, Y_batch.ravel(), np.float32(rate))
                if batch_counter % 100 == 0:
                    print "At batch #%d, batch cost: %f" % (batch_counter, cost)
                batch_counter += 1

            print "After epoch %d: validation error: %f%%" % \
                  (epoch + 1, self.compute_error_rate(validation_stream) * 100)

    def init_training(self):
        cost = self.layers[-1].cost(self.Y)
        gradients = T.grad(cost, self.parameters)
        rate = T.scalar('rate', dtype='float32')
        updates = [(p, p - (rate * g)) for p, g in zip(self.parameters, gradients)]
        self.train_step = function([self.X, self.Y, rate], cost, updates=updates)

    def predict(self, inputs):
        return self.run(inputs)

    def compute_error_rate(self, stream):
        errors = 0.0
        total = 0
        for X, Y in stream.get_epoch_iterator():
            errors += (self.predict(X) != Y.ravel()).sum()
            total += Y.shape[0]
        return errors / total
