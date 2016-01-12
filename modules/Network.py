import theano.tensor as T
import numpy as np
from theano import function
from theano import shared


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

    def train(self, train_stream, validation_stream, epochs, rate, momentum, weight_decay):
        self.init_training()
        batch_counter = 1
        for epoch in range(epochs):
            for X_batch, Y_batch in train_stream.get_epoch_iterator():
                k = 2000
                rate = rate * k / np.maximum(k, batch_counter)
                cost = self.train_step(X_batch, Y_batch.ravel(), np.float32(rate),
                                                                 np.float32(momentum),
                                                                 np.float32(weight_decay))
                if batch_counter % 100 == 0:
                    print "At batch #%d, batch cost: %f" % (batch_counter, cost)
                batch_counter += 1

            print "After epoch %d: validation error: %f%%" % \
                  (epoch + 1, self.compute_error_rate(validation_stream) * 100)

    def init_training(self):
        rate = T.scalar('rate', dtype='float32')
        momentum = T.scalar('momentum', dtype='float32')
        weight_decay = T.scalar('weight_decay', dtype='float32')

        cost = self.layers[-1].cost(self.Y)
        gradients = T.grad(cost, self.parameters)
        velocities = [shared(np.zeros_like(p.get_value())) for p in self.parameters]
        updates = []
        for p, g, v in zip(self.parameters, gradients, velocities):
            v_update = momentum * v - rate * (g + weight_decay*p)
            p_update = p + v_update
            updates += [(v, v_update), (p, p_update)]

        self.train_step = function([self.X, self.Y, rate, momentum, weight_decay], cost, updates=updates)

    def predict(self, inputs):
        return self.run(inputs)

    def compute_error_rate(self, stream):
        errors = 0.0
        total = 0
        for X, Y in stream.get_epoch_iterator():
            errors += (self.predict(X) != Y.ravel()).sum()
            total += Y.shape[0]
        return errors / total
