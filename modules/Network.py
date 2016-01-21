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
        self.predict = function([self.X], T.argmax(self.output, axis=1))

    def connect_layers(self):
        self.layers[0].propagate(self.X)
        for i, layer in enumerate(self.layers[1:]):
            layer.propagate(self.layers[i].outputs)
        return self.layers[-1].outputs

    @property
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters]

    def snapshot_parameters(self):
        return [param.get_value() for param in self.parameters]

    def load_parameters(self, snapshot):
        for param, snap in zip(self.parameters, snapshot):
            param.set_value(snap)

    def train(self, train_stream, validation_stream, rate, momentum, weight_decay):
        self.init_training()
        batch_counter, epoch = 1, 1
        patience_expansion, epochs = 1.5, 3
        best_valid_error, best_params = np.inf, self.snapshot_parameters()
        train_cost, valid_errors, train_errors = [], [], []

        while epoch < epochs:
            for X_batch, Y_batch in train_stream.get_epoch_iterator():
                k = 2000
                learning_rate = rate * k / np.maximum(k, batch_counter)
                cost, train_error = self.train_step(X_batch, Y_batch.ravel(), np.float32(learning_rate),
                                                                              np.float32(momentum),
                                                                              np.float32(weight_decay))
                train_cost.append((batch_counter, cost))
                train_errors.append((batch_counter, train_error))
                if batch_counter % 100 == 0:
                    print "At batch #%d, batch cost: %f" % (batch_counter, cost)
                batch_counter += 1

            valid_error = self.compute_error_rate(validation_stream)
            if valid_error < best_valid_error:
                epochs = np.maximum(epochs, epoch * patience_expansion + 1)
                best_valid_error = valid_error
                best_params = self.snapshot_parameters()
            valid_errors.append((batch_counter, valid_error))
            print "After epoch %d: validation error: %f%%" % \
                  (epoch, valid_error * 100)
            print "Currently going to do %d epochs" % epochs
            epoch += 1

        print "Setting the best obtained parameters..."
        self.load_parameters(best_params)

        return np.array(train_cost), np.array(valid_errors), np.array(train_errors)

    def init_training(self):
        self.init_parameters()
        rate = T.scalar('rate', dtype='float32')
        momentum = T.scalar('momentum', dtype='float32')
        weight_decay = T.scalar('weight_decay', dtype='float32')
        err_rate = T.neq(T.argmax(self.output, axis=1), self.Y.ravel()).mean()

        cost = self.layers[-1].cost(self.Y) + weight_decay*sum([T.sum(p**2) for p in self.parameters if p.name == 'W'])
        gradients = T.grad(cost, self.parameters)
        velocities = [shared(np.zeros_like(p.get_value())) for p in self.parameters]
        updates = []
        for p, g, v in zip(self.parameters, gradients, velocities):
            v_update = momentum * v - rate * g
            p_update = p + v_update
            updates += [(v, v_update), (p, p_update)]

        self.train_step = function([self.X, self.Y, rate, momentum, weight_decay], [cost, err_rate], updates=updates)

    def compute_error_rate(self, stream):
        errors = 0.0
        total = 0
        for X, Y in stream.get_epoch_iterator():
            errors += (self.predict(X) != Y.ravel()).sum()
            total += Y.shape[0]
        return errors / total

    def init_parameters(self):
        for p in self.parameters:
            p.set_value(p.tag.initializer.generate(p.get_value().shape))
