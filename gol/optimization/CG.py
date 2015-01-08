
__author__ = 'florian'

import theano
import numpy as np

rng = np.random

class CG:

    def __init__(self, X, cost, gradient, W, k, n, t_init=1.0, rho=0.6, max_iter_line_search=20, other_givens={}):
        self.k = k
        self.n = n
        self.X = X
        self.rho = np.float32(rho)
        self.last_G = theano.shared(np.float32(np.zeros((k, n))))
        self.G = theano.shared(np.float32(np.zeros((k, n))))
        self.search_direction = theano.shared(np.float32(np.zeros((k, n))))
        self.beta = theano.shared(np.float32(0), )
        self.max_iter = max_iter_line_search
        self.mapped = theano.shared(np.float32(np.zeros((k, n))))
        self.total_iter = 0
        self.sgd_iter = 0
        self.c = 0
        self.c_param=0.001
        givens = {W: self.X}
        givens.update(other_givens)
        self.cost_x = theano.function([], cost, givens=givens, allow_input_downcast=True)
        self.gradient = theano.function([], gradient, givens=givens, updates=[(self.G, gradient)],
                                          allow_input_downcast=True)
        givens = {W: self.mapped}
        givens.update(other_givens)
        self.cost_mapped = theano.function([], cost, givens=givens, allow_input_downcast=True)
        self.t_init = np.float32(t_init)
        self.t = theano.shared(self.t_init, )

        self.make_step = self.create_make_step()
        self.accept_new_value = self.create_accept_new_value()
        self.store_gradient = self.create_store_gradient()
        self.calc_beta = self.create_calc_beta()
        self.update_direction = self.create_update_direction()
        self.armijo_rule = theano.function([],-self.c_param * (self.G*self.search_direction).sum())

    def create_update_direction(self):
        new_search_direction = self.G + self.beta * self.search_direction
        return theano.function([], new_search_direction, updates=[(self.search_direction, new_search_direction)])

    def create_calc_beta(self):
        beta = ((self.G - self.last_G) * self.G).sum() /  (
        ((self.G - self.last_G) * self.search_direction).sum() + np.spacing(np.single(1)))
        # beta = ((self.G**2).sum()/(self.last_G**2).sum())
        return theano.function([], [], updates=[(self.beta, beta)])

    def create_store_gradient(self):
        return theano.function([], [], updates=[(self.last_G, self.G)])


    def create_accept_new_value(self):
        return theano.function([], [], updates=[(self.X, self.mapped)])

    def create_make_step(self):
        step = self.X - self.t * self.search_direction
        return theano.function([], [], updates=[(self.mapped, step)])

    def getCost(self):
        return self.c

    '''
        Returns false if line search was unsuccessful
    '''
    def step(self):
        cInit = self.cost_x()
        self.c = cInit
        c = cInit

        # print "Initial cost: ",cInit
        iterations = 0
        self.total_iter = self.total_iter + 1

        s = self.armijo_rule()

        if self.max_iter > -1:
            while c > (cInit+ self.t.get_value()*s) and iterations < self.max_iter:
                iterations += 1
                self.t.set_value(np.float32(self.t.get_value() * self.rho))
                self.make_step()
                c = self.cost_mapped()

        if self.total_iter% (self.n*self.k)==0:
            self.gradient()
            self.beta.set_value(np.float32(0))
            self.update_direction()
        else:
            self.store_gradient()
            self.gradient()
            self.calc_beta()
            self.update_direction()

        if c <= (cInit+ self.t.get_value()*s):
            self.c = c
            self.accept_new_value()
            self.t.set_value(np.float32(self.t.get_value() / ( self.rho**2)))
        else:
            return False

        return c

    def optimize(self, max_iter, eval_func=None, eval_rate=10):
        self.gradient()  # updates G
        self.beta.set_value(np.float32(0))
        self.update_direction()

        self.t.set_value(np.float32(1.0))
        self.total_iter = 0
        while self.total_iter < max_iter:
            c = self.step()
            if not c:
                if eval_func:
                    eval_func({'X': self.X, 'c': c})
                return
            if eval_func and self.total_iter % eval_rate == 0:
                print "Iteration: ", self.total_iter
                eval_func({'X': self.X, 'c': c})
