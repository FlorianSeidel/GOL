'''
Created on 17.10.2012

@author: florian
'''


import theano.tensor as T
import theano
import numpy as np
import time

import math

rng = np.random

class ObliqueCG:

    '''
        X:= shared variable containing the actual weights
        cost:= cost function in terms of W
        gradient:= of cost function in terms of W
        W:= symbolic weight variable (kxn)
        D:= dataset shared variable (nxm)

        Activation: W*D
    '''

    def __init__(self, X, cost, gradient, W, k, n, t_init=1.0, rho=0.6, max_iter_line_search=20,other_givens={}):

        self.k = k
        self.n = n
        self.X = X
        self.rho = np.float32(rho)
        self.last_G = theano.shared(np.float32(np.zeros((k, n))))
        self.G = theano.shared(np.float32(np.zeros((k, n))))
        self.search_direction = theano.shared(np.float32(np.zeros((k, n))))
        self.beta = theano.shared(np.float32(0),)
        self.max_iter = max_iter_line_search
        self.mapped = theano.shared(np.float32(np.zeros((k, n))))
        self.total_iter = 0

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
        self.t_prev = theano.shared(self.t_init,)
        self.store_t = theano.function([],[],updates=[(self.t_prev,self.t)])
        self.exp_mapping = self.create_exp_mapping()
        self.accept_new_value = self.create_accept_new_value()
        self.stabilizer = self.create_stabilizer()
        self.gradient_norm = self.create_gradient_norm()
        self.store_gradient = self.create_store_gradient()
        self.calc_beta = self.create_calc_beta()
        self.update_direction = self.create_update_direction()
        self.transport_gradient = self.create_transport_gradient()
        self.transport_search_direction = self.create_transport_search_direction()
        self.ts_project_func = self.create_ts_project()


    def create_update_direction(self):
        new_search_direction = -self.G + self.beta*self.search_direction
        return theano.function([],new_search_direction,updates=[(self.search_direction,new_search_direction)])

    def create_calc_beta(self):
        yk = (self.G-self.last_G)
        denom = ((yk*self.search_direction).sum() + np.spacing(np.single(1)))
        beta1 = (yk*self.G).sum() / denom
        beta2 = (self.G**2).sum() /denom
        return theano.function([],[beta1,beta2],updates=[(self.beta,T.maximum(0,T.minimum(beta1,beta2)))])

    def create_store_gradient(self):
        return theano.function([],[],updates=[(self.last_G,self.G)])

    def create_gradient_norm(self):
        return theano.function([], T.sqrt((self.G ** 2).sum(axis=1)))

    def create_transport_gradient(self):
        HL2 = T.sqrt((self.search_direction ** 2).sum(axis=0,keepdims=True)) + np.spacing(np.single(1))

        transported = self.last_G - (self.last_G*self.search_direction).sum(axis=0,keepdims=True)/(HL2**2) * \
                               (self.X*HL2 * T.sin(self.t_prev*HL2) + self.search_direction * (1.0-T.cos(self.t_prev*HL2)))
        f = theano.function([],[],updates=[(self.last_G,transported)])
        return f

    def create_transport_search_direction(self):
        HL2 = T.sqrt((self.search_direction ** 2).sum(axis=1,keepdims=True))
        transported = self.X*HL2 * T.sin(self.t_prev*HL2) - self.search_direction * T.cos(self.t_prev*HL2)
        return theano.function([],[],updates=[(self.search_direction,transported)])

    def create_exp_mapping(self):
        HL2 = T.sqrt((self.search_direction ** 2).sum(axis=1,keepdims=True))
        step=(self.X * T.cos(self.t * HL2)) + (self.search_direction/HL2 * (T.sin(self.t * HL2)))
        return theano.function([], [], updates=[
            (self.mapped, step )])

    def create_stabilizer(self):
        ML2 = T.sqrt((self.mapped ** 2).sum(axis=1,keepdims=True))
        return theano.function([], [], updates=[(self.mapped, (self.mapped / ML2))])

    def create_accept_new_value(self):
        return theano.function([], [], updates=[(self.X, self.mapped)])

    def ts_project(self):
        self.ts_project_func()

    def create_ts_project(self):
        ts_project_func = self.G - self.X*((self.X*self.G).sum(axis=1,keepdims=True))
        return theano.function([],[],updates=[(self.G,ts_project_func)])

    def getCost(self):
        return self.c

    '''
        Returns false if line search was unsuccessful
    '''

    def step(self):
        cInit = self.cost_x()
        self.c = cInit
        c = cInit
        if self.total_iter % (self.k*self.n) != 0:
            self.store_gradient()
            self.gradient()  # updates G
            self.ts_project()
            self.transport_gradient()
            self.transport_search_direction()
            self.calc_beta()
            self.update_direction()
        else:
            print "Initialization..."
            self.gradient()
            self.ts_project()
            self.beta.set_value(np.float32(0))
            self.update_direction()
            #self.search_direction.set_value(-1*self.G.get_value())
            self.t.set_value(np.float32(2.0*math.pi/(np.sqrt(np.sum(self.G.get_value()**2)))))


        # print "Initial cost: ",cInit
        iterations = 0
        self.total_iter = self.total_iter + 1
        while c >= cInit and iterations < self.max_iter:
            iterations += 1
            self.exp_mapping()
            self.stabilizer()
            c = self.cost_mapped()
            self.t.set_value(np.float32(self.t.get_value() * self.rho))
            self.store_t()

        self.t.set_value(np.float32(self.t.get_value()/(self.rho**2)))
        if c < cInit:
            self.c = c
            self.accept_new_value()
        else:
            print " No progress "
            return False

        return c

    def gram_matrix(self):
        Whost = self.X.get_value()
        wgram = np.zeros((Whost.shape[0], Whost.shape[0])).astype(np.float32)
        for i in xrange(0, Whost.shape[0]):
            for j in xrange(0, Whost.shape[0]):
                wgram[i, j] = np.dot(Whost[i, :], Whost[j, :])
        return wgram

    def gram_matrix_mapped(self):
        Whost = self.mapped.get_value();
        wgram = np.zeros((Whost.shape[0], Whost.shape[0])).astype(np.float32)
        for i in xrange(0, Whost.shape[0]):
            for j in xrange(0, Whost.shape[0]):
                wgram[i, j] = np.dot(Whost[i, :], Whost[j, :])
        return wgram

    def optimize(self, max_iter, eval_func=None, eval_rate=10):
        self.total_iter=0
        start = time.clock()
        while self.total_iter < max_iter:
            c = self.step()
            D = {'c': c ,'iter':self.total_iter,'elapsed':time.clock()-start}
            if not c:
                if eval_func:
                    eval_func(D)
                return
            if eval_func and self.total_iter % eval_rate == 0:
                print "Iteration: ", self.total_iter
                eval_func(D)


