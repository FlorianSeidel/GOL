from math import cos

import theano.tensor as T
import theano
import numpy as np

rng = np.random

import scipy.linalg as sla

class GrassmanGD:

    def __init__(self, U, y, cost, gradient, U_sym, rho=0.6, max_iter_line_search=20,other_givens={},step_size_func=lambda iter: 1.0/iter):

        self.U = U
        self.y = y
        self.shape = self.U.get_value().shape
        self.rho = np.float32(rho)
        self.step_size_func=step_size_func

        self.G = theano.shared(np.zeros(self.shape,dtype=np.float32))
        self.search_direction = theano.shared(np.zeros(self.shape,dtype=np.float32))
        self.max_iter = max_iter_line_search
        self.mapped = theano.shared(np.zeros(self.shape,dtype=np.float32))
        self.total_iter = 0


        givens = {U_sym: self.U}
        givens.update(other_givens)
        self.cost_x = theano.function([], cost, givens=givens, allow_input_downcast=True)
        self.gradient = theano.function([], gradient, givens=givens, updates=[(self.G, gradient)],
                                        allow_input_downcast=True)
        givens = {U_sym: self.mapped}
        givens.update(other_givens)
        self.cost_mapped = theano.function([], cost, givens=givens, allow_input_downcast=True)

        t_init = np.float32(1.0)
        self.t = theano.shared(t_init, )
        self.exp_mapping = self.create_exp_mapping()
        self.accept_new_value = self.create_accept_new_value()
        self.ts_project_func = self.create_ts_project()

        self.UFactor = None
        self.s = None
        self.VFactor = None


    def create_store_gradient(self):
        return theano.function([],[],updates=[(self.last_G,self.G)])


    def create_exp_mapping(self):

        UFactor = T.fmatrix("UFactor")
        Sc = T.fmatrix("Sc")
        Ss = T.fmatrix("Ss")
        VFactor = T.fmatrix("VFactor")

        mapping_func = T.dot(T.dot(self.U,T.dot(VFactor,Sc))
                             + T.dot(UFactor,Ss),VFactor.T)
        mapping = theano.function([UFactor,Sc,Ss,VFactor],[],updates=[(self.mapped,mapping_func)])
        def exp_mapping():
            mapping(self.UFactor,np.diag(np.cos(self.s*self.t.get_value())),np.diag(np.sin(self.s*self.t.get_value())),self.VFactor)

        return exp_mapping


    def create_accept_new_value(self):
        return theano.function([], [], updates=[(self.U, self.mapped)])

    def ts_project(self):
        self.ts_project_func()

    def create_ts_project(self):
        ts_project_func = self.G - T.dot(self.U,T.dot(self.U.T,self.G))
        return theano.function([],[],updates=[(self.G,ts_project_func)])

    def get_cost(self):
        return self.c

    '''
        Returns false if line search was unsuccessful
    '''

    def step(self):
        cInit = self.cost_x()
        print "Init cost", cInit
        self.c = cInit
        c = cInit

        # print "Initial cost: ",cInit
        iterations = 0
        self.total_iter = self.total_iter + 1
        #step_size=0
        self.t.set_value(self.step_size_func(self.total_iter))
        print "step size", self.t.get_value()
        self.gradient()
        self.ts_project()
        self.UFactor,self.s,self.VFactor = sla.svd(-self.G.get_value(), full_matrices=False)
        while c >= cInit and iterations < self.max_iter:
            iterations += 1
            self.exp_mapping()
            c = self.cost_mapped()
            self.t.set_value(np.float32(self.t.get_value() * self.rho))
            print c

        if c < cInit:
            self.c = c
            self.accept_new_value()
        else:
            print " No progress "
            return False

        return c



