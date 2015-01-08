__author__ = 'florian'
'''
Copyright (c) 2015 FlorianSeidel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import cv2

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import theano
import theano.tensor as T

from gol.optimization.CG import CG
from gol.optimization.GrassmanGD import GrassmanGD


class pROST:
    def __init__(self, n, k, p, mu,t,phi,y_iters=5,U_init=None,step_size_func=lambda x: 1.0/x,median_filter_radius=3):
        self.n = n
        self.k = k
        self.p = p
        self.mu = mu
        self.y_iters=y_iters
        self.phi = theano.shared(np.float32(phi))
        self.t = theano.shared(np.float32(t))
        self.image_shape = None
        self.step_size_func=step_size_func
        self.median_filter_radius=median_filter_radius

        if U_init==None:
            U = npr.randn(n, k).astype(np.float32)
            U, _ = npl.qr(U)
        else:
            U = U_init

        self.U_sym = T.fmatrix("U")
        self.U_shared = theano.shared(U.astype(np.float32))

        self.W_shared = theano.shared(np.ones((n,1), dtype=np.float32))

        self.y_sym = T.fmatrix("y")
        self.y_shared = theano.shared(npr.randn(k,1).astype(np.float32))

        self.image_sym = T.fmatrix("image")
        self.image_shared = theano.shared(np.zeros((n,1), dtype=np.float32))
        self.init_y_func = T.dot(self.U_shared.T,self.image_shared)
        self.init_y = theano.function([],[],updates={self.y_shared:self.init_y_func})
        self.cost = (self.W_shared * ((((self.image_sym - T.dot(self.U_sym, self.y_sym))) ** 2 + self.mu) ** (self.p / 2))).sum()
        self.reconstruction_func = T.dot(self.U_sym, self.y_sym)
        self.error_image_func = T.abs_(self.image_sym - T.dot(self.U_sym, self.y_sym))

        self.segmentation_func = self.error_image_func > self.t

        self.grad_y = theano.grad(self.cost, self.y_sym)
        self.grad_U = theano.grad(self.cost, self.U_sym)
        self.optimize_y = CG(self.y_shared, self.cost, self.grad_y, self.y_sym, self.k, 1, t_init=0.1, rho=0.5,
                             max_iter_line_search=200,
                             other_givens={self.U_sym: self.U_shared, self.image_sym: self.image_shared})
        self.optimize_U = GrassmanGD(self.U_shared, self.y_shared, self.cost, self.grad_U, self.U_sym,
                                     rho=0.6, max_iter_line_search=5,
                                     other_givens={self.image_sym: self.image_shared, self.y_sym: self.y_shared},
                                     step_size_func=self.step_size_func)
        self.recerrsegweight = theano.function([], [self.reconstruction_func, self.error_image_func,self.segmentation_func],
                                                        givens={self.U_sym: self.U_shared, self.y_sym: self.y_shared,
                                                                self.image_sym: self.image_shared})

        self.error_image=None
        self.reconstruction=None
        self.segmentation=None


    def process_image(self, image):
        self.image_shape=image.shape
        self.image_shared.set_value(image.reshape(np.prod(self.image_shape),1))
        self.init_y()
        self.optimize_y.optimize(self.y_iters)
        self.reconstruction,self.error_image,self.segmentation = self.recerrsegweight()
        self.segmentation=self.segmentation.reshape(self.image_shape)
        gray = cv2.cvtColor(self.segmentation.astype(np.float32), cv2.COLOR_BGR2GRAY)
        gray[gray>0]=1.0
        gray = cv2.medianBlur(gray,self.median_filter_radius)
        self.W_shared.set_value(1.0 - (1.0-self.phi.get_value())*cv2.merge([gray,gray,gray]).reshape((self.n,1)))

        self.segmentation=np.uint8(gray)*255

        self.optimize_U.step()

    def get_error_image(self):
        return self.error_image.reshape(self.image_shape)

    def get_reconstruction(self):
        return self.reconstruction.reshape(self.image_shape)

    def get_segmentation(self):
        return self.segmentation

    def get_segmentation_color(self):
        return cv2.merge([self.segmentation,self.segmentation,self.segmentation])

    def get_weights(self):
        return self.W_shared.get_value().reshape(self.image_shape)


    def set_threshold(self,t):
        self.t.set_value(np.float32(t))

    def set_foreground_weighting(self,phi):
        self.phi.set_value(np.float32(phi))