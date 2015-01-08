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

from math import log

import theano.tensor as T
from theano import Op
from theano.gof import Apply
import theano.sandbox.linalg.ops as op
import numpy as np
from scipy.linalg import svd



class LogDet(Op):
    """Matrix determinant
    """

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        o = T.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def set_x(self,x):
        self.x_trick=x

    def perform(self, node, (x, ), (z, )):
        try:
            s = svd(x.T, compute_uv=False)
            z[0] = np.asarray(np.sum(np.log(s**2)), dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * T.dot(x,op.matrix_inverse(T.dot(x.T,x)))]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "Det"


logdet = LogDet()

'''
    creates a constaint of the form - 1/(n*log(n)) * log det (1/k W^T*W)
    W is kxn and k>=n
'''


def full_rank_constraint(W, k, n):
    return (float(k * log(k)) - logdet(W)) / float(n * log(n))


'''
    creates a constraint of the form - sum_over_all_pairs_i!=j log(1 - (W[i,:]^T*W[j,:])^2)
    W is kxn and k>=n
'''
def no_linear_dependencies_constraint(W, k):
    M=T.dot(W, W.T)
    M=M-T.diag(T.diagonal(M))
    cost = -T.sum(T.log(1.0 - M ** 2))/2
    return cost


if __name__ == '__main__':
    pass