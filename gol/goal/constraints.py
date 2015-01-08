'''
Created on 17.10.2012

@author: florian
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