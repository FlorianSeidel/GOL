'''
Created on 17.10.2012

@author: florian
'''
import theano.tensor as T
from theano.tensor.nnet import conv
from constraints import full_rank_constraint
from constraints import no_linear_dependencies_constraint
#from ops.nolineardepsops import NoLinearDepOp, NoLinearDepOpGradient


def reconstruction_logsquared_cost(Iest,Omega,gamma):
    return T.log(1+gamma*conv.conv2d(Iest,Omega,border_mode='valid')**2)

def lpnorm_cost(V, p, gamma, axis=None):
    return 1 / float(p) * ((V ** 2 + gamma) ** (p / 2.0)).sum(axis=axis)

def logsquared_cost(V,gamma,axis=None):
    return T.log(1+gamma*V**2).sum(axis=axis)

'''
    m is the number of samples
'''
def normalized_l2norm_cost(V, m):
    return 1 / (2.0 * m) * (V ** 2).sum()


if __name__ == '__main__':
    pass