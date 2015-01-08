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