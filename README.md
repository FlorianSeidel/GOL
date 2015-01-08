# GOL

### About
This package was written to acompany the [pROST] paper. We first had a CUDA/C++ version of the algorithm, but had all kinds of problems with getting it to work on other computers. So here I implemented the algorithm in Python based on [numpy], [scipy] and [Theano]. The implementation is not as efficient as the CUDA/C++ implementation, but considerably easier to install and understand. The results are very close to what was reported in the paper.  
Additionally, I implemented the Analysis Operator learning algorithm from the [GOAL] paper by Simon Hawe, Martin Kleinsteuber and Klaus Diepold.

#### gol.optimization
The gol.goal.optimization package contains implementations of algorithms for minimizing functions defined on the Grassmann manifold and the Oblique manifold. The algorithm for optimization on the Grassmann manifold is a simple gradient decent algorithm. The algorithm for optimization on the Oblique manifold is a conjugate gradient algorithm.

#### gol.goal
The gol.goal package contains an implementation of the GOAL similar to what is described in [GOAL]. The results you get with this implementation might differ from the ones you get with the MATLAB implementation. If you configure Theano for the GPU you should get a good speedup over the MATLAB implementation. On my machine I get a speedup of around one magnitude over the MATLAB implementation (GTX 660 GPU).

#### gol.subspace_tracking
The gol.subspace_tracking package contains an implementation of the pROST algorithm as described in [pROST]. 

#### gol.blockmatching
The gol.blockmatching package contains functionality for blockwise image processing.

### GPU support
This package is implemented on top of Theano, therefore the GPU can be used as a computing device.
During the first execution, Theano compiles the symbolically defined computations into C and CUDA code. This can take a while. Subsequent executions are faster because the binaries are cached by Theano.

### Documentation
Currently, there is no other documentation than what you can read here and in the INSTALL.txt file. If you are interested in using the algorithms implemented here in your own project, I recommend looking at gol.subspace_tracking.main.py, gol.goal.GOAL_learning.py and gol.goal.GOAL_denoise.py. The files contain executable examples. The GOAL_*.py files should be executable right away, for main.py the changedetection.net dataset has to be installed and the path to the dataset has to be set in main.py. 

[GOAL]:http://arxiv.org/pdf/1204.5309.pdf
[pROST]:http://arxiv.org/abs/1302.2073
[numpy]:http://www.numpy.org
[scipy]:http://www.scipy.org/
[theano]:http://deeplearning.net/software/theano/
