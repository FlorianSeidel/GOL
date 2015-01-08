'''
Created on Oct 27, 2014

@author: florian
'''
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("gol/blockmatching/blockmatching.pyx"),
)