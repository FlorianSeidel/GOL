from setuptools import setup, Extension
from Cython.Build import cythonize


blockmatching = Extension('blockmatching',
    sources = ['gol.blockmatching.blockmatching.pyx'])

setup(name='gol',
		version='0.1',
		description='A collection of matrix manifold optimization algorithms and a few applications',
		author='Florian Seidel',
		author_email='seidel.florian@gmail.com',
		packages=['gol','gol.optimization','gol.blockmatching','gol.goal','gol.optimization','gol.subspace_tracking',
					'gol.subspace_tracking.utilities'],
		install_requires=['numpy','scipy','theano','nose','Cython','matplotlib'],
		ext_modules=cythonize("gol/blockmatching/blockmatching.pyx"),
		zip_safe=False)
