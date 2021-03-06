try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='DEFCoN',
    version='0.1.0',
    packages=['defcon'],
    url='https://github.com/LEB-EPFL/DEFCoN',
    license='BSD3',
    author='Baptiste Ottino, Kyle M. Douglass',
    author_email='kyle.douglass@epfl.ch',
    description='Fluorescence spot density estimation for single '
                'molecule microscopy.',
    install_requires=['numpy',
                      'scikit-image',
                      'h5py',
                      'pandas',
                      'tifffile',
                      'tensorflow==1.3.0',
                      'keras==2.0.8'],
    package_data={'': ['*.h5']}
)

