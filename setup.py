from setuptools import setup

setup(
    name='DEFCoN',
    version='0.0.0',
    packages=['leb', 'leb.defcon'],
    url='https://github.com/LEB-EPFL/DEFCoN',
    license='LGPL',
    author='Baptiste Ottino',
    author_email='',
    description='Fluorescence spot density estimation for single molecule microscopy.',
    install_requires=[
        'numpy',
        'scikit-image',
        'h5py',
        'tensorflow==1.4.1',
        'keras==2.1.1',
        'pandas'
    ]
)
