# DEFCoN

- [![Join the chat at https://gitter.im/leb_epfl/DEFCoN](https://badges.gitter.im/leb_epfl/DEFCoN.svg)](https://gitter.im/leb_epfl/DEFCoN?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Density Estimation by Fully Convolutional Networks (DEFCoN) - A
fluorescent spot counter for single molecule localization microscopy.

## Description

DEFCoN is a fluorescent spot counter implemented as a fully
convolutional neural network. It is designed with the following
criteria in mind:

- Fast enough for real-time analysis (~10 ms per frame)
- Parameter free predictions
- Trainable on custom datasets

## Installation

### Requirements

**DEFCoN requires Python 3.5, TensorFlow 1.4.1, and Keras
2.1.1**. Please see the setup.py file for the full list of required
packages

### Installing tensorflow-gpu (optional, but recommended)

We recommend using following the directions for installing Tensorflow
to ensure that it is properly configured to use your system's GPU:
https://www.tensorflow.org/install/install_linux#installing_with_anaconda

In particular, we use the following commands from within a Python
virtual environment:

```
pip install --ignore-installed --upgrade tensorflow-gpu==1.4.1
```

### DEFCoN

To install DEFCoN, use the following command from inside DEFCoN's
parent directory:

```
pip install DEFCoN
```

If you wish to develop DEFCoN, we recommend installing as a
development package:

```
pip install -e DEFCoN
```

## Quickstart

DEFCoN is a pre-designed, fully convolutional neural network
architecture that you interact with through a Python API. To get
started, download and run the
[quickstart.ipynb](https://github.com/kmdouglass/DEFCoN/blob/FIRST_RELEASE/examples/quickstart.ipynb)
Jupyter notebook in the [examples
folder](https://github.com/kmdouglass/DEFCoN/tree/FIRST_RELEASE/examples).

## Getting Help

- How to use DEFCoN: https://gitter.im/leb_epfl/DEFCoN
- Bug reports: https://github.com/LEB-EPFL/DEFCoN/issues
- Feature requests: https://github.com/LEB-EPFL/DEFCoN/issues
- Developer questions: https://gitter.im/leb_epfl/DEFCoN

## Acknowledgements

### Software

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/)
- [Jupyter](https://jupyter.org/)
- [scikit-image](http://scikit-image.org/)
- [tifffile](https://pypi.python.org/pypi/tifffile)
- [Pandas](https://pandas.pydata.org/)
 
