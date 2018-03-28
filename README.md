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
- Simple API

![An example density map estimate from DEFCoN](images/defcon_demo.png)

## Installation

### Requirements

Please see the setup.py file for the full list of required
packages. Please note that DEFCoN requires Python 3.5, TensorFlow
1.4.1, and Keras 2.1.1, specifically.

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

### Installing tensorflow-gpu (optional, but recommended)

The steps required to set up TensorFlow to use the GPU vary widely
depending on your system and your Python environment. We recommend
reading the following directions for installing TensorFlow to ensure
that it is properly configured to use your system's GPU in your
particular environment:
https://www.tensorflow.org/install/install_linux

In particular, we use the following commands from within a Python
virtual environment **after we have already installed DEFCoN.**

```
pip uninstall tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu==1.4.1
```

For us, uninstalling the CPU version of TensorFlow before installing
the GPU version is necessary to successfully detect the GPU.

You may verify that TensorFlow successfully detects your GPU after
setup by running the following commands from within your Python
interpreter:

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

If you see a string containing something similar to **name:
"/device:GPU:0" device_type: "GPU"** that is printed to the screen,
then you should be good to go.

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

DEFCoN was written by [Baptiste Ottino](https://github.com/bottino) as
a Master's thesis project under the guidance of [Kyle
M. Douglass](https://github.com/kmdouglass) and Suliana Manley in the
[Laboratory of Experimental Biophysics.](https://leb.epfl.ch)

### Software

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/)
- [Jupyter](https://jupyter.org/)
- [scikit-image](http://scikit-image.org/)
- [tifffile](https://pypi.python.org/pypi/tifffile)
- [Pandas](https://pandas.pydata.org/)
 
