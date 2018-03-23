# DEFCoN

- [![Join the chat at https://gitter.im/leb_epfl/DEFCoN](https://badges.gitter.im/leb_epfl/DEFCoN.svg)](https://gitter.im/leb_epfl/DEFCoN?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Density Estimation by Fully Convolutional Networks (DEFCoN) - A
fluorescent spot counter for single molecule localization microscopy.

## Description

The "DEFCoN" folder is a python package containing all the necessary
methods to run DEFCoN. "workflow_example.ipynb" and the "example"
folder can be used to perform a step-by-step standard wrokflow for
DEFCoN, from SASS training images to prediction and serving.

The "FCNN" folder is just a legacy name for the DEFCoN package. I keep
it just to ensure backward compatibility with the rest of my code. It
will issue a warning if the name "FCNN" is imported instead of
"DEFCoN", but it shouldn't be the case for any function in the
package.

"recommended_config.ini" contains the training parameters used in the
thesis. The example config file have fewer epochs and larger batch
size, to run faster.

## Installation

--

## Documentation

--

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
