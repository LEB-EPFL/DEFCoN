# DEFCoN
The "DEFCoN" folder is a python package containing all the necessary methods to run DEFCoN. "workflow_example.ipynb" and the "example" folder can be used to perform a step-by-step standard wrokflow for DEFCoN, from SASS training images to prediction and serving.

The "FCNN" folder is just a legacy name for the DEFCoN package. I keep it just to ensure backward compatibility with the rest of my code. It will issue a warning if the name "FCNN" is imported instead of "DEFCoN", but it shouldn't be the case for any function in the package.

"recommended_config.ini" contains the training parameters used in the thesis. The example config file have fewer epochs and larger batch size, to run faster.