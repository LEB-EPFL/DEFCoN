import sys
import warnings

# FCNN is the deprecated name for DEFCoN
sys.modules[__name__] = sys.modules['DEFCoN']

#Deprecation warning
warnings.warn('Module name "FCNN" is deprecated. Use "DEFCoN" instead.',
              DeprecationWarning,
              stacklevel=2)