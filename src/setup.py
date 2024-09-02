from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import glob

# Automatically find all .pyx files in the directory
cython_modules     = glob.glob("*.pyx")

# Define the extensions
extensions         = [Extension(name = module.replace(".pyx", ""),  # Remove the .pyx suffix for the module name
                                sources = [module],
                                include_dirs = [np.get_include()], )
                      for module in cython_modules]

# Setup configuration
setup(name = "DPD_WBC", ext_modules = cythonize(extensions),)
