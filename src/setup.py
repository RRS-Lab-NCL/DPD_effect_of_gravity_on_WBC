from distutils.core import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules = cythonize(["Physics_engine.pyx", "Func_caller.pyx", "NNS.pyx", "NNS_algo.pyx",
                             "Fluid_func.pyx", "Solid_Int_Force_func.pyx", "Solid_func.pyx"],
                            
    annotate = True, language_level = "3"),
    install_requires = ["numpy"]   
)

