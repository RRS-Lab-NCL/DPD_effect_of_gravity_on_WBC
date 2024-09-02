"""
NNS_algo.pyx: 
	requires class objects as input generated from Fluid.py, Solid_WBC.py, Parameters.py, Force_parameters.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	NIL
	
Compilation: 
	This file is written in cyton extension format, need to be compiled with all its neceesary dependencies.
	
"""
# distutils: language = c++

import numpy as np

cimport cython
cimport numpy as cnp

from Hashmap import from_points

cdef Particle_hashmap(object Force_params, cnp.ndarray[cnp.float64_t, ndim = 2] sim_object): 
    
    cdef double rc                                      = Force_params.rc
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] point     = sim_object
    
    hashmap                                             = from_points(cell_size = rc, points = point)
    
    return hashmap

def Particle_hashmap_wrapper(object Force_params, cnp.ndarray[cnp.float64_t, ndim = 2] sim_object):
    
    hashmap                                             = Particle_hashmap(Force_params, sim_object)
    
    return hashmap
