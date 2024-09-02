# distutils: language = c++

import numpy as np

cimport cython
cimport numpy as cnp

cdef neighbour_cell(cnp.ndarray[cnp.float64_t, ndim = 1] pos_i, dict position_key, object hashmap):
    
    cdef Py_ssize_t i
    
    position_key[0]                                             = hashmap.key(pos_i)

    # Offsets for the 8 cardinal directions
    cdef cnp.ndarray[long, ndim = 2] offsets                    = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]])

    # Calculate positions for the 8 cardinal directions
    positions                                                   = np.array(position_key[0], dtype = np.float64) + offsets
    
    # Assign the calculated positions to position_key
    for i in range(1, 9):
        
        position_key[i] = tuple(positions[i - 1])

    return position_key

def neighbour_cell_wrapper(cnp.ndarray[cnp.float64_t, ndim = 1] pos_i, dict position_key, object hashmap):    
    
    return neighbour_cell(pos_i, position_key, hashmap)
