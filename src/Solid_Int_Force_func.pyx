"""
Solid_Int_Force_func.pyx: 
	requires class objects as input generated from Fluid.py, Solid_WBC.py, Parameters.py, Force_parameters.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	NIL
	
Compilation: 
	This file is written in cyton extension format, need to be compiled with all its neceesary dependencies.
	
"""

# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

# Import required libraries
import numpy as np

cimport cython
cimport openmp
cimport numpy as np

from cython.parallel cimport prange
from libc.math cimport sqrt, cos, sin, acos
from numpy cimport float64_t, int32_t, bool

# Declare types for arrays used in functions
ctypedef np.float64_t dtype_t
ctypedef np.int32_t int_dtype_t
ctypedef np.npy_bool bool_dtype_t

###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Partial_spring(np.ndarray[np.float64_t, ndim = 2] l, double mag_l):
    
    cdef np.ndarray[double, ndim = 2] result    = np.zeros((2, 2), dtype = np.float64)
    cdef double[:, :] l_view                    = l
    cdef double mag_l_view                      = mag_l
    cdef double[:, :] result_view               = result
    
    result_view[0, 0]                           = l_view[0, 0]/mag_l_view
    result_view[0, 1]                           = -l_view[0, 0]/mag_l_view
    result_view[1, 0]                           = l_view[0, 1]/mag_l_view
    result_view[1, 1]                           = -l_view[0, 1]/mag_l_view
    
    return result    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Partial_area(np.ndarray[np.float64_t, ndim = 2] idx1, double x, double y):
    
    cdef np.ndarray[double, ndim = 2] result    = np.zeros((2, 2), dtype = np.float64)
    cdef double[:, :] result_view               = result
    cdef double[:, :] idx1_view                 = idx1
    
    result_view[0, 0]                           = idx1_view[0, 1] + 0.5*y
    result_view[0, 1]                           = -(idx1_view[0, 1] + 0.5*y)
    result_view[1, 0]                           = x - 0.5*x
    result_view[1, 1]                           = 0.5*x
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Spring_force(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, int ncoord,
                   double m_l0, double kBT, double Lm, double Lp, double SPE, 
                   np.ndarray[np.float64_t, ndim = 2] Force_spring):
    
    cdef Py_ssize_t i
    cdef double m_l, x, E
    cdef np.ndarray[np.float64_t, ndim = 2] idx1, idx2, l, partial_spring, Force
    
    
    
    for i in range(len_shp):
        
        idx1        = np.zeros((1, ncoord), dtype = np.float64)
        idx2        = np.zeros((1, ncoord), dtype = np.float64)
        l           = np.zeros((1, ncoord), dtype = np.float64)
        
        partial_spring  = np.zeros((ncoord, ncoord), dtype = np.float64)
        Force           = np.zeros((ncoord, ncoord), dtype = np.float64)
        
        for j in range(ncoord):
            
            idx1[0, j]            = pos_inc[i, j]
            if i == (len_shp-1):            
                idx2[0, j]        = pos_inc[0, j]
                
            else:            
                idx2[0, j]        = pos_inc[i+1, j]

            l[0, j]               = idx1[0, j] - idx2[0, j]
            
        m_l             = sqrt(l[0, 0]*l[0, 0] + l[0, 1]*l[0, 1])
        x               = m_l/m_l0

        ## Potential Energy
        E               = (kBT*Lm/(4*Lp))*((3*x**2 - 2*x**3)/(1 - x))
        SPE             += E

        ## Force
        partial_spring  = Partial_spring(l, m_l)
        Force           = -(kBT/(4*Lp))*(4*x - 1 + (1/(1 - x)**2))*partial_spring

        if i != (len_shp-1):
            
            Force_spring[i, 0:ncoord]       += Force[:, 0]
            Force_spring[i + 1, 0:ncoord]   += Force[:, 1]
            
        else:
            
            Force_spring[-1, 0:ncoord]      += Force[:, 0]
            Force_spring[0, 0:ncoord]       += Force[:, 1]

    return Force_spring, SPE

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Bending_force(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, 
                    int ncoord, double BPE, double kb, 
                    np.ndarray[np.float64_t, ndim = 2] Force_bend):    
    
    for i in range(len_shp):
        
        if i == 0:
            
            idx1            = pos_inc[-1, 0:ncoord]
            idx2            = pos_inc[i, 0:ncoord]
            idx3            = pos_inc[i+1, 0:ncoord]
            
        elif i == (len_shp-1):
            
            idx1            = pos_inc[i-1, 0:ncoord]
            idx2            = pos_inc[-1, 0:ncoord]
            idx3            = pos_inc[0, 0:ncoord]
            
        else:
            
            idx1            = pos_inc[i-1, 0:ncoord]
            idx2            = pos_inc[i, 0:ncoord]
            idx3            = pos_inc[i+1, 0:ncoord]

        l12                 = idx1 - idx2
        l23                 = idx2 - idx3

        m_l12               = np.linalg.norm(l12)
        m_l23               = np.linalg.norm(l23)
        
        A, B                = idx1, idx2
        vec1                = B - A
        normal1             = np.array([-vec1[1], vec1[0]])
        
        A, B                = idx2, idx3
        vec2                = B - A
        normal2             = np.array([-vec2[1], vec2[0]])
        
        normal_avg          = 0.5*(normal1 + normal2)
        unit_normal         = normal_avg/(sqrt(normal_avg[0]*normal_avg[0] + normal_avg[1]*normal_avg[1]))

        midpoint            = (idx1 + idx2 + idx3)/3
        mp_norm             = midpoint/(sqrt(midpoint[0]*midpoint[0] + midpoint[1]*midpoint[1]))

        S1                  = sqrt((mp_norm[0] - unit_normal[0])**2 + (mp_norm[1] - unit_normal[1])**2)
        S2                  = sqrt((mp_norm[0] + unit_normal[0])**2 + (mp_norm[1] + unit_normal[1])**2)
        
        if S1 > S2:
            theta           = acos((l12[0]*l23[0] + l12[1]*l23[1])/(m_l12*m_l23))
        elif S1 < S2:
            theta           = -acos((l12[0]*l23[0] + l12[1]*l23[1])/(m_l12*m_l23))

        x                   = l12[0]
        y                   = l12[1]
        partial_spring12    = np.array([[x/m_l12, -x/m_l12, 0], [y/m_l12, -y/m_l12, 0]], dtype = np.float64)

        x                   = l23[0]
        y                   = l23[1]
        partial_spring23    = np.array([[0, x/m_l23, -x/m_l23], [0, y/m_l23, -y/m_l23]], dtype = np.float64)

        E                   = kb*(1 - cos(theta))
        BPE                 += E

        a                   = l12[0]*l23[0] + l12[1]*l23[1]
        b                   = m_l12*m_l23
        first_derv          = (-1)/sqrt(1 - (a/b)**2)
        partial_a           = np.array([[l23[0], (-l23[0] + l12[0]), -l12[0]], 
                                        [l23[1], (-l23[1] + l12[1]), -l12[1]]], dtype = np.float64)
        
        partial_b           = (m_l23*partial_spring12) + (m_l12*partial_spring23)
        chain_derv          = (b*partial_a - a*partial_b)/b**2
        
        if S1 > S2:
            partial_theta   = first_derv*chain_derv
        elif S1 < S2:
            partial_theta   = -first_derv*chain_derv

        theta0              = 0
        sine                = sin(theta)
        cosine              = cos(theta)
        Force               = -kb*((cos(theta0)*sine*partial_theta - sin(theta0)*cosine*partial_theta))

        if i == 0:
            
            Force_bend[-1, 0:ncoord]    += Force[:, 0]
            Force_bend[i, 0:ncoord]     += Force[:, 1]
            Force_bend[i + 1, 0:ncoord] += Force[:, 2]
            
        elif i == (len_shp-1):
            
            Force_bend[i - 1, 0:ncoord] += Force[:, 0]
            Force_bend[-1, 0:ncoord]    += Force[:, 1]
            Force_bend[0, 0:ncoord]     += Force[:, 2]
            
        else:
            
            Force_bend[i - 1, 0:ncoord] += Force[:, 0]
            Force_bend[i, 0:ncoord]     += Force[:, 1]
            Force_bend[i + 1, 0:ncoord] += Force[:, 2]

    return Force_bend, BPE

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Area_force(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, 
                 int ncoord, double APE, double A_ref, double ka,
                 np.ndarray[np.float64_t, ndim = 2] Force_area):
    
    cdef Py_ssize_t i, j    
    cdef double x, y
    cdef double Ar = 0.0
    cdef np.ndarray[double, ndim = 2] idx1, idx2, partial_area, Force        
    
    idx1        = np.zeros((1, ncoord), dtype = np.float64)
    idx2        = np.zeros((1, ncoord), dtype = np.float64)    
    
    for i in range(len_shp):
        
        for j in range(ncoord):
        
            if i == (len_shp-1):
                
                idx1[0, j]  = pos_inc[i, j]
                idx2[0, j]  = pos_inc[0, j]
                
            else:
                
                idx1[0, j]  = pos_inc[i, j]
                idx2[0, j]  = pos_inc[i+1, j]

        x               = idx1[0, 0] - idx2[0, 0]
        y               = idx2[0, 1] - idx1[0, 1]
        Ar              += idx1[0, 1]*x + 0.5*(x*y)

    APE                 = 0.5*ka*((Ar - A_ref)/A_ref)**2
    
    for i in range(len_shp):
        
        for j in range(ncoord):
        
            if i == (len_shp-1):
                
                idx1[0, j]  = pos_inc[i, j]
                idx2[0, j]  = pos_inc[0, j]
                
            else:
                
                idx1[0, j]  = pos_inc[i, j]
                idx2[0, j]  = pos_inc[i+1, j]

        x               = idx1[0, 0] - idx2[0, 0]
        y               = idx2[0, 1] - idx1[0, 1]
        
        partial_area    = Partial_area(idx1, x, y)
        Force           = -ka*((Ar - A_ref)/A_ref)*partial_area

        if i == (len_shp-1):
            
            Force_area[i, 0:ncoord]      += Force[:, 0]
            Force_area[0, 0:ncoord]      += Force[:, 1]
            
        else:
            
            Force_area[i, 0:ncoord]      += Force[:, 0]
            Force_area[i + 1, 0:ncoord]  += Force[:, 1]
            
    return Force_area, APE

#############################################################################################################################################################################################################################################

def Spring_force_wrapper(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, int ncoord,
                         double m_l0, double kBT, double Lm, double Lp, double SPE,
                         np.ndarray[np.float64_t, ndim = 2] Force_spring):
    
    Force_spring, SPE       = Spring_force(len_shp, pos_inc, ncoord, m_l0, kBT, Lm, 
                                           Lp, SPE, Force_spring)
    
    return Force_spring, SPE

def Bending_force_wrapper(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, int ncoord, 
                          double BPE, double kb, np.ndarray[np.float64_t, ndim = 2] Force_bend):
    
    Force_bend, BPE         = Bending_force(len_shp, pos_inc, ncoord, BPE, kb, Force_bend)

    
    return Force_bend, BPE

def Area_force_wrapper(int len_shp, np.ndarray[np.float64_t, ndim = 2] pos_inc, int ncoord, 
                       double APE, double A_ref, double ka, np.ndarray[np.float64_t, ndim = 2] Force_area):
    
    Force_area, APE         = Area_force(len_shp, pos_inc, ncoord, APE, A_ref, ka, Force_area)
    
    return Force_area, APE
