"""
Fluid_func.pyx: 
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
cimport numpy as cnp

from cython.parallel cimport prange
from libc.math cimport sqrt
from numpy cimport float64_t, int32_t, bool

# Declare types for arrays used in functions
ctypedef cnp.float64_t dtype_t
ctypedef cnp.int32_t int_dtype_t
ctypedef cnp.npy_bool bool_dtype_t

###############################################################################

# Functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline ipd(double[:, :] pos_i, double[:, ::1] pos_nn):   # Inter particle distance
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = pos_nn.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.zeros((size, 2), dtype = np.float64)
    
    cdef double[:, :] result_view                       = result
    cdef double[:, :] pos_i_view                        = pos_i
    cdef double[:, :] pos_nn_view                       = pos_nn
    
    for i in range(size):
        
        result_view[i, 0]                               = pos_i_view[0, 0] - pos_nn_view[i, 0]
        result_view[i, 1]                               = pos_i_view[0, 1] - pos_nn_view[i, 1]
            
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline ipv(double[:, :] vel_i, double[:, :] vel_nn):   # Inter particle velocity
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = vel_nn.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 2), dtype = np.float64)
    
    cdef double[:, :] result_view                       = result
    cdef double[:, :] vel_i_view                        = vel_i
    cdef double[:, :] vel_nn_view                       = vel_nn
    
    for i in range(size):
        
        result_view[i, 0]                               = vel_i_view[0, 0] - vel_nn_view[i, 0]
        result_view[i, 1]                               = vel_i_view[0, 1] - vel_nn_view[i, 1]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline magnitude(double[:, :] r_ij):                   # Inter particle distance: magnitude
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = r_ij.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 1), dtype = np.float64)
    
    cdef double[:, :] result_view                   = result
    cdef double[:, :] r_ij_view                     = r_ij

    for i in range(size):
        
        result_view[i, 0]                           = sqrt(r_ij_view[i, 0]*r_ij_view[i, 0] + r_ij_view[i, 1]*r_ij_view[i, 1])
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline unitvector(double[:, :] r_ij, double[:, :] mag):  # Unit vector
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = r_ij.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 2), dtype = np.float64)
    
    cdef double[:, :] result_view                   = result
    cdef double[:, :] r_ij_view                     = r_ij
    cdef double[:, :] mag_view                      = mag
    
    for i in range(size):
        
       if mag_view[i, 0] != 0:
           result_view[i, 0]                        = r_ij_view[i, 0]/mag_view[i, 0]
           result_view[i, 1]                        = r_ij_view[i, 1]/mag_view[i, 0]
       else:
           result_view[i, 0]                        = 0.0
           result_view[i, 1]                        = 0.0
           
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dotproduct(double[:, ::1] r_ij, double[:, ::1] uvec):
    
    cdef Py_ssize_t size                                = r_ij.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 1), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] r_ij_view                       = r_ij
    cdef double[:, ::1] uvec_view                       = uvec
    
    for i in range(size):
        
        result_view[i, 0]                               = r_ij_view[i, 0]*uvec_view[i, 0] + r_ij_view[i, 1]*uvec_view[i, 1]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline weights(double[:, ::1] mag, double rc, cnp.ndarray[cnp.npy_bool, ndim = 2] radial_check):
    
    cdef Py_ssize_t size                                = mag.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 1), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] mag_view                        = mag
    cdef double radius                                  = rc
    
    for i in range(size):
        
        result_view[i, 0] = (1 - (mag_view[i, 0]/radius))*radial_check[i, 0]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline conservative_force(double aij, double[:, ::1] wc, double[:, ::1] uvec):
    
    cdef Py_ssize_t size                                = wc.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, uvec.shape[1]), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] wc_view                         = wc
    cdef double[:, ::1] uvec_view                       = uvec
    
    for i in range(size):
        
        result_view[i, 0]                               = aij*wc_view[i, 0]*uvec_view[i, 0]
        result_view[i, 1]                               = aij*wc_view[i, 0]*uvec_view[i, 1]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dissipative_force(double gammaij, double[:, ::1] wd, double[:, ::1] vdp, double[:, ::1] uvec):
    
    cdef Py_ssize_t size                                = wd.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, uvec.shape[1]), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] wd_view                         = wd
    cdef double[:, ::1] uvec_view                       = uvec
    cdef double[:, ::1] vdp_view                        = vdp
    
    for i in range(size):
        
        result_view[i, 0]                               = -gammaij*wd_view[i, 0]*vdp_view[i, 0]*uvec_view[i, 0]
        result_view[i, 1]                               = -gammaij*wd_view[i, 0]*vdp_view[i, 0]*uvec_view[i, 1]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline random_force(double sigmaij, double[:, ::1] wr, double[:, ::1] random_var, double inv_dt, double[:, ::1] uvec):
    
    cdef Py_ssize_t size                                = wr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, uvec.shape[1]), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] wr_view                         = wr
    cdef double[:, ::1] uvec_view                       = uvec
    cdef double[:, ::1] rand_view                       = random_var
    
    for i in range(size):
        
        result_view[i, 0]                               = sigmaij*wr_view[i, 0]*rand_view[i, 0]*inv_dt*uvec_view[i, 0]
        result_view[i, 1]                               = sigmaij*wr_view[i, 0]*rand_view[i, 0]*inv_dt*uvec_view[i, 1]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Fluid_Fluid(nn_fluid, int ncoord,  
                            cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc, 
                            cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc, 
                            double rcff, double aff, double gammaff, 
                            double sigmaff, double inv_dt, cnp.ndarray[cnp.float64_t, ndim = 2] force_Fluid):
    
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] random_var    = np.random.normal(0, 1, (len(nn_fluid), 1))
        
        idx                                                     = nn_fluid
        cdef double[:, :] i_pos                                 = pos_i[:, 0:ncoord]
        cdef double[:, ::1] neigh_pos                           = pos_inc[idx, 0:ncoord]
        
        cdef double[:, :] i_vel                                 = vel_i[:, 0:ncoord]
        cdef double[:, :] neigh_vel                             = vel_inc[idx, 0:ncoord]
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] r_ij          = ipd(i_pos, neigh_pos)
        cdef double[:, :] rij                                   = r_ij
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] mag           = magnitude(rij)
        cdef double[:, :] second_norm                           = mag
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] uvec          = unitvector(rij, second_norm)
        uvec[np.isnan(uvec)]                                    = 0
       
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] v_ij          = ipv(i_vel, neigh_vel)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] vdp           = dotproduct(v_ij, uvec)
       
        cdef cnp.ndarray[cnp.int32_t, ndim = 1] index           = np.arange(len(r_ij), dtype = np.int32)
        rc                                                      = rcff
        cdef cnp.ndarray[cnp.npy_bool,  ndim = 2] radial_check  = (mag[index] <= rc)
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wc            = weights(mag, rc, radial_check)
        wc[wc == 1]                                             = 0
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wr            = wc**0.25
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wd            = wr**2
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fc            = conservative_force(aff, wc, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fd            = dissipative_force(gammaff, wd, vdp, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fr            = random_force(sigmaff, wr, random_var, inv_dt, uvec)
        
        force_Fluid                                             = np.sum(fc + fr + fd, axis = 0).reshape(1, ncoord)
        
        return force_Fluid
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Fluid_Solid(dict Solid, int ncoord, cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, 
                          cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, double rcfs, 
                          double afs, double gammafs, double sigmafs, double inv_dt,
                          Py_ssize_t len_Solid, cnp.ndarray[cnp.float64_t, ndim = 2] force_Solid):
        
        cdef Py_ssize_t k
            
        for k in range(len_Solid):
            
            random_var      = np.random.normal(0, 1, (len(Solid[k].pos), 1))
            
            idx             = np.arange(len(Solid[k].pos))
            i_pos           = pos_i[:, 0:ncoord]
            neigh_pos       = Solid[k].pos_inc[idx, 0:ncoord]
            
            r_ij            = ipd(pos_i[:, 0:ncoord], Solid[k].pos_inc[idx, 0:ncoord])
            mag             = magnitude(r_ij)
           
            uvec            = unitvector(r_ij, mag)
            uvec[np.isnan(uvec)] = 0
           
            i_vel           = vel_i[:, 0:ncoord]
            neigh_vel       = Solid[k].vel_inc[idx, 0:ncoord]
            
            v_ij            = ipv(vel_i[:, 0:ncoord], Solid[k].vel_inc[idx, 0:ncoord])
            vdp             = dotproduct(v_ij, uvec) 
            
            rc              = rcfs
            radial_check    = (mag[idx] <= rc)
            
            wc              = weights(mag, rc, radial_check)
            wc[wc == 1]     = 0
            wr              = wc**0.25
            wd              = wr**2
            
            fc              = conservative_force(afs, wc, uvec)
            fd              = dissipative_force(gammafs, wd, vdp, uvec)
            fr              = random_force(sigmafs, wr, random_var, inv_dt, uvec)
            
            force_Solid     = np.sum(fc + fr + fd, axis = 0).reshape(1, ncoord)
            
        return force_Solid
        
#############################################################################################################################################################################################################################################

def Fluid_Fluid_wrapper(nn_fluid, int ncoord,
                        cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc,
                        cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc,
                        double rcff, double aff, double gammaff,double sigmaff, double inv_dt, 
                        cnp.ndarray[cnp.float64_t, ndim = 2] force_Fluid):
    
    return Fluid_Fluid(nn_fluid, ncoord, pos_i, pos_inc, vel_i, vel_inc, rcff, aff, gammaff, sigmaff, inv_dt, force_Fluid)

def Fluid_Solid_wrapper(dict Solid, int ncoord, cnp.ndarray[cnp.float64_t, ndim = 2] pos_i,
                        cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, double rcfs,
                        double afs, double gammafs, double sigmafs, double inv_dt,                        
                        Py_ssize_t len_Solid, cnp.ndarray[cnp.float64_t, ndim = 2] force_Solid):
    
    return Fluid_Solid(Solid, ncoord, pos_i, vel_i, rcfs, afs, gammafs, sigmafs, inv_dt, len_Solid, force_Solid)

