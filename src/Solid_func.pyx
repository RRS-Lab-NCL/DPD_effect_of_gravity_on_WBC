"""
Solid_func.pyx: 
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
from libc.math cimport sqrt, cos, sin, acos
from numpy cimport float64_t, int32_t, bool

# Declare types for arrays used in functions
ctypedef cnp.float64_t dtype_t
ctypedef cnp.int32_t int_dtype_t
ctypedef cnp.npy_bool bool_dtype_t

###############################################################################

# Functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline ipd(double[:, :] pos_i, double[:, :] pos_nn):   # Inter particle distance
    
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
    
    cdef double[:, :] result_view                       = result
    cdef double[:, :] r_ij_view                         = r_ij

    for i in range(size):
        
        result_view[i, 0]                               = sqrt(r_ij_view[i, 0]*r_ij_view[i, 0] + r_ij_view[i, 1]*r_ij_view[i, 1])
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline unitvector(double[:, :] r_ij, double[:, :] mag):  # Unit vector
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = r_ij.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 2), dtype = np.float64)
    
    cdef double[:, :] result_view                       = result
    cdef double[:, :] r_ij_view                         = r_ij
    cdef double[:, :] mag_view                          = mag
    
    for i in range(size):
        if mag_view[i, 0] != 0:
            
            result_view[i, 0]                            = r_ij_view[i, 0]/mag_view[i, 0]
            result_view[i, 1]                            = r_ij_view[i, 1]/mag_view[i, 0]
        else:
            
            result_view[i, 0]                            = 0.0
            result_view[i, 1]                            = 0.0

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
cdef inline dotmultiply(double[:, ::1] r_ij, double[:, ::1] uvec):
    
    cdef Py_ssize_t size                                = r_ij.shape[0]
    cdef Py_ssize_t coord                               = r_ij.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, 2), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] r_ij_view                       = r_ij
    cdef double[:, ::1] uvec_view                       = uvec
    
    for i in range(size):
        for j in range(coord):
            
            result_view[i, j]                           = r_ij_view[i, j]*uvec_view[i, j]
    
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
        
        result_view[i, 0]                               = (1 - (mag_view[i, 0]/radius))*radial_check[i, 0]
    
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
    
    cdef Py_ssize_t i
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
    
    cdef Py_ssize_t i
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
cdef inline dissipative_force_solid(double gamma_t, double gamma_c, double[:, ::1] v_ij, 
                              double[:, ::1] wd, double[:, ::1] vdp,
                              double[:, ::1] uvec, double[:, ::1] vuv):
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size                                = wd.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, uvec.shape[1]), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] wd_view                         = wd
    cdef double[:, ::1] uvec_view                       = uvec
    cdef double[:, ::1] vij_view                        = v_ij
    cdef double[:, ::1] vdp_view                        = vdp
    
    for i in range(size):
        
        result_view[i, 0]                               = (-wd_view[i, 0]*gamma_t*vdp_view[i, 0]*uvec_view[i, 0]) + (-wd_view[i, 0]*gamma_c*(vij_view[i, 0] - vdp_view[i, 0]*uvec_view[i, 0]))
        result_view[i, 1]                               = (-wd_view[i, 0]*gamma_t*vdp_view[i, 0]*uvec_view[i, 1]) + (-wd_view[i, 0]*gamma_c*(vij_view[i, 1] - vdp_view[i, 0]*uvec_view[i, 1]))
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline random_force_solid(double sqr_kBT, double gamma_t, double gamma_c, 
                               double[:, ::1] wr, double[:, ::1] random_var,
                               double[:, ::1] uvec):
    
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t size                                = wr.shape[0]
    cdef Py_ssize_t coord                               = uvec.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] result    = np.empty((size, uvec.shape[1]), dtype = np.float64)
    
    cdef double[:, ::1] result_view                     = result
    cdef double[:, ::1] wr_view                         = wr
    cdef double[:, ::1] uvec_view                       = uvec
    cdef double[:, ::1] rand_view                       = random_var
    
    cdef double sqr_gamma_t                             = sqrt(2*gamma_t)
    cdef double sqr_gamma_ct                            = sqrt(3*gamma_c - gamma_t)
    cdef double matrix_trace                            = trace(rand_view) 
    
    for i in range(size):
        for j in range(coord):
            for k in range(size):
            
                result_view[i, j]                       = (wr_view[i, 0]*sqr_gamma_t*rand_view[i, k] + wr_view[i, 0]*sqr_gamma_ct*matrix_trace)*(sqr_kBT*uvec_view[i, j])
        
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline trace(double[:, :] matrix):
    
    cdef Py_ssize_t i
    cdef double result                                  = 0.0
    cdef Py_ssize_t rows                                = matrix.shape[0]
    cdef Py_ssize_t cols                                = matrix.shape[1]

    if rows != cols:
        return -1.0

    for i in range(rows):
        result                                          += matrix[i, i]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Solid_Solid(int ncoord, double kBT,
                  cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc, 
                  cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc, 
                  double rcss, double ass, double gammass, double sigmass,
                  cnp.ndarray[cnp.float64_t, ndim = 2] force_Solid,
                  double inv_dt, cnp.ndarray[cnp.float64_t, ndim = 2] xi):
    
        cdef double[:, :] i_pos                                 = pos_i[:, 0:ncoord]
        cdef double[:, :] neigh_pos                             = pos_inc[:, 0:ncoord]
        
        cdef double[:, :] i_vel                                 = vel_i[:, 0:ncoord]
        cdef double[:, :] neigh_vel                             = vel_inc[:, 0:ncoord]
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] r_ij          = ipd(i_pos, neigh_pos)
        cdef double[:, :] rij                                   = r_ij
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] mag           = magnitude(rij)
        cdef double[:, :] second_norm                           = mag
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] uvec          = unitvector(rij, second_norm)
        uvec[np.isnan(uvec)]                                    = 0
       
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] v_ij          = ipv(i_vel, neigh_vel)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] vdp           = dotproduct(v_ij, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] vuv           = dotmultiply(vdp, uvec)
       
        cdef cnp.ndarray[cnp.int32_t, ndim = 1] index           = np.arange(len(r_ij), dtype = np.int32)
        rc                                                      = rcss
        cdef cnp.ndarray[cnp.npy_bool,  ndim = 2] radial_check  = (mag[index] <= rc)
        
        cdef cnp.ndarray[cnp.float64_t,  ndim = 2] random_var   = xi
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wc            = weights(mag, rc, radial_check)
        wc[wc == 1]                                             = 0
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wr            = 1.0*radial_check
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wd            = 1.0*radial_check
        
        cdef double aij                                         = ass
        cdef double gamma_c                                     = gammass/3
        cdef double gamma_t                                     = gammass
        cdef double sqr_kBT                                     = sqrt(2*kBT)
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fc            = conservative_force(aij, wc, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fd            = dissipative_force_solid(gamma_t, gamma_c, v_ij, wd, vdp, uvec, vuv)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fr            = random_force_solid(sqr_kBT, gamma_t, gamma_c, wr, random_var, uvec)
        
        force_Solid                                             = np.sum(fc + fr + fd, axis = 0).reshape(1, ncoord)
        
        return force_Solid
        
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Solid_Fluid(nn_fs, int ncoord,
                  cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] fpos_inc,
                  cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] fvel_inc,
                  double rcsf, double asf, double gammasf, double sigmasf, double inv_dt, 
                  cnp.ndarray[cnp.float64_t, ndim = 2] force_Fluid):
    
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] random_var    = np.random.normal(0, 1, (len(nn_fs), 1))
        
        idx                                                     = nn_fs
        cdef double[:, :] i_pos                                 = pos_i[:, 0:ncoord]
        cdef double[:, :] neigh_pos                             = fpos_inc[idx, 0:ncoord]
        
        cdef double[:, :] i_vel                                 = vel_i[:, 0:ncoord]
        cdef double[:, :] neigh_vel                             = fvel_inc[idx, 0:ncoord]
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] r_ij          = ipd(i_pos, neigh_pos)
        cdef double[:, :] rij                                   = r_ij
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] mag           = magnitude(rij)
        cdef double[:, :] second_norm                           = mag
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] uvec          = unitvector(rij, second_norm)
        uvec[np.isnan(uvec)]                                    = 0
       
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] v_ij          = ipv(i_vel, neigh_vel)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] vdp           = dotproduct(v_ij, uvec)
       
        cdef cnp.ndarray[cnp.int32_t, ndim = 1] index           = np.arange(len(r_ij), dtype = np.int32)
        rc                                                      = rcsf
        cdef cnp.ndarray[cnp.npy_bool,  ndim = 2] radial_check  = (mag[index] <= rc)
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wc            = weights(mag, rc, radial_check)
        wc[wc == 1]                                             = 0
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wr            = wc**0.25
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] wd            = wr**2
        
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fc            = conservative_force(asf, wc, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fd            = dissipative_force(gammasf, wd, vdp, uvec)
        cdef cnp.ndarray[cnp.float64_t, ndim = 2] fr            = random_force(sigmasf, wr, random_var, inv_dt, uvec)
        
        force_Fluid                                             = np.sum(fc + fr + fd, axis = 0).reshape(1, ncoord)
        
        return force_Fluid
    
############################################################################################################################################################################

def Solid_Solid_wrapper(int ncoord, double kBT,
                        cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc,
                        cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc,
                        double rcss, double ass, double gammass, double sigmass,
                        cnp.ndarray[cnp.float64_t, ndim = 2] force_Solid, double inv_dt, 
                        cnp.ndarray[cnp.float64_t, ndim = 2] xi):
    
    force_Solid         = Solid_Solid(ncoord, kBT, pos_i, pos_inc, vel_i, vel_inc, 
                                      rcss, ass, gammass, sigmass, force_Solid, inv_dt, xi)
    
    return force_Solid

def Solid_Fluid_wrapper(nn_fs, int ncoord,
                        cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, cnp.ndarray[cnp.float64_t, ndim = 2] fpos_inc,
                        cnp.ndarray[cnp.float64_t, ndim = 2] vel_i, cnp.ndarray[cnp.float64_t, ndim = 2] fvel_inc,
                        double rcsf, double asf, double gammasf, double sigmasf, double inv_dt,
                        cnp.ndarray[cnp.float64_t, ndim = 2] force_Fluid):
    
    force_Fluid         = Solid_Fluid(nn_fs, ncoord, pos_i, fpos_inc, vel_i, fvel_inc, 
                                      rcsf, asf, gammasf, sigmasf, inv_dt, force_Fluid)
    
    return force_Fluid
