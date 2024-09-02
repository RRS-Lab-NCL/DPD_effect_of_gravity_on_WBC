"""
Func_caller.pyx: 
	requires class objects as input generated from Fluid.py, Solid_WBC.py, Parameters.py, Force_parameters.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	pymp-pypi, "Fluid_func.pyx", "Solid_func.pyx", "Solid_Int_Force_func.pyx", "NNS_algo.pyx", "NNS.pyx"
	
Compilation: 
	This file is written in cyton extension format, need to be compiled with all its neceesary dependencies.
	
"""

# distutils: language = c++

# Import required libraries
import numpy as np
import pymp
import Fluid_func
import Solid_Int_Force_func
import Solid_func
import NNS_algo
import NNS

from multiprocessing import cpu_count

cimport cython
cimport openmp
cimport numpy as cnp

from cython.parallel import prange, parallel
from libc.math cimport sqrt
from numpy cimport float64_t, int32_t, bool

# Declare types for arrays used in functions
ctypedef cnp.float64_t dtype_t
ctypedef cnp.int32_t int_dtype_t
ctypedef cnp.npy_bool bool_dtype_t

########################################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef nn_search(cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc_view, int i, object NN_fluid):
        
    # hashmap creation    
    cdef list valid_neighbs     = []
    cdef dict nn_key            = {}
        
    pos                         = pos_inc_view[i, :]
    
    # Identifying keys of 8 cardinal directions from the particle
    nn_key                      = NNS.neighbour_cell_wrapper(pos, {}, NN_fluid)

    # Fluid neighbor identification
    valid_neighbs               = [NN_fluid.query(nn_key[k]) for k in range(len(nn_key)) if len(NN_fluid.query(nn_key[k])) > 0]
    nn_fluid                    = np.concatenate(valid_neighbs, axis = 0) if valid_neighbs else np.array([], dtype = np.float64)

    return nn_fluid

########################################################################################################################################

## Fluid

@cython.boundscheck(False)
@cython.wraparound(False)
# Velocity verlet scheme 1
cdef object vv_update1(object Fluid, object Parameters):
    
    """
    Incrementing particle velocity, positon and accleration based on the 
    VV scheme
    v(i + 0.5)      = v(i) + lambda*acc(i)*dt
    pos(i + 1)      = pos(i) + v(i + 0.5)*dt + 0.5*acc(i)*dt**2
    acc(i + 0.5)    = acc(i)
    """  
    
    cdef object params          = Parameters
    cdef int ncoord             = params.ncoord
    cdef double lmbda           = params.lmbda
    cdef double dt              = params.dt
    cdef double dt_sqr          = dt*dt
    cdef double lambda_dt       = lmbda*dt

    # Create NumPy arrays for pos_inc, vel_inc, acc_inc, and key
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos       = Fluid.pos
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel       = Fluid.vel
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] acc       = Fluid.acc
    
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc   = np.zeros(Fluid.pos.shape, dtype = np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc   = np.zeros(Fluid.vel.shape, dtype = np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] acc_inc   = np.zeros(Fluid.acc.shape, dtype = np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] key       = np.arange(len(Fluid.pos), dtype = np.float64)
        
    vel_inc[:, :ncoord]         = vel[:, :ncoord] + lambda_dt*acc[:, :ncoord]
    pos_inc[:, :ncoord]         = pos[:, :ncoord] + vel_inc[:, :ncoord]*dt + 0.5*acc[:, :ncoord]*dt_sqr
            
    pos_inc[:, -1]              = key
    vel_inc[:, -1]              = key
    acc_inc[:, -1]              = key

    Fluid.pos_inc               = pos_inc
    Fluid.vel_inc               = vel_inc
    Fluid.acc_inc               = acc_inc

    return Fluid

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object compute_forces(object Fluid, object WBC, object RBC, object Parameters, 
                           object Fcalc_params):
    
    """
    acc(i + 1) = Sum(Fc + Fr + Fd)
    Fc = aij*wc*unit_vector
    Fd = -gamma*wd*(v_ij.unit_vector)*unit_vector
    Fr = sigma*wr*eta(x,i)*dt**(-0.5)*unit_vector
    """
    
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, vel_i, random_var
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] fc, fr, fd, uvec
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] mag, vdp
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] wc, wr, wd
    cdef cnp.ndarray[cnp.npy_bool, ndim = 2] radial_check
    cdef double rc
    
    cdef dict Solid_WBC     = WBC
    cdef dict Solid_RBC     = RBC
    cdef object params      = Parameters
    cdef object fcalc_par   = Fcalc_params
    
    cdef int ncoord         = params.ncoord
    cdef double rcff        = fcalc_par.rc
    cdef double rcfs        = fcalc_par.rcfs
    cdef double inv_dt      = fcalc_par.inv_dt
    cdef double aff         = fcalc_par.aij
    cdef double afs         = fcalc_par.afs
    cdef double gammaff     = fcalc_par.gammaij
    cdef double gammafs     = fcalc_par.gammafs
    cdef double sigmaff     = fcalc_par.sigmaij
    cdef double sigmafs     = fcalc_par.sigmafs
    
    # Access the shape of vectors and allocate the for loop variable
    cdef Py_ssize_t i, j
    cdef Py_ssize_t len_shp = Fluid.pos_inc.shape[0]
    cdef Py_ssize_t wid_shp = Fluid.pos_inc.shape[1]
    cdef Py_ssize_t len_WBC = len(Solid_WBC)
    cdef Py_ssize_t len_RBC = len(Solid_RBC)
    
    # CPU allocation
    ncpus                   = cpu_count()
    
    # shared array creation for multi-processor
    tarr                    = pymp.shared.array((len_shp, ncoord), dtype = np.float64)
    
    # Array creation
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc_view  = Fluid.pos_inc
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc_view  = Fluid.vel_inc
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] force_fluid   = np.zeros((1, ncoord))
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] force_WBC     = np.zeros((1, ncoord))
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] force_RBC     = np.zeros((1, ncoord))
        
    # Nearest neighbour search
    cdef object NN_fluid    = NNS_algo.Particle_hashmap_wrapper(fcalc_par, pos_inc_view)
    
    # force calculation
    with pymp.Parallel(ncpus) as p:
    
        for i in p.range(len_shp):
                
            pos             = pos_inc_view[i, :]
            pos_i           = pos_inc_view[i, :ncoord].reshape(1, ncoord)
            vel_i           = vel_inc_view[i, :ncoord].reshape(1, ncoord)
            
            ######################################################################################
            
            # Identifying keys of 8 cardinal directions from the particle
            nn_fluid        = nn_search(pos_inc_view, i, NN_fluid)
                
            ######################################################################################
                
            ## fluid - fluid
            force_fluid     = Fluid_func.Fluid_Fluid_wrapper(nn_fluid, ncoord, pos_i, pos_inc_view,
                                          vel_i, vel_inc_view, rcff, aff, gammaff, sigmaff, inv_dt, 
                                          force_fluid)
            
            ######################################################################################
            
            ## fluid - solid
            force_WBC       = Fluid_func.Fluid_Solid_wrapper(Solid_WBC, ncoord, pos_i, vel_i, rcfs, 
                                                  afs, gammafs, sigmafs, inv_dt, len_WBC, force_WBC)
            
            ######################################################################################
            
            ## fluid - solid
            force_RBC       = Fluid_func.Fluid_Solid_wrapper(Solid_RBC, ncoord, pos_i, vel_i, rcfs, 
                                                  afs, gammafs, sigmafs, inv_dt, len_RBC, force_RBC)            
            
            ######################################################################################            
                
            tarr[i, :]      	= force_fluid + force_WBC + force_RBC
    
    Fluid.acc_inc[:, 0:ncoord]  	    = tarr
    
    return Fluid

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object vv_update2(Fluid, object Parameters, int ts):
    
    """
    pvak = position, velocity, accln and key
    pvak_inc = inc(position, velocity, accln and key)
    """
    cdef int i, j
    cdef object params      = Parameters
    
    cdef int ncoord         = params.ncoord
    cdef int len_shp        = len(Fluid.pos)
    cdef double dt          = params.dt
    cdef double length      = params.length
    cdef double width       = params.width
    cdef double elipson     = 1e-6
            
    cdef cnp.ndarray[cnp.npy_bool,  ndim = 1] inlet, outlet
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] force_ext     = params.force_ext
    
    # Create NumPy arrays for pos_inc, vel_inc, acc_inc, and key
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos           = Fluid.pos
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel           = Fluid.vel
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] acc           = Fluid.acc
    
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc       = Fluid.pos_inc
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc       = Fluid.vel_inc
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] acc_inc       = Fluid.acc_inc 
    
    # Acceleration updation
    acc_inc[:, 0:ncoord]    += force_ext[0, 0:ncoord]
    
    # Incremental Velocity updation
    vel_inc[:, 0:ncoord]    = vel[:, 0:ncoord] + 0.5*(acc[:, 0:ncoord] + acc_inc[:, 0:ncoord])*dt
    
    # Initial Velocity updation
    vel[:, 0:ncoord]        = vel_inc[:, 0:ncoord] - 0.001*np.mean(vel_inc[:, 0:ncoord], 0)
    
    # Open BC
    # inlet
    inlet                   = pos_inc[:, 0] < 0
    pos_inc[inlet, 0]       += length
    
    # outlet
    outlet                  = pos_inc[:, 0] > length
    pos_inc[outlet, 0]      = np.abs(length - pos_inc[outlet, 0])
            
    pos[:, 0:ncoord]        = pos_inc[:, 0:ncoord]
    acc[:, 0:ncoord]        = acc_inc[:, 0:ncoord]
            
    # bottom wall boundary
    cdef bint[:] bot_part   = (pos[:, 1] < elipson).astype(np.int32)
    cdef int len_bot        = len(bot_part)
    cdef double n           = 1
    cdef double[:, :] vRx   = np.random.normal(0, 1, (len_bot, 1))
    cdef double[:, :] vRy   = np.random.normal(0, 1, (len_bot, 1))
    
    for i in range(len_bot):
        
        if bot_part[i]:
            surface_normal      = np.array([0, -1])
            magnitude           = (pos[i, 0]**2 + pos[i, 1]**2)**0.5
            
            dt_ac               = pos[i, :ncoord]/magnitude
            reflect_pt          = 2*np.dot(surface_normal, dt_ac)*surface_normal
            
            pos[i, 0]           = pos[i, 0] - reflect_pt[0]
            pos[i, 1]           = pos[i, 1] - reflect_pt[1]
            
            vel[i, 0]           = vRx[i, 0]
            vel[i, 1]           = vRy[i, 0] + n*(sqrt((n*vRy[i, 0]) ** 2) - (n*vRy[i, 0]))

    # top wall boundary
    cdef bint[:] top_part   = (pos[:, 1] > (width + elipson)).astype(np.int32)
    cdef int len_top        = len(top_part)
    n                       = -1
    vRx                     = np.random.normal(0, 1, (len_top, 1))
    vRy                     = np.random.normal(0, 1, (len_top, 1))
    
    for i in range(len_top):
        
        if top_part[i]:
            surface_normal      = np.array([0, -1])
            magnitude           = (pos[i, 0]**2 + pos[i, 1]**2)**0.5
            
            dt_ac               = pos[i, :ncoord]/magnitude
            reflect_pt          = 2*np.dot(surface_normal, dt_ac)*surface_normal
            
            pos[i, 0]           = pos[i, 0] - reflect_pt[0]
            pos[i, 1]           = pos[i, 1] - reflect_pt[1]
            
            vel[i, 0]           = vRx[i, 0]
            vel[i, 1]           = vRy[i, 0] + n*(sqrt((n*vRy[i, 0]) ** 2) - (n*vRy[i, 0]))

    # system temperature and KE
    cdef double sum_vel_sq  = 0
    
    for i in range(vel.shape[0]):
        
        for j in range(ncoord):
            
            sum_vel_sq      += vel[i, j] ** 2
            
    Fluid.pos               = pos
    Fluid.pos_inc           = pos_inc
    Fluid.vel               = vel
    Fluid.vel_inc           = vel_inc
    Fluid.acc               = acc
    Fluid.acc_inc           = acc_inc
            
    Fluid.kBT[ts]           = (1/3)*sum_vel_sq/(len_shp*ncoord - 3)
    Fluid.KE[ts]            = (1/2)*sum_vel_sq/(len_shp*ncoord)
    
    return Fluid

########################################################################################################################################

## Solid 

@cython.boundscheck(False)
@cython.wraparound(False)
# Velocity verlet scheme 1
cdef object vv_update1_solid(object Solid, object Parameters):
    
    cdef object params      = Parameters
    
    cdef Py_ssize_t i    
    cdef Py_ssize_t nsolids = len(Solid)
    
    cdef int ncoord         = params.ncoord
    cdef int len_shp        = len(Solid[0].pos)
    cdef double lmbda       = params.lmbda
    cdef double dt          = params.dt
    cdef double dt_sqr      = dt*dt
    cdef double lambda_dt   = lmbda*dt
    
    # Create NumPy arrays for pos_inc, vel_inc, acc_inc, and key
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc   = np.zeros(Solid[0].pos.shape, dtype = np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] vel_inc   = np.zeros(Solid[0].vel.shape, dtype = np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] acc_inc   = np.zeros(Solid[0].acc.shape, dtype = np.float64)    
    cdef cnp.ndarray[cnp.float64_t, ndim = 1] key       = np.arange(len_shp, dtype = np.float64)
    
    for i in range(nsolids):
        
        for j in range(len_shp):
        
            vel_inc[j, 0:ncoord]    = Solid[i].vel[j, 0:ncoord] + lambda_dt*Solid[i].acc[j, 0:ncoord]
            pos_inc[j, 0:ncoord]    = Solid[i].pos[j, 0:ncoord] + vel_inc[j, 0:ncoord]*dt + 0.5*Solid[i].acc[j, 0:ncoord]*dt_sqr
        
            vel_inc[:, -1]          = key[:]
            pos_inc[:, -1]          = key[:]
            acc_inc[:, -1]          = key[:]
    
        Solid[i].vel_inc         = vel_inc
        Solid[i].pos_inc         = pos_inc
        Solid[i].acc_inc         = acc_inc
        
    return Solid

@cython.boundscheck(False)
@cython.wraparound(False)
# Internal forces of solid
cdef object internal_forces(object Solid, object Parameters):
    
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nsolids = len(Solid)
    
    cdef object params      = Parameters
    cdef int ncoord         = params.ncoord
    cdef double kBT         = params.kBT
    
  ########################################################################################################################################
    
    ## Spring
    cdef double Lp          = Solid[0].Lp
    cdef double Lm          = Solid[0].Lm
    cdef double m_l0        = Solid[0].m_l0
    cdef double SPE         = 0.0
    
    ## Bending
    cdef double kb          = Solid[0].kb/len(Solid[0].pos)
    cdef double BPE         = 0.0
    
    ## Area
    cdef double ka          = Solid[0].ka
    cdef double A_ref       = Solid[0].A_ref
    cdef double APE         = 0.0
    
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] Force_spring, Force_bend, Force_area 
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc
    
    cdef Py_ssize_t len_shp         = Solid[0].pos_inc.shape[0]
    cdef Py_ssize_t wid_shp         = Solid[0].pos_inc.shape[1]

    for j in range(nsolids):
        
        pos_inc                     = Solid[j].pos_inc[:, 0:ncoord]
        
        Force_spring                = np.zeros((len_shp, ncoord), dtype = np.float64)
        Force_bend                  = np.zeros((len_shp, ncoord), dtype = np.float64)
        Force_area                  = np.zeros((len_shp, ncoord), dtype = np.float64)
        
        ######################################################################################
        
        Force_spring, SPE           = Solid_Int_Force_func.Spring_force_wrapper(len_shp, pos_inc, ncoord,
                                                                    m_l0, kBT, Lm, Lp, SPE, 
                                                                    Force_spring)
        
        ######################################################################################
                
        Force_bend, BPE             = Solid_Int_Force_func.Bending_force_wrapper(len_shp, pos_inc, ncoord, 
                                                                    BPE, kb, Force_bend)
        
        ######################################################################################
        
        Force_area, APE             = Solid_Int_Force_func.Area_force_wrapper(len_shp, pos_inc, ncoord, 
                                                                APE, A_ref, ka, Force_area)
        
        ######################################################################################
        
        Solid[j].APE                = APE
        Solid[j].BPE                = BPE
        Solid[j].SPE                = SPE
        Solid[j].Internal_energy    = APE + BPE + SPE
        Solid[j].Internal_forces    = Force_spring + Force_bend + Force_area
            
    return Solid

@cython.boundscheck(False)
@cython.wraparound(False)
# Forces on particles
cdef object compute_forces_solid(object Solid, object Fluid, object Parameters,
                                 object Fcalc_params, cnp.ndarray[cnp.float64_t, ndim = 2] xi):
    
    """
    acc(i + 1) = Sum(Fc + Fr + Fd)
    Fc = aij*wc*unit_vector
    Fd = -gamma*wd*(v_ij.unit_vector)*unit_vector
    Fr = sigma*wr*eta(x,i)*dt**(-2)*unit_vector
    """
      
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_i, vel_i, random_var
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] fc, fr, fd, uvec
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] force_fluid, force_RBC
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] mag, vdp
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] wc, wr, wd
    cdef cnp.ndarray[cnp.npy_bool, ndim = 2] radial_check
    cdef cnp.ndarray[cnp.float64_t, ndim = 2] pos_inc, vel_inc
    cdef double rc
    
    cdef object fluid       = Fluid
    cdef object params      = Parameters
    cdef object fcalc_par   = Fcalc_params
    
    cdef int ncoord         = params.ncoord
    cdef int kBT            = params.kBT        
    cdef double rcss        = fcalc_par.rc
    cdef double rcsf        = fcalc_par.rcfs
    cdef double inv_dt      = fcalc_par.inv_dt
    cdef double ass         = fcalc_par.aij
    cdef double asf         = fcalc_par.afs
    cdef double gammass     = fcalc_par.gammaij
    cdef double gammasf     = fcalc_par.gammafs
    cdef double sigmass     = fcalc_par.sigmaij
    cdef double sigmasf     = fcalc_par.sigmafs
    
    # CPU allocation
    ncpus                       = cpu_count()
    
    # Access the shape of vectors and allocate the for loop variable
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nsolids     = len(Solid) 
    cdef Py_ssize_t len_shp     = Solid[0].pos_inc.shape[0]
    cdef Py_ssize_t wid_shp     = Solid[0].pos_inc.shape[1]
    
    # Nearest neighbour search
    cdef object NN_fs           = NNS_algo.Particle_hashmap_wrapper(fcalc_par, fluid.pos_inc)
    
    xi                          = np.random.permutation(xi)
    xi                          = np.triu(xi.T, 1) + np.tril(xi)
    
    force_solid                 = np.zeros((1, ncoord), dtype = np.float64)
    force_fluid                 = np.zeros((1, ncoord), dtype = np.float64)
        
    for j in range(nsolids):
        
        # Array creation
        pos_inc                 = Solid[j].pos_inc
        vel_inc                 = Solid[j].vel_inc
        
        fpos_inc                = fluid.pos_inc
        fvel_inc                = fluid.vel_inc
        tarr                    = np.zeros((len_shp, ncoord), dtype = np.float64)
                        
        for i in range(len_shp):
            
            # force calculation
            pos             = pos_inc[i, :]
            pos_i           = pos_inc[i, 0:ncoord].reshape(1, ncoord)
            vel_i           = vel_inc[i, 0:ncoord].reshape(1, ncoord)
            
            ######################################################################################
            
            # Nearest neighbour search
            nn_fs           = nn_search(pos_inc, i, NN_fs)
            
            ######################################################################################
            
            force_solid     = Solid_func.Solid_Solid_wrapper(ncoord, kBT, pos_i, pos_inc, vel_i, vel_inc, 
                                                             rcss, ass, gammass, sigmass, force_solid, 
                                                             inv_dt, xi)
            
            ######################################################################################
            
            force_fluid     = Solid_func.Solid_Fluid_wrapper(nn_fs, ncoord, pos_i, fpos_inc, vel_i, 
                                                             fvel_inc, rcsf, asf, gammasf, sigmasf, 
                                                             inv_dt, force_fluid)
            
            ######################################################################################
            
            # summation of solid and fluid forces
            tarr[i, 0:Parameters.ncoord]      = force_solid + force_fluid
            
        Solid[j].acc_inc[:, 0:Parameters.ncoord]    = tarr[:, 0:Parameters.ncoord]
    
    return Solid

@cython.boundscheck(False)
@cython.wraparound(False)
# Forces on particles
cdef object vv_update2_solid(object Solid, object Parameters):
    
    cdef object params      = Parameters
    
    cdef int ncoord         = params.ncoord
    cdef double dt          = params.dt
    cdef double length      = params.length
    cdef double width       = params.width
    cdef double elipson     = 1e-4
    cdef Py_ssize_t nsol    = len(Solid) 
    
    force_ext_view                      = params.force_ext
    cdef double[:, ::1] reflect_pt      = np.zeros((len(Solid[0].pos), ncoord))
    
    for j in range(nsol):
        
        # Acceleration updation
        Solid[j].acc_inc[:, :ncoord]    += (Solid[j].Internal_forces[:, :ncoord] + force_ext_view)
        
        # Incremental Velocity updation
        Solid[j].vel_inc[:, :ncoord]    = Solid[j].vel[:, :ncoord] + 0.5*(Solid[j].acc[:, :ncoord] + Solid[j].acc_inc[:, :ncoord])*dt
        
        # Initial Velocity updation
        Solid[j].vel[:, :ncoord]        = Solid[j].vel_inc[:, :ncoord]
        
        # Position reassignment
        Solid[j].pos                    = Solid[j].pos_inc
        
        # Acceleration reassignment
        Solid[j].acc                    = Solid[j].acc_inc
        
        # bottom wall boundary
        check                           = Solid[j].pos[:, 1] < elipson
        if np.sum(check) != 0:
            dt_ac                       = (Solid[j].pos[:, 1] - elipson)/Solid[j].vel[:, 1]
            Solid[j].vel[:, 1]          = -Solid[j].vel[:, 1]                                   # reverse normal component of velocity
            Solid[j].pos[:, 1]          = dt_ac*Solid[j].vel[:, 1]

        # top wall boundary
        check                           = Solid[j].pos[:, 1] > (Parameters.width + elipson)
        if np.sum(check) != 0:
            dt_ac                       = (Solid[j].pos[:, 1] - (Parameters.width + elipson))/Solid[j].vel[:, 1]
            Solid[j].vel[:, 1]          = -Solid[j].vel[:, 1]                                   # reverse normal component of velocity
            Solid[j].pos[:, 1]          = dt_ac*Solid[j].vel[:, 1]
        
        # Periodic BC
        if (np.sum(Solid[j].pos[:, 0] > Parameters.length) == len(Solid[j].pos)):
            Solid[j].pos = np.abs(Solid[j].pos - Parameters.length)
            
        elif (np.sum(Solid[j].pos[:, 0] < Parameters.length) == len(Solid[j].pos)):
            Solid[j].pos = np.abs(Solid[j].pos)
    
    return Solid

########################################################################################################################################

# Fluid Wrapper

def vv_update1_wrapper(object Fluid, object Parameters):
    
    vv_update1(Fluid, Parameters)
    
    return Fluid

def compute_forces_wrapper(object Fluid, object WBC, object RBC, object Parameters, 
                           object Fcalc_params):
    
    compute_forces(Fluid, WBC, RBC, Parameters, Fcalc_params)
    
    return Fluid

def vv_update2_wrapper(Fluid, object Parameters, int ts):
    
    vv_update2(Fluid, Parameters, ts)
    
    return Fluid

########################################################################################################################################

# Solid Wrapper    
   
def vv_update1_solid_wrapper(object Solid, object Parameters):
    
    vv_update1_solid(Solid, Parameters)
        
    return Solid

def internal_forces_wrapper(object Solid, object Parameters):
    
    internal_forces(Solid, Parameters)
    
    return Solid

def compute_forces_solid_wrapper(object Solid, object Fluid, object Parameters, object Fcalc_params,
                  cnp.ndarray[cnp.float64_t, ndim = 2] xi):
    
    compute_forces_solid(Solid, Fluid, Parameters, Fcalc_params, xi)
    
    return Solid

def vv_update2_solid_wrapper(object Solid, object Parameters):
    
    vv_update2_solid(Solid, Parameters)
    
    return Solid
