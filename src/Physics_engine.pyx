"""
Physics_engine.pyx: 
	requires class objects as input generated from Fluid.py, Solid_WBC.py, Parameters.py, Force_parameters.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	"Func_caller.pyx"
	
Compilation: 
	This file is written in cyton extension format, need to be compiled with all its neceesary dependencies.
	
"""

# distutils: language = c++

# Import required libraries
import Func_caller
import numpy as np

#############################################################################################################################################################################################################################################

class Equation_2D_fluid(object):
    
    # Velocity verlet scheme 1
    def VV_update1(self, object Parameters):
        
        Func_caller.vv_update1_wrapper(self, Parameters)
                
        return self
    
    # Forces on fluid particles
    def Compute_forces(self, object WBC, object RBC, object Parameters, object Fcalc_params):
        
        Func_caller.compute_forces_wrapper(self, WBC, RBC, Parameters, Fcalc_params)
        
        return self
    
    def VV_update2(self, object Parameters, int ts):
        
        Func_caller.vv_update2_wrapper(self, Parameters, ts)

        return self

#############################################################################################################################################################################################################################################

class Equation_2D_RBC(object): 
    
    # Velocity verlet scheme 1
    def VV_update1(self, object Parameters):
                
        Func_caller.vv_update1_solid_wrapper(self, Parameters)
    
        return self
        
    # Internal forces of the object
    def Internal_Forces(self, object Parameters):
        
        Func_caller.internal_forces_wrapper(self, Parameters)
                
        return self
    
    # Forces on particles
    def Compute_forces(self, object Fluid, object Parameters, object Fcalc_params,
                       xi):
        
        Func_caller.compute_forces_solid_wrapper(self, Fluid, Parameters, Fcalc_params, xi)
        
        return self
    
    # Velocity verlet scheme 2
    def VV_update2(self, Parameters):
        
        Func_caller.vv_update2_solid_wrapper(self, Parameters)
        
        return self

#############################################################################################################################################################################################################################################

class Equation_2D_WBC(object): 
    
    # Velocity verlet scheme 1
    def VV_update1(self, object Parameters):
                
        Func_caller.vv_update1_solid_wrapper(self, Parameters)
    
        return self
        
    # Internal forces of the object
    def Internal_Forces(self, object Parameters):
        
        Func_caller.internal_forces_wrapper(self, Parameters)
                
        return self
    
    # Forces on particles
    def Compute_forces(self, object Fluid, object Parameters, object Fcalc_params,
                       xi):
        
        Func_caller.compute_forces_solid_wrapper(self, Fluid, Parameters, Fcalc_params, xi)
        
        return self
    
    # Velocity verlet scheme 2
    def VV_update2(self, Parameters):
        
        Func_caller.vv_update2_solid_wrapper(self, Parameters)
        
        return self
