"""
Force parameter file: requires Parameter class object generated from parameters.py and the generated Fluid/ Solid class object generated from Fluid.py/Solid_XXX.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)
	Solid_class_WBC.pkl (Contains minimised WBC structure with its position, velocity and accleration)

Dependencies:
	NIL
	
"""
import numpy as npy
import math

class Force_calc_params:
    
    """
    Generates force parameter class object for DPD simulation. creates and stores two seperate values for fluid-fluid (ff), fluid-solid (fs) and solid-solid (ss)
    """
    
    def __init__(self, Parameters, class_object):
        
        print('\n####################################')
        print('## Force params class constructed ##')
        print('####################################\n')
                
        if (type(class_object) == dict): 
            # Force calc Parameters for solid - solid
            self.rc                 = 1.0
            self.aij                = 50.00
            self.gammaij            = 7.282432576110537
            self.sigmaij            = npy.sqrt(2*self.gammaij*Parameters.kBT)
            # Force calc Parameters for solid - fluid
            self.rcfs               = 0.5
            self.afs                = 15.80
            self.gammafs            = 4.50
            self.sigmafs            = npy.sqrt(2*self.gammafs*Parameters.kBT)
            self.inv_dt             = 1/npy.sqrt(Parameters.dt)
        else:
            # Force calc Parameters for fluid - fluid
            self.rc                 = 1.0
            self.aij                = 75*Parameters.kBT/4
            self.gammaij            = 4.50
            self.sigmaij            = npy.sqrt(2*self.gammaij*Parameters.kBT)
            # Force calc Parameters for fluid - solid
            self.rcfs               = 0.5
            self.afs                = 15.00
            self.gammafs            = 4.50
            self.sigmafs            = npy.sqrt(2*self.gammafs*Parameters.kBT)
            self.inv_dt             = 1/npy.sqrt(Parameters.dt)
