"""
Main file: requires two inputs: 
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	pymp-pypi package required for multiprocessing (pip install pymp-pypi)
	Cython 3.0.11 convert .pyx files to c extension files (pip install Cython); 
	
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from tqdm import tqdm
from matplotlib.colors import Normalize
from Parameters import Simulation_Parameters
from Fluid import Fluid_init
from Solid_WBC import WBC_init
from Force_parameters import Force_calc_params
from RN_matrix_gen import Xi
from Physics_engine import Equation_2D_fluid, Equation_2D_WBC
from Data_dump import Data_parameters, Data_pickling, Data_WBC
from Save_location import Data_path

def main(body_force, multiple):

    parameters                      = Simulation_Parameters(body_force, multiple)				# Initiates parameter set
    Fluid                           = Fluid_init(parameters)							# Initiates Fluid particles from Fluid module
        
    Solid_WBC                       = WBC_init(parameters)							# Initiates WBC from Solid module
    
    xi_WBC                          = Xi(len(Solid_WBC[0].pos)) 						# Preallocates (n x n) random number set for solid particles
    
    fcalc_params_fluid              = Force_calc_params(parameters, Fluid) 					# Initiates DPD parameters (aij, gammaij, sigmaij) for fluid
    fcalc_params_WBC                = Force_calc_params(parameters, Solid_WBC)				# Initiates DPD parameters (aij, gammaij, sigmaij) for solid
    
    path                            = Data_path(parameters)							# Initiates path to save the trajectories
    saved_path                      = Data_parameters(path, parameters, Fluid)				# Saves simulation's metadata

    for ts in (range(int(parameters.nt))):									# Simulation start; Time marching
        
        ############################## FIRST VERLET STEP ############################################################
    
        Fluid                       = Equation_2D_fluid.VV_update1(Fluid, parameters)
        Solid_WBC                   = Equation_2D_WBC.VV_update1(Solid_WBC, parameters)
        
        ############################## COMPUTING DPD FORCES #########################################################
                                                                    
        Fluid                       = Equation_2D_fluid.Compute_forces(Fluid, Solid_WBC, {},
                                                                        parameters,fcalc_params_fluid)
        Solid_WBC                   = Equation_2D_WBC.Compute_forces(Solid_WBC, Fluid, parameters, 
                                                                      fcalc_params_WBC, xi_WBC)
        
        ############################## COMPUTING WBC INTERNAL FORCES ################################################
        
        Solid_WBC                   = Equation_2D_WBC.Internal_Forces(Solid_WBC, parameters)
        
        ############################## SECOND VERLET STEP ###########################################################
        
        Fluid                       = Equation_2D_fluid.VV_update2(Fluid, parameters, ts)        
        Solid_WBC                   = Equation_2D_WBC.VV_update2(Solid_WBC, parameters)
        
        ############################## SAVING SIMULATION DATA #######################################################
                
        if (np.mod(ts, 250) == 0):
            
            saved_path              = Data_pickling(path, Fluid, ts)
                
            for nwbc in range(len(Solid_WBC)):
                obj                 = Solid_WBC[nwbc]
                saved_path          = Data_WBC(path, obj, nwbc, ts)                
                
            print(ts, Fluid.KE[ts], Fluid.kBT[ts])
            
if __name__ == "__main__":
    
    body_force  = sys.argv[1]
    multiple    = sys.argv[2]
    op          = main(float(body_force), float(multiple))
