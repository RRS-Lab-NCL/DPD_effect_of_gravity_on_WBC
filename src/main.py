#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:51:21 2024

@author: itd-2495
"""

"""
Main file: requires two inputs: 
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	pymp-pypi package required for multiprocessing (pip install pymp-pypi)
	Cython 3.0.11 convert .pyx files to c extension files (pip install Cython); 
	
"""
import os
import sys
import pickle
import numpy as np

from Parameters import Simulation_Parameters
from Fluid import Fluid_init
from Solid_WBC import WBC_init
from Force_parameters import Force_calc_params
from RN_matrix_gen import Xi
from Physics_engine import Equation_2D_fluid, Equation_2D_WBC
from Data_dump import Data_parameters, Data_pickling, Data_WBC
from Save_location import Data_path

def load_checkpoint(path):
    
    try:
        with open(os.path.join(path, 'last_checkpoint.pkl'), 'rb') as f:
            checkpoint      = pickle.load(f)
        print("Resuming from saved data...")
        return checkpoint
    
    except FileNotFoundError:
        print("No checkpoint found, starting a new simulation...")
        return None

def save_checkpoint(path, Fluid, Solid_WBC, ts):
    
    checkpoint = {'Fluid': Fluid,
                  'Solid_WBC': Solid_WBC,
                  'ts': ts
                  }
    
    with open(os.path.join(path, 'last_checkpoint.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)

def main(body_force, multiple):

    parameters                      = Simulation_Parameters(body_force, multiple)				# Initiates parameter set
    path                            = Data_path(parameters)                                     # Path for saving data
    
    while True:
        
        checkpoint                      = load_checkpoint(path)
        
        if checkpoint:
            Fluid                       = checkpoint['Fluid']
            Solid_WBC                   = checkpoint['Solid_WBC']
            start_ts                    = checkpoint['ts'] + 1                                      # Resume from next step
        else:
            Fluid                       = Fluid_init(parameters)
            Solid_WBC                   = WBC_init(parameters)
            start_ts                    = 0
            
        xi_WBC                          = Xi(len(Solid_WBC[0].pos)) 				        		# Preallocates (n x n) random number set for solid particles
        
        fcalc_params_fluid              = Force_calc_params(parameters, Fluid) 		    			# Initiates DPD parameters (aij, gammaij, sigmaij) for fluid
        fcalc_params_WBC                = Force_calc_params(parameters, Solid_WBC)	     			# Initiates DPD parameters (aij, gammaij, sigmaij) for solid
        
        saved_path                      = Data_parameters(path, parameters, Fluid)  				# Saves simulation's metadata
        
        try:
    
            for ts in (range(start_ts, int(parameters.nt))):								        # Simulation start; Time marching
                
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
                    save_checkpoint(path, Fluid, Solid_WBC, ts)                                     # Save checkpoint periodically
    
        except Exception as e:
            
            print(f"An error occurred at time step {ts}: {e}")
            print("Exiting the current cycle and restarting from the previous checkpoint...")
            continue                                                                             # Exit the program to allow a safe restart
                
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python simulation.py <body_force> <multiple>")
        sys.exit(1)
    
    body_force      = sys.argv[1]
    multiple        = sys.argv[2]
    main(float(body_force), float(multiple))
