"""
Fluid file: requires Parameter class object generated from parameters.py file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	NIL
	
"""

import numpy as npy
import pandas as pd
import math
import os

class Fluid(object):
        
    def __init__(self, fluid_pos, fluid_vel, fluid_acc, Parameters):
        
        """
        Fluid in this case is blood, lymph or interstitial fluid
        Object is made up of appended list comprising of position, velocity and accln and Indx number.
        position = array[:, 0:1]    velocity = array[:, 0:1]    accln = array[:, 0:1]   key = array[:, 0:1]
        """  
        
        print('\n####################################')
        print('##### Fluid class constructed ######')
        print('####################################\n')
        
        self.pos 	= fluid_pos
        self.vel 	= fluid_vel
        self.acc 	= fluid_acc
        self.kBT 	= npy.zeros((int(Parameters.nt), ))
        self.KE  	= npy.zeros((int(Parameters.nt), ))
        
def Fluid_init(Parameters):       
        
    def FCC_init(Parameters):
    	"""
    	Generating particle positions in FCC (Face Centered Cube) structure 
    	"""
        pos_max         	= int((2*math.sqrt(Parameters.np/2)))
        part_pos_max 		= npy.zeros((pos_max+2, Parameters.ncoord))
        part_pos 	    	= npy.zeros((Parameters.np + 2*int(0.5*pos_max)+1, Parameters.ncoord), dtype = npy.float64)
        nx 		        = (Parameters.length/pos_max)
        ny 		        = (Parameters.width/pos_max)
        i 		        = -1
    
        for k in range(0, (pos_max+2), 2):
            part_pos_max[k, 0] 	    = -(k*nx) + (Parameters.length)
            for m in range(0, (pos_max+2), 2):
                part_pos_max[m, 1] 	= -(m*ny) + (Parameters.width)
                i 			        = i+1
                part_pos[i, 0] 	    = part_pos_max[k, 0]
                part_pos[i, 1] 	    = part_pos_max[m, 1]
    
        for k in range(1, (pos_max+1), 2):
            part_pos_max[k, 0] 	    = -(k*nx) + (Parameters.length)
            for m in range(1, (pos_max+1), 2): 
                part_pos_max[m, 1] 	= -(m*ny) + (Parameters.width)
                i 			        = i+1
                part_pos[i, 0] 	    = part_pos_max[k, 0]
                part_pos[i, 1] 	    = part_pos_max[m, 1]
                
        v 		                    = npy.random.randn(len(part_pos), Parameters.ncoord)
        acc 		                    = npy.random.randn(len(part_pos), Parameters.ncoord)        
        
        return part_pos, v, acc
    
    def fluid_classifaction(part_pos, v, acc):
    	"""
    	Initiate the formation of Fluid class object for fluid particles with their own seperate identifier (key)
    	
    	"""
        
        key 		     = npy.arange(0, len(part_pos), 1).reshape(len(part_pos), 1).astype(int)
        
        fluid_pos 	     = (npy.append(part_pos, key, 1))
        fluid_vel 	     = (npy.append(v, key, 1))
        fluid_acc 	     = (npy.append(acc, key, 1))
        
        Fluid_object 	     = Fluid(fluid_pos, fluid_vel, fluid_acc, Parameters)
                     
        return Fluid_object
    
    part_pos, v, acc 		= FCC_init(Parameters)
    fluid 		        = fluid_classifaction(part_pos, v, acc)
    
    return fluid
