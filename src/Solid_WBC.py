"""
Solid file: requires Parameter class object generated from parameters.py and Solid_class_WBC.pkl file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)
	Solid_class_WBC.pkl (Contains minimised WBC structure with its position, velocity and accleration)

Dependencies:
	NIL
	
"""
import os
import pandas as pd
import numpy as npy

class Solid(object):
    
    """
    Solid in this case will be RBC, WBC or any other cell
    """
    
    def __init__(self, solid_pos, solid_vel, solid_acc, m_l0, A_ref, ka, kb, 
                 r_wbc, Lp, Lm):
        
        self.pos                = solid_pos
        self.vel                = solid_vel
        self.acc                = solid_acc
        
        self.m_l0               = m_l0
        self.r_wbc              = r_wbc
        self.A_ref              = A_ref
        self.ka                 = ka
        self.kb                 = kb
        self.Lp                 = Lp
        self.Lm                 = Lm
        
def WBC_init(Parameters):  
    
    def wbc_init(df, Parameters, rotation_matrix):
        
        wbc                     = {}
                    
        for i in range(0, Parameters.nwbc):
        
            x                   = df.posx
            y                   = df.posy
            wbc_pos             = npy.vstack((x, y)).T
                
            xunit               = (Parameters.length*0.05)
            yunit               = (Parameters.width/2) - npy.mean(df.posy)
            wbc_pos[:, 0]       += xunit
            wbc_pos[:, 1]       += yunit
            r_wbc               = 4.50
            
            vel_wbc             = npy.vstack((df.velx, df.vely)).T
            acc_wbc             = npy.vstack((df.accx, df.accy)).T
        
            # Length
            m_l0                = 1.225*(2*npy.pi*r_wbc)/len(wbc_pos)
            Lp                  = 0.001770245768690
        
            # Area
            s                   = 1.050
            A_ref               = (s*npy.pi*(2*r_wbc)**2)/4   
            
            wbc[i]              = {"wbc_pos": wbc_pos, "wbc_vel": vel_wbc, 
                                   "wbc_acc": acc_wbc, "m_l0": m_l0, 
                                   "A_ref": A_ref, "ka": Parameters.ka_wbc, 
                                   "kb": Parameters.kb_wbc, "r_wbc": r_wbc,
                                   "Lp": Lp, "Lm": m_l0}
            
        return wbc

    def classifaction_solids(cell):        
        """
        Classifying the solid in case of Multiple different types of solids are being used in the simulation.
        """ 
        key             = npy.arange(0, len(cell["wbc_pos"]), 1).reshape(
                                        len(cell["wbc_pos"]), 1).astype(int)
        
        solid_pos       = (npy.append(cell["wbc_pos"], key, 1))
        solid_vel       = (npy.append(cell["wbc_vel"], key, 1))
        solid_acc       = (npy.append(cell["wbc_acc"], key, 1))
        
        solid           = Solid(solid_pos, solid_vel, solid_acc, cell["m_l0"], 
                                cell["A_ref"], cell["ka"], cell["kb"], cell["r_wbc"],
                                cell["Lp"], cell["Lm"])         
        return solid    
    
    Data_path           = os.getcwd()
    df                  = pd.read_pickle(Data_path + "/Solid_class_WBC.pkl")
    
    theta               = npy.random.randn()
    sine                = npy.sin(theta)
    cosine              = npy.sin(theta)
    rotation_matrix     = npy.array([[cosine, -sine], 
                                     [sine,  cosine]])
    
    cell                = wbc_init(df, Parameters, rotation_matrix)
    Cells               = {}
    
    for i in range (0, Parameters.nwbc):
        
        Cells[i]        = classifaction_solids(cell[i])
    
    return Cells
