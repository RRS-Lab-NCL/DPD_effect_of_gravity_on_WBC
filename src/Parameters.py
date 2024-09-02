"""
Parameter file: requires two inputs, provided through main file
	body_force (external force applied in x-axis; datatype: float)
	multiple (gravity applied in y-axis, provide in multiples (2g = 2.0); datatype: float)

Dependencies:
	NIL
	
"""

import numpy as npy
import math

class Sim_params:
    
    def __init__(self, kBT_model, Force, time, Body_force, gravity, ka_rbc, 
                 kb_rbc, ka_wbc, kb_wbc, body_force, multiple):
        
        """
        Dictionary of all base, derived and Simulation parameters.
        1) Derived Parameters
        2) Base Parameters
        3) Simulation Parameters
        4) Time parameters
        5) External Force
        """
        print('####################################')
        print('### Parameter class constructed ####')
        print('####################################')
        
        ##
        self.kBT                = kBT_model
        self.Force              = Force
        self.time               = time
        self.Body_force         = Body_force
        self.gravity            = gravity
        self.ka_rbc             = ka_rbc
        self.kb_rbc             = kb_rbc
        self.ka_wbc             = ka_wbc
        self.kb_wbc             = kb_wbc
        ##
        self.mass               = 1
        self.np                 = 16200
        self.ncoord             = 2
        self.width              = 20
        self.particle_density   = 4
        self.length             = (self.np/(self.particle_density*self.width))
        self.depth              = 1
        self.df                 = 3*self.ncoord
        self.nrbc               = 5
        self.nwbc               = 1
        ##            
        self.lmbda              = 0.65
        self.v_max              = math.sqrt(2*self.kBT/self.mass)
        ##              
        self.dt                 = 0.001
        self.t                  = self.time
        self.nt                 = int(self.t/self.dt)
        ##           
        self.multiple           = multiple
        self.force_ext          = npy.zeros([1, 2])
        self.body_force         = body_force
        self.force_ext[:, 0]    = self.particle_density*self.body_force
        self.force_ext[:, 1]    = -self.multiple*self.gravity
        self.Body_force         = (str(self.force_ext[:, 0]))
        self.F_ext              = (str(self.multiple))
        ##
        self.RBC_condition      = 1                                             # 1 = upright, 2 = parallel
        
def Simulation_Parameters(body_force, multiple):
    
    def params():
        
        ## Base Parameters    
        # Length
        L_physical              = 1.00e-6                                       # m
        L_model                 = 1
        L_scale                 = L_physical/L_model                            # m
        
        # Fluid properties
        density_physical        = 1000                                          # kg*m^-3
        number_density          = 4                                             # m^-3
        coarse_graining_parameter = 1e17
        mass_water              = 1e-26
        mass_physical           = mass_water*coarse_graining_parameter
        
        viscosity_fluid         = 0.0010                                        # kg/(m*s)
        viscosity_ifluid        = 0.0050                                        # kg/(m*s)
        viscosity_rbcmembrane   = 0.0220                                        # kg/(m*s)
        viscosity_physical      = viscosity_fluid + viscosity_ifluid + viscosity_rbcmembrane
        
        # Energy
        T                       = 310.15                                        # K
        kB                      = 1.38e-23                                      # kg*m^2*s^-2*K^-1
        kBT_phy                 = kB*T                                          # N*m // (kg*m^2*s^-2)
        Thermal_velocity        = 0.0010                                        # m*s^-1
    
        # Material Property
        Shear_mod_physical      = 6.3e-6                                        # N/m // (kg*s^-2)
        poission_ratio          = 0.4999
        Young_mod_physical      = Shear_mod_physical*(2*(1 + poission_ratio))   # N/m // (kg*s^-2)
        Shear_mod_model         = 392.453
        Young_mod_model         = Shear_mod_model*(2*(1 + poission_ratio))
        Young_mod_scale         = Young_mod_physical/Young_mod_model            # N/m // (kg*s^-2)
                
        ## Physical Parameters    
        # Energy
        Energy_scale            = Young_mod_scale*L_scale**2                    # N*m // (kg*m^2*s^-2)
        kBT_model               = kBT_phy/Energy_scale
        
    	# Model viscosity
        viscosity_model         = ((315*kBT_model)/(128*math.pi*4.5*L_model**3)) + ((512*math.pi*4.5*(number_density**2)*L_model**5)/51975)

        # Time
        alpha                   = 1.00
        t_scale                 = ((L_scale/L_model)*(viscosity_physical/viscosity_model)*(Young_mod_model/Young_mod_physical))**alpha                    # s
        time_physical           = 150.0                                         # s
        time                    = time_physical/t_scale
        nt                      = time/0.001
        
        # Force
        force                   = npy.arange(0, 200e-12, 5e-12)
        Force_scale             = Young_mod_scale*L_scale                       # N
        Force                   = force/Force_scale
    
        # Physical Body Force
        Body_force_scale        = (Young_mod_scale/L_scale**2)*(number_density/density_physical)
        Body_force_physical     = npy.arange(0, 10, 1)                          # m*s^-2
        gravity_physical        = 9.80665                                       # m*s^-2
    
        Body_force              = Body_force_physical/Body_force_scale
        gravity                 = gravity_physical/Body_force_scale
        
        # RBC Cell parameters
        bending_constant        = 50*kBT_phy                                    # N*m
        area_coefficient        = 10.0e-6  					                    # N/m
        
        kb_rbc                  = bending_constant/Energy_scale
        ka_rbc                  = area_coefficient*L_scale**2/Energy_scale
        
        # WBC Cell parameters
        bending_constant        = 500*kBT_phy                                   # N*m
        area_coefficient        = 250.0e-5 					                    # N/m
        
        kb_wbc                  = bending_constant/Energy_scale
        ka_wbc                  = area_coefficient*L_scale**2/Energy_scale
    
        return kBT_model, Force, time, Body_force, gravity, ka_rbc, kb_rbc, ka_wbc, kb_wbc
    
    kBT_model, Force, time, Body_force, gravity, ka_rbc, kb_rbc, ka_wbc, kb_wbc = params()
    
    Simulation_parameters = Sim_params(kBT_model, Force, time, Body_force, 
                 gravity, ka_rbc, kb_rbc, ka_wbc, kb_wbc, body_force, multiple)
    
    return Simulation_parameters

