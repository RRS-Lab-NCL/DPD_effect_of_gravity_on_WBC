import os
import numpy as npy
import pandas as pd
from scipy.io import savemat
from datetime import datetime, date

def Data_parameters(path, Parameters, Fluid):
    
    now                 = datetime.now()
    sim_date            = str(now).replace(" ", "_")
    sim_date            = str(sim_date).replace(":", "-")
    delim               = ", "
    writestring         = \
        'Body_force = ' + str(Parameters.Body_force) + delim + \
            'F_ext = '      + str(Parameters.F_ext) + delim + \
                'dt = '         + str(Parameters.dt) + delim + \
                    'kBT = '        + str(Parameters.kBT) + delim + \
                        'length = '     + str(Parameters.length) + delim + \
                            'np = '         + str(len(Fluid.pos)) + delim + \
                                'width = '      + str(Parameters.width) + delim + \
                                    'nRBC = '      + str(Parameters.nrbc) + delim + \
                                        'nWBC = '      + str(Parameters.nwbc) + delim + \
                                            'ka_rbc = '      + str(Parameters.ka_rbc) + delim + \
                                                'kb_rbc = '      + str(Parameters.kb_rbc) + delim + \
                                                    'ka_wbc = '      + str(Parameters.ka_wbc) + delim + \
                                                        'kb_wbc = '      + str(Parameters.kb_wbc) + delim + \
                                                            'nt = '      + str(int(Parameters.nt)) + delim + \
                                                                'Date & time = '      + sim_date                                    
    
    filename        = path + '/Simulation_parameters.csv' 
        
    with open(filename, "w") as file:
        file.write(writestring)
    
    return path

def Data_pickling(path, Fluid, i):
    
    val             = npy.append(npy.append(Fluid.pos, Fluid.vel, 1), Fluid.acc, 1)
    df              = pd.DataFrame(val, columns = ['posx', 'posy', 'index', 
                                                    'velx', 'vely', 'index',
                                                    'accx', 'accy', 'index'])
    df.head()
    savefile        = df.to_csv(path + '/Fluid_class_' + str(i) + '.dat')
    
    return path

def Data_RBC(path, Solid, j, i):
    
    val             = npy.append(npy.append(Solid.pos, Solid.vel, 1), Solid.acc, 1)
    df              = pd.DataFrame(val, columns = ['posx', 'posy', 'index', 
                                                    'velx', 'vely', 'index',
                                                    'accx', 'accy', 'index'])
    df.head()
    savefile        = df.to_csv(path + '/Solid_class_RBC_' + str(i) + '_' + str(j) + '.dat')
        
    return path

def Data_WBC(path, Solid, j, i):
    
    val             = npy.append(npy.append(Solid.pos, Solid.vel, 1), Solid.acc, 1)
    df              = pd.DataFrame(val, columns = ['posx', 'posy', 'index', 
                                                    'velx', 'vely', 'index',
                                                    'accx', 'accy', 'index'])
    df.head()
    savefile        = df.to_csv(path + '/Solid_class_WBC_' + str(i) + '_' + str(j) + '.dat')
        
    return path
