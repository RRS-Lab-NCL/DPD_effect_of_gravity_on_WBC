import os
from datetime import datetime

def Data_path(Parameters):
    
    now                 = datetime.now()
    sim_date            = str(now).replace(" ", "_")
    sim_date            = str(sim_date).replace(":", "-")
    path                = "/scratch/users/ramrup/Anirudh/DPD_py/FSI_Data_dump_" + Parameters.F_ext + "g_" + Parameters.Body_force + '_pressure_' + sim_date + sim_date
    
    if not os.path.exists(path):
        mkdir           = os.mkdir(path, mode = 0o777)
        
    return path
