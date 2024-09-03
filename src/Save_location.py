import os
from datetime import datetime

def Data_path(Parameters):
    
    now                 = datetime.now()
    sim_date            = str(now).replace(" ", "_")
    sim_date            = str(sim_date).replace(":", "-")
    current_dir 	 = os.path.dirname(os.path.abspath(__file__))
    base_dir 		 = os.path.dirname(current_dir)
    path                = base_dir + '/Data_dump/DPD_data_' + Parameters.F_ext + "g_" + Parameters.Body_force + '_pressure_' + sim_date + sim_date
    
    if not os.path.exists(path):
        mkdir           = os.mkdir(path, mode = 0o777)
        
    return path
