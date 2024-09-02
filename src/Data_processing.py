import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def Data_processing(Data_dump, Parameters):
    
    # Locating all available files in the folder
    count = 0    
    for root_dir, cur_dir, files in os.walk(Data_dump):        
        count += len(files)
        
    save_freq               = 2500
    
    for i in range(0, int(count/3)*save_freq, save_freq):
        
        df1                 = pd.read_csv(Data_dump + '/Fluid_class_' + str(i) 
                                              + '.dat')        
        df2                 = pd.read_csv(Data_dump + '/Solid_class_' + str(i) 
                                              + '_0.dat')
        df3                 = pd.read_csv(Data_dump + '/Solid_class_' + str(i) 
                                              + '_1.dat')
        
        epsilon             = 1e-2    
        fig, ax             = plt.subplots()
        
        # initialise scatter plot
        color               = np.sqrt(df1.accx**2 + df1.accy**2)
        norm                = Normalize()
        colormap            = cm.jet
        plt.scatter(df1.posx, df1.posy, color=colormap(norm(color)), s = 5)
        ax.quiver(df1.posx, df1.posy, df1.velx, df1.vely, color=colormap(norm(color)), alpha = 1)
        
        # initialise scatter plot
        color               = np.sqrt(df2.accx**2 + df2.accy**2)
        norm                = Normalize()
        colormap            = cm.cividis
        plt.plot(df2.posx, df2.posy, color='gold')
        plt.scatter(df2.posx, df2.posy, color='blue', s = 25)
        
        color               = np.sqrt(df2.velx**2 + df2.vely**2)
        norm                = Normalize()
        colormap            = cm.plasma
        ax.quiver(df2.posx, df2.posy, df2.velx, df2.vely, color=colormap(norm(color)), alpha = 0.1)
        
        # initialise scatter plot
        color               = np.sqrt(df3.accx**2 + df3.accy**2)
        norm                = Normalize()
        colormap            = cm.cividis
        plt.plot(df3.posx, df3.posy, color='cyan')
        plt.scatter(df3.posx, df3.posy, color='red', s = 20)
        # initialise colormap
        color               = np.sqrt(df1.velx**2 + df1.vely**2)
        norm                = Normalize()
        colormap            = cm.plasma
        # set quiver plot
        ax.quiver(df1.posx, df1.posy, df1.velx, df1.vely, color=colormap(norm(color)), alpha = 1)
        ax.set(xlim = (50, 150), ylim = (0, 20))
        
        ax.set_aspect('equal')
        # set the title, label and legend
        plt.title('Fluid movement, timestep = ' + str(i))
        plt.xlabel('Particle Position')
        plt.ylabel('Particle Position')
        plt.legend('', frameon=False)

        plt.pause(0.00005)