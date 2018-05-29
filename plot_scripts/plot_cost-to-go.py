import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys

# Usage:
# python plot_cost-to-go.py ./DIRECTORY

# assuming cost-to-go has been computed every 40 episodes
for i in range(0, 800, 40):
    filename = sys.argv[1]+'/cost-to-go_'+str(i)+'.npy'
    if os.path.exists(filename):

        data = np.load(filename)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-1.2, 0.6, (0.6 + 1.2)/50)
        Y = np.arange(-0.07, 0.07, (0.07 + 0.07)/50)
        X, Y = np.meshgrid(X, Y)
        
        Z = data

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # set default view angle
        ax.elev = 90
        ax.azim = 270
        #ax.dist = 3

        plt.show()
        plt.close()