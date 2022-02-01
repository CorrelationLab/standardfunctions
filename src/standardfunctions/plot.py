import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from matplotlib.colors import LogNorm


def plot1DPlot(Data_X, Data_Y, SavePath=None, Unit_X="", Unit_Y="", Title="", Points='b.'):
    plt.cla()
    plt.clf()
    Fig = plt.figure()
    Ax = Fig.add_subplot()
    Ax.set_title = Title
    Ax.set_xlabel = Unit_X
    Ax.set_ylabel = Unit_Y
    plt.plot(Data_X, Data_Y, fmt=Points)
    if SavePath[-3:].lower() in ['pdf', 'png', 'jpg']:
        Fig.savefig(SavePath)
    else:
        plt.show()
    plt.close('all')


def plotHeightMap(Data_2D, SavePath=None, PxToUnit=(1, 1), Unit_X="X in Pixel", Unit_Y="Y in Pixel", Title="", Centered=(False, False), Deviation=(0, 0), Origin='upper', FigShape=None, Norm=None, IntFunction=None, LogZeroValue=0.01):
# Initial Manipulations
    if IntFunction is not None:
        Data_2D = IntFunction(Data_2D)
    Maxi = max(Data_2D.shape)

# Set Figure
    if FigShape is None:
        Fig = plt.figure(figsize=(16 * Data_2D.shape[1] / Maxi, 16 * Data_2D.shape[0] / Maxi))
    else:
        Fig = plt.figure(figsize=tuple(FigShape))
    
# Set Extent
    Extent_X = [0, Data_2D.shape[1]]
    if Centered[0] is True:
        Extent_X[0] = int(-Data_2D.shape[1] / 2.0)
        Extent_X[1] = int(Data_2D.shape[1] / 2.0)
    Extent_X[0] -= Deviation[0]
    Extent_X[1] -= Deviation[0]
    
    Extent_Y = [0, Data_2D.shape[0]]
    if Centered[1] is True:
        Extent_Y[0] = int(-Data_2D.shape[0] / 2.0)
        Extent_Y[1] = int(Data_2D.shape[0] / 2.0)
        Extent_X[0] -= Deviation[0]
        Extent_X[1] -= Deviation[0]

#  Matches the Extent for Odd amount of Points
    if Data_2D.shape[1] % 2 == 1:
        Extent_X[0] -= 0.5
        Extent_X[1] += 0.5
    if Data_2D.shape[0] % 2 == 1:
        Extent_Y[0] -= 0.5
        Extent_Y[1] += 0.5
    Extent = Extent_X + Extent_Y

#Set Axis Ticker
    if type(PxToUnit[0]) is list:
        TicksX = ticker.FuncFormatter(lambda x, pos: '{0:.6f}'.format(x*PxToUnit[0][0] + PxToUnit[0][1]))
    else:
        TicksX = ticker.FuncFormatter(lambda x, pos: '{0:.6f}'.format(x*PxToUnit[0]))
    if type(PxToUnit[1]) is list:
        TicksY = ticker.FuncFormatter(lambda x, pos: '{0:.6f}'.format(x*PxToUnit[1][0] + PxToUnit[1][1]))
    else:
        TicksY = ticker.FuncFormatter(lambda x, pos: '{0:.6f}'.format(x*PxToUnit[1]))

# Set Valuearea
    if Norm == 'Log':
        Data_2D[Data_2D <= 0] = LogZeroValue
    DataMax = max(np.ravel(Data_2D))
    DataMin = min(np.ravel(Data_2D))

# Plot
    Ax = Fig.add_subplot(1, 1, 1)# Set the title of plot
    if Norm == 'Log':
        Pic = Ax.imshow(Data_2D, cmap=plt.cm.RdYlBu_r, norm=LogNorm(vmin=max([Data_2D.min(), 0.0001]), vmax=Data_2D.max()), aspect='equal', origin=Origin, extent=Extent)
    else:
        Pic = Ax.imshow(Data_2D, cmap=plt.cm.RdYlBu_r, vmin=DataMin, vmax=DataMax, aspect='equal', origin=Origin, extent=Extent)
    Ax.set_title(Title)
    Ax.set_xlabel(Unit_X)
    Ax.set_ylabel(Unit_Y)
    Ax.xaxis.set_major_formatter(TicksX)
    Ax.yaxis.set_major_formatter(TicksY)
    Fig.colorbar(Pic, ax=Ax)
    plt.legend()
    if SavePath is not None:
        if SavePath[-3:].lower() in ['pdf', 'png', 'jpg']:
            Fig.savefig(SavePath)
    else:
        plt.show()