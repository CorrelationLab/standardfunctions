from scipy.optimize import curve_fit
import numpy as np
from uncertainties import unumpy as unp
import cv2


# Base Functions
# Different Functions for Fitting a 1D Graph

def Gauss_1D(Data_X, Height, MaxPosition, Sigma):
    return Height * np.exp(-((Data_X - MaxPosition) / (np.sqrt(2) * Sigma))**2)


def Lorentz_1D(Data_X, Height, MaxPosition, Gamma):
    return (Height * Gamma**2) / ((Data_X - MaxPosition)**2 + Gamma**2)

# Different Functions for Fitting a 2D heatmap

def gauss2D(Meshgrid_2D, Amplitude, X0, Y0, SigmaX, SigmaY, Ellipticity):
    (x, y) = Meshgrid_2D
    g = Amplitude*np.exp(-(SigmaX * ((x-X0)**2) + 2 * Ellipticity * (x-X0) * (y-Y0) + (SigmaY * (y-Y0)**2)))
    return g.ravel()


def lorentz2D(Meshgrid_2D, Amplitude, X0, Y0, SigmaX, SigmaY):
    (x, y) = Meshgrid_2D
    g = Amplitude / (1 + (((x - X0) / SigmaX)**2)+(((y - Y0) / SigmaY)**2))
    return g.ravel()


# getting optimated Parameters

def getOptimatedFittingParameters_1D(Data_X, Data_Y, Function):

    def OptimatedParameters__Gauss_1D(Data_X, Data_Y):
        MaxPosition_Index = np.argmax(Data_Y)
        Height = Data_Y[MaxPosition_Index]
        MaxPosition = Data_X[MaxPosition_Index]

        i = MaxPosition_Index
        HalfOfMaximum = Height / 2
        while Data_Y[i] > HalfOfMaximum:
            i += 1
        j = MaxPosition_Index
        while Data_Y[j] > HalfOfMaximum:
            j -= 1
        Sigma = abs(Data_X[j] - Data_X[i]) / (2 * np.sqrt(2*np.log(2)))
        return [Height, MaxPosition, Sigma]

    def OptimatedParameters__Lorentz_1D(Data_X, Data_Y):
        """
        To find a proper value for gamma this design bases on the cauchy distribution
        """
        MaxPosition_Index = np.argmax(Data_Y)
        Height = Data_Y[MaxPosition_Index]
        MaxPosition = Data_X[MaxPosition_Index]

        i = MaxPosition_Index
        HalfOfMaximum = Height / 2
        while Data_Y[i] > HalfOfMaximum:
            i += 1
        j = MaxPosition_Index
        while Data_Y[j] > HalfOfMaximum:
            j -= 1
        Gamma = abs(Data_X[j] - Data_X[i]) / 2
        return [Height, MaxPosition, Gamma]


    Functions_1D = {'Gauss_1D': OptimatedParameters__Gauss_1D,
                    'Lorentz_1D': OptimatedParameters__Lorentz_1D
                    }
    assert(str(Function).split(' ')[1] in Functions_1D.keys())
    return Functions_1D[str(Function).split(' ')[1]](Data_X, Data_Y)

# kann momentan vll noch keine dateifremden funktionen aufnehmen



def getOptimatedFittingParameters_2D(Data, Function):

    def OptimatedParameters__gauss_2D(Data):
        # Sigma x und y sind noch disfunktional, Elipticity fehlt noch komplett
        assert(Data.dim == 2), " Your Data has the wrong dimension. It has to be two-dimensional"
        Shape = Data.shape
        MaxPosition = np.unravel_index(np.argmax(Data, axis=None), Shape)
        Height = Data[MaxPosition[0], MaxPosition[1]]
        MaxPos_X = MaxPosition[1]
        MaxPos_Y = MaxPosition[0]
        Sigma_X = 2.772/(Shape[1]**2)
        Sigma_Y = 2.772/(Shape[0]**2)
        Ellipticity = 0
        return [Height, MaxPos_X, MaxPos_Y, Sigma_X, Sigma_Y, Ellipticity]

    def OptimatedParameters__lorentz_2D(Data):
        assert(Data.dim == 2)
        Shape = Data.shape
        MaxPosition = np.unravel_index(np.argmax(Data, axis=None), Shape)
        Height = Data[MaxPosition[0], MaxPosition[1]]
        MaxPos_X = MaxPosition[1]
        MaxPos_Y = MaxPosition[0]
        return [3]


    Functions_2D = {'Gauss_2D': OptimatedParameters__gauss_2D,
                    'Lorentz_2D': OptimatedParameters__lorentz_2D
                    }
    assert(str(Function) in Functions_2D.keys())
    return Functions_2D[str(Function)](Data)


def FitData_1D(Data_X, Data_Y, Function, StartParameter=None, ReturnValue='ValueOnly'):
    assert(len(Data_X) == len(Data_Y)), "Your X and Y Data have different lengths"
    if StartParameter is None:
        StartParameter = getOptimatedFittingParameters_1D(Data_X, Data_Y, Function)
    try:
        print(Function)
        Parameter, Cov = curve_fit(Function, Data_X, Data_Y, p0=StartParameter)
    except:
        print("Could not estimate Parameters")
        raise
    if ReturnValue == 'ValueOnly':
        return Parameter
    elif ReturnValue == 'ValuePlusError':
        return unp.uarray(Parameter, np.sqrt(np.diag(Cov)))
    elif ReturnValue == 'ErrorOnly':
        return np.sqrt(np.diag(Cov))
    else:
        raise

# Filter Data (They return pnly parts of given Data which fulfill a condition)


# for 2D Data
def getMaskOfDataInRange(Data_2D, Threshold_Lower=None, Threshold_Upper=None, ThresholdType='Rel'):
    if ThresholdType == 'Rel':
        if Threshold_Lower is not None:
            Threshold_Lower = min(Data_2D.ravel())*Threshold_Lower
        if Threshold_Upper is not None:
            Threshold_Upper = min(Data_2D.ravel())*Threshold_Upper
    elif ThresholdType == "Abs":
        pass
    else:
        raise
    if Threshold_Lower is not None:
        Mask_Lower = Data_2D[Data_2D > Threshold_Lower]
    else:
        Mask_Lower = np.ones(Data_2D.shape)
    if Threshold_Upper is not None:
        Mask_Upper = [Data_2D > Threshold_Lower]
    else:
        Mask_Upper = np.ones(Data_2D.shape)
    Mask = Mask_Lower + Mask_Upper
    return (Mask[Mask == 2] * 255).astype('uint8')


def getMinimalRectangleFromMask(Data, Mask, OnlyRectWithMax=True, Mode=cv2.RETR_EXTERNAL, Method=cv2.CHAIN_APPROX_SIMPLE):
    Contours = cv2.findContours(Mask, Mode, Method)
    Contours = Contours[0] if len(Contours) == 2 else Contours[1]
    ROIS = []
    if OnlyRectWithMax is False:
        for f in Contours:
            x, y, w, h = cv2.boundingRect(f)
            ROIS += Data[y:y+h+1, x:x+w+1]
    elif OnlyRectWithMax is True:
        MaxPosition = np.unravel_index(np.argmax(Data, axis=None), Data.shape)
        for f in Contours:
            x, y, w, h = cv2.boundingRect(f)
            if MaxPosition[0] >= y and MaxPosition[0] <= y+h and MaxPosition[1] >= x and MaxPosition[1] <= x+w:
                ROIS += Data[y:y+h+1, x:x+w+1]
    return ROIS


def findPeaksByGradient_1D(Data_1D, StartPeakPositions, Distance, MinGradient=None, MaxGradient=None):
    LastIndex = len(Data_1D) - 1
    StartPeakPositions = [i for i in StartPeakPositions if not (i - Distance < 0 or i + Distance > LastIndex)]
    PeakGradient = [[i, (max(Data_1D[i - Distance: i]) - min(Data_1D[i: i + Distance]))] for i in StartPeakPositions]
    if MinGradient is not None:
        PeakGradient = [f for f in PeakGradient if f[1] >= MinGradient]
    if MaxGradient is not None:
        PeakGradient = [f for f in PeakGradient if f[1] <= MaxGradient]
    return np.array(PeakGradient).T[0].astype('int32')


# DataManipulations