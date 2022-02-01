import numpy as np


def Filter_Threshold(Data, AbsoluteThresholds=False, UpperThreshold=100, LowerThreshold=0,ThresholdMode=0):
    # Calculates the absolute Upper and Lower Limit
    if AbsoluteThresholds is False:
        Max, Min = max(Data.ravel()), min(Data.ravel())
        if ThresholdMode == 0:
            Limit_Upper = Max * UpperThreshold
            Limit_Lower = Max * LowerThreshold
        elif ThresholdMode == 1:
            Limit_Upper = (Max-Min) * UpperThreshold + Min
            Limit_Lower = (Max-Min) * LowerThreshold + Min
    # Applies the Filter to Data
    Data_Modified = Data
    if AbsoluteThresholds is False and ThresholdMode == 0:
        Data_Modified[Data_Modified < Limit_Lower] = 0
        Data_Modified[Data_Modified > Limit_Upper] = 0
    elif AbsoluteThresholds is True or (AbsoluteThresholds is False and ThresholdMode == 0):
        Data_Modified[Data_Modified < Limit_Lower] = Min
        Data_Modified[Data_Modified > Limit_Upper] = Max
    return Data_Modified


def Norm_Rowwise_2D(Data_2D, ValueBelowZero):
    Max = np.max(Data_2D, axis=(1), keepdims=True)
    Max[Max <= 0] = ValueBelowZero
    return Data_2D / Max


def Norm_Columnwise_2D(Data_2D, ValueBelowZero):
    Max = np.max(Data_2D, axis=(0), keepdims=True)
    Max[Max <= 0] = ValueBelowZero
    return Data_2D / Max
