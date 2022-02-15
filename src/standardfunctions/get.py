# Necessary Imports
import os
import re
import pandas as pd
import numpy as np
import spe_loader as sl
from copy import deepcopy
import networkx as kx
from operator import itemgetter
import toolz
import pickle
from scipy.optimize import curve_fit
from uncertainties import uarray
import scipy.constants as sc

#Basic Functions:

# Get Paths ########################################################################################

def getPathsFromFolderTree(FolderPath, FileTypes=None, Pattern=None):
    """
        This function returns all files in a given directory and its subdirectories. In case of failure it returns None
        The optional parameters are:
        - FileTypes: specifies the returned files by their extension. The Extensions have to inserted as a list (eg. FileTypes=['.spe', '.img']) 
        - Pattern: specifies the returned files by their a pattern for their name. The pattern uses the formal construction of regular expressions as has to be given as a raw string

    """
    try:
        if ((FileTypes is None) and (Pattern is None)):
            return [os.path.join(dp, f) for dp, dn, filenames in os.walk(FolderPath) for f in filenames]
        elif ((FileTypes is not None) and (Pattern is None)):
            return [os.path.join(dp, f) for dp, dn, filenames in os.walk(FolderPath) for f in filenames if (os.path.splitext(f)[1] in FileTypes)]
        elif ((FileTypes is None) and (Pattern is not None)):
            return [os.path.join(dp, f) for dp, dn, filenames in os.walk(FolderPath) for f in filenames if (re.search(Pattern, os.path.splitext(f)[0]) is not None)]
        else:
            return [os.path.join(dp, f) for dp, dn, filenames in os.walk(FolderPath) for f in filenames if ((os.path.splitext(f)[1] == FileTypes) and (re.search(Pattern, os.path.splitext(f)[0]) is not None))]
    except:
        return None


def getBaseNameAndExtFromPath(FilePath):
    """
        returns tuple of (filename,extension) of a given path.
        - if path corresponds to a directory it returns (filename,'')
        - if it fails it returns (None, None)
    """
    try:
        return os.path.splitext(os.path.basename(FilePath))
    except:
        return (None, None)


# Manipulate Paths####################################################################################


def addToFileNames(File_FolderPath, NameAdditions, FileTypes=None, Pattern=None, IgnoreIsIn=False, Seperator='_'):
    """
    extend the filename of one or multiple files by a list of constant strings which are seperated by a Seperator
    Necessary Arguments:
        - File_FolderPath:  path to a single file or to a Directory. In case of last all files of given filetypes and / or pattern will be modified. Therefore 'getPathsFromFolderTree' is used.
        - NameAdditions:    list Of strings which should be added
    Optional Arguments:
        - FileTypes:        list of extensions eg: FileTypes=['.spe', '.img']
        - Pattern:          regular expression which defines the files which should be modified
        - IgnoreIsIn:       Boolean: if True it will be also checked if a given substring is already present in the actual modified filename
        - Seperator:        used seperator for the appended substrings 
    """
    if os.path.isfile(File_FolderPath):
        FilePaths = [File_FolderPath]
    if os.path.isdir(File_FolderPath):
        FilePaths = getPathsFromFolderTree(File_FolderPath, FileTypes, Pattern)
    else:
        raise
    if IgnoreIsIn is False:
        for f in FilePaths:
            NewNameAdditions = [g for g in NameAdditions if g not in f]
            if NewNameAdditions is not []:
                NewNameAdditions = Seperator.join(NewNameAdditions) + Seperator
            else:
                NewNameAdditions = ""
            NewName = os.path.join(os.path.dirname(f), NewNameAdditions + os.path.basename(f))
            os.rename(f, NewName)
    else:
        for f in FilePaths:
            NewName = os.path.join(os.path.dirname(f), Seperator.join(NameAdditions) + Seperator + os.path.basename(f))
            os.rename(f, NewName)


def renameFileNamesByPattern(File_FolderPath, Substitute, FileTypes=None, Pattern=None, SubstituteIsPattern=True, IgnoreIsIn=True):
    """
        !!!WIP!!!
        renames Filesnames which have a special pattern by substituting the pattern with another one. ATTENTION: It does not care if the replacement has already be done in the past !!!

    """
    
    if os.path.isfile(File_FolderPath):
        FilePaths = [File_FolderPath]
    if os.path.isdir(File_FolderPath):
        FilePaths = getPathsFromFolderTree(File_FolderPath, FileTypes, Pattern)
    else:
        raise
    if SubstituteIsPattern is False:
        for f in FilePaths:
            for g in Substitute:
                os.rename(g[0], g[1])
    else:
        if IgnoreIsIn is True:
            for f in FilePaths:
                Original = f
                for g in Substitute:
                    New = re.sub(g[0], g[1], Original)
                    os.rename(Original, New)
                    Original = New
        else:
            for f in FilePaths:
                Original = f
                for g in Substitute:            
                    Groups = re.findall(r'\(.*\)', g[0], flags=re.DOTALL)
                    Groups = [""] + Groups
                    NewSubstitute = re.sub(r'\\(\d)+', r'{\1}', g[1])
                    NewSubstitute = NewSubstitute.format(*Groups)
                    if not re.search(NewSubstitute, Original):
                        New = re.sub(g[0], g[1], Original)
                        os.rename(Original, New)
                        Original = New


def copyDirTree(InputPath, OutputPath):
    """
    recreates a directorytree which seeds from a given directory at another directory:
    Necessary Arguments:
    - InputPath: Path where the original directorytree can be found
    - OutPutPath: Path where directorytree should be recreated    
    """
    for dirpath, dirnames, filenames in os.walk(InputPath):
        structure = os.path.join(OutputPath, os.path.relpath(dirpath, InputPath))
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")



# Get Data From File ########################################################################################


def getDataFromCSV(FilePath, Header=None, Seperator=','):
    """
    returns Data of a CSV-File as numpy array. In case of failure it returns None
    Necessary Arguments:
    - FilePath:     path to file
    Optional Arguments: 
    - Header:       header of the csv file
    - Seperator:    seperator used in the csv file
    """
    try:
        return np.array(pd.read_csv(FilePath, header=Header, sep=Seperator))
    except:
        return None


def getDataFromSPE(FilePath, Frame=0, ROI=0):
    """
    returns Data of an SpE-File as numpy array. In case of failure it returns None
    Necessary Arguments:
    - FilePath:       path to file
    Optional Arguments:
    - Frame:          framenumber in case of multiple frames saved to one file. default is 0
    - ROI:            ROInumber in case of multiple ROIs are saved to one frame. default is 0
    """
    try:
        return np.array((sl.load_from_files([FilePath])).data[Frame][ROI]).astype(np.float64)
    except:
        return None


def getAllDataFromSPE(FilePath):
    """
    returns Frames and ROIS of an SpeFile as an 2D list of numpy arrays. In case of failure it returns None
    Necessary Arguments:
    - FilePath:       path to file
    """
    try:
        return list((sl.load_from_files([FilePath])).data)
    except:
        return None


def getDataFromIMG(FilePath):
    """
    returns Data from an IMG file as numpy array. In case of failure it returns None
    Necessary Arguments:
    - FilePath:       path to file
    """
    try:
        File = open(FilePath, 'rb')
        File_Inread = File.read()
        Data_StartPos = File_Inread.find(b'\x65\x6E\x74\x3d\x22\x22')
        MetaData = File_Inread[64: Data_StartPos + 6].decode("utf8")
        Shape = (getMetaDataInfoFromIMG_MetaData(MetaData, r"VWidth=\"[0-9]+\"", Type=int), getMetaDataInfoFromIMG_MetaData(MetaData, r"HWidth=\"[0-9]+\"", Type=int))
        Data = deepcopy((np.frombuffer(File_Inread[Data_StartPos + 6:], dtype="int32")).reshape(Shape[0], Shape[1]))
        File.close()
        return Data
    except:
        return None


# Get MetaData From File

def getMetaDataInfoFromFilePath(FilePath, Pattern, SubPattern=r"[0-9]+[i]?[0-9]*", Substitution=("i", "."), Type=float):
    """
    returns MetaData which is saved inside a filepath. In case of failure it returns None
    Necessary Arguments:
    - Filepath:       path to file
    Pattern:        regular expression of the quantity you are searching (eg. Pattern=r"Pow[0-9]+[i]?[0-9]*")
    - SubPattern:     regular expression of the way the value of the quantity is saved. default is SubPattern=r"[0-9]+[i]?[0-9]*"
    - Substitution:   tupel of the way decimal points are saved. default is Substitution=("i", ".")
    - Type:           datatype of the saved quantity. default is float

    """
    try:
        return Type(re.sub(Substitution[0], Substitution[1], re.search(SubPattern, (re.search(Pattern, FilePath).group())).group()))
    except:
        return None
#________________________
# DEFAULTWERT EINBAUEN !!!!!!
#________________________

def getMetaDataFromIMG(FilePath):
    """
    returns MetaData of an IMG File as a String, In case of failure it returns None
    Necessary Arguments:
    - FilePath:         path to file
    """
    try:
        File = open(FilePath, 'rb')
        File_Inread = File.read()
        Data_StartPos = File_Inread.find(b'\x65\x6E\x74\x3d\x22\x22')
        MetaData = deepcopy(File_Inread[64: Data_StartPos + 6].decode("utf8"))
        File.close()
        return MetaData
    except:
        return None



# NEEDS REVISION
def getMetaDataInfoFromIMG(FilePath, Pattern, SubPattern=r"[0-9]+[i]?[0-9]*", Type=float):
    """
    ATTENTION : THIS FUNCTION SEEMS TO HAVE PROBLEMS AND NEED A REVISION
    returns Quantity which is saved in the MetaData of an img file. In case of failure it returns None 
    Necessary Arguments:
    - FilePath:       path to file
    - Pattern:        regular expression under which the searched quantity can be found in the img files MetaData
    Optional Arguments:
    - SubPattern:     regular expression which can be used to find the value of the searched quantity inside the pattern. default is SubPattern=r"[0-9]+[i]?[0-9]*"
    - Type:           datatype of the returned value. default is float
    given in IMG MetaData of an IMG File by FilePath, which fulfills a given regular expression
    """
    try:
        File = open(FilePath, 'rb')
        File_Inread = File.read()
        Data_StartPos = File_Inread.find(b'\x65\x6E\x74\x3d\x22\x22')
        MetaData = File_Inread[64: Data_StartPos + 6].decode("utf8")
        return Type(re.search(SubPattern, (re.search(Pattern, MetaData).group())).group())
    except:
        return None


# NEEDS REVISION
def getMetaDataInfoFromIMG_MetaData(MetaData, Pattern, SubPattern=r"[0-9]+[.]?[0-9]*", SubberPattern=None, Type=float):
    """
    ATTENTION: THIS FUNCTION HAS NO ERRORHANDLING
    returns Quantity which is saved in a given img MetaData. In case of failure it returns None 
    Necessary Arguments:
    - FilePath:       path to file
    - Pattern:        regular expression under which the searched quantity can be found in the img files MetaData
    Optional Arguments:
    - SubPattern:     regular expression which can be used to find the value of the searched quantity inside the pattern. default is SubPattern=r"[0-9]+[i]?[0-9]*"
    - Type:           datatype of the returned value. default is float
    given in IMG MetaData of an IMG File by FilePath, which fulfills a given regular expression
    """
    if SubberPattern is None:
        return Type(re.search(SubPattern, (re.search(Pattern, MetaData).group())).group())
    else:
        return Type(re.search(SubberPattern, re.search(SubPattern, (re.search(Pattern, MetaData).group())).group()).group())



def getMetaDataInfoFromSPE(Spe, Quantity):
    """
    returns quantity which is saved in the metadata of a spe file
    Necessary Arguments:
    - Spe:          path to spe file
    - Quantity:     searched quantity: For more information about the supported quantities look up for the function "getMetaDataInfoFromSpeFile"
    """
    try:
        SPEFile = sl.load_from_files([Spe])
        return getMetaDataInfoFromSPEFile(SPEFile, Quantity)
    except:
        return None



def getMetaDataInfoFromSPEFile(SPEFile, Quantity):
    """
    ATTENTION: NO ERRORHANDLING INCLUDED
    returns quantity SpeMetaData
    Necessary Arguments:
    - SPEFile:      Spe Object, created by spe_loaders.load_from_files([])
    - Quantity:     quantity which can be found in the spefiles xmltree. Currently supported quantities are: "ExpTime", "WaveLength", "CenterWaveLength", "Grating". Every other quantity returns None
    """
    if Quantity == "ExpTime":
        return int(SPEFile.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.ShutterTiming.ExposureTime.cdata)
    elif Quantity == "WaveLength":
        return np.array([float(f) for f in (str(SPEFile.footer.SpeFormat.Calibrations.WavelengthMapping.Wavelength.cdata).split(','))])
    elif Quantity == "CenterWaveLength":
        return int(SPEFile.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Spectrometers.Spectrometer.Grating.CenterWavelength.cdata)
    elif Quantity == "Grating":
        Grating = SPEFile.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Spectrometers.Spectrometer.Grating.Selected.cdata
        return getMetaDataInfoFromIMG_MetaData(Grating, r"nm,[0-9]+", Type=int)
    else:
        return None



# Basic Classes:

# NEEDS REVISION
def getParameters(FilePath, Extension, Parameters):
    """
    ATTENTION: THIS FUNCTION USES the bad getMetaDataInfoFromIMG AND NEEDS A REVISION
    returns dictionary of Parameters which can be searched in the filepath or the files metadata
    Necessary Arguments:
    - FilePath:     path to file
    - Extension:    file extension
    - Parameters:   dict of the searched Parameters. The dict has to have a specific structure to work: {"KEY1": [QUANTITYPLACE, ADDARGUMENT1, ADDARGUMENT2...]}
                    
                    The Function has three methods for searching:
                     QUIANTITYPLACE is 'd' for direct:      Then their is only one ADDARGUMENT which is the actual value
                                    is 'f' for filepath:    Then the ADDARGUMENTS are Pattern, SubPattern, Substitution and Type. For more Information read the doc of "getMetaDataInfoFromFilePath"
                                    is 'm' for metadata:    Then the ADDARGUMENTS differ for .spe and .img files:
                                                            For .spe files they are Quantity. For more Information read the doc of "getMetaDataInfoFromSPEFile"
                                                            For .img files they are Pattern, SubPattern and Type. For more Information read the doc of "getMetaDataInfoFromIMG"


    """
    assert type(Parameters) is dict, "The Structure you are using is not correct Youre using" + str(type(Parameters))
    MyParameters = {}
    Keys = Parameters.keys()
    for f in Keys:
        if Parameters[f][0] == 'd':
            MyParameters[f] = Parameters[f][1]
        elif Parameters[f][0] == 'f':
            MyParameters[f] = getMetaDataInfoFromFilePath(FilePath, *Parameters[f][1:])
        elif Parameters[f][0] == 'm':
            if Extension == '.img':
                MyParameters[f] = getMetaDataInfoFromIMG(FilePath, *Parameters[f][1:])
            elif Extension == '.spe':
                MyParameters[f] = getMetaDataInfoFromSPE(FilePath, Parameters[f][1])
        else:
            raise
    return MyParameters


class Mes_Object:
    def __init__(self, FilePath, Parameters=None, OD=None, ExpTime=None, SaveFullMetaData=False):
        NameInfo = getBaseNameAndExtFromPath(FilePath)
        self.Path = os.path.dirname(FilePath)
        self.Name = NameInfo[0]
        self.Ext = NameInfo[1]
        self.Parameters = getParameters(FilePath, self.Ext, Parameters) if Parameters is not None else {}
        self.Static_Parameters = {}
        self.OwnStuff = {}

        if self.Ext == '.img':
            self.Data = getDataFromIMG(FilePath)
            MetaData = getMetaDataFromIMG(FilePath)
            self.CalibrationState = {"OD": getMetaDataInfoFromIMG_MetaData(MetaData, r"Filter=\"Pos[0-9]{2}\"", r"[0-9]{2}") - 1,
                                     "MCP": getMetaDataInfoFromIMG_MetaData(MetaData, r"MCP Gain=\"[0-9]+[.]?[0-9]*\""),
                                     "ExpTime": getMetaDataInfoFromIMG_MetaData(MetaData, r"ExposureTime=[0-9]+", r"[0-9]+"),
                                     "ExpNumber": getMetaDataInfoFromIMG_MetaData(MetaData, r"NrExposure=[0-9]+", r"[0-9]+")
                                    }
            self.CalibrationOrigin = {
                                    "ExpTime": self.CalibrationState["ExpTime"],
                                    "ExpNumber": self.CalibrationState["ExpNumber"],
                                    "OD": self.CalibrationState["OD"],
                                    "MCP": self.CalibrationState["MCP"]
                                     }
            self.CalibrationInfo = {"BG": None,
                                    "Calibration": None,
                                   }
            if getMetaDataInfoFromIMG_MetaData(MetaData, r"Mode=\"[a-z,A-Z]+\"", r"=\"[a-z,A-Z]+\"", r"[a-z,A-Z]+", str) != "Focus":
                self.Static_Parameters["TimeRange"] = getMetaDataInfoFromIMG_MetaData(MetaData, r"Time Range=\"[0-9]+[.]?[0-9]*\"")
                self.Static_Parameters["TimeScale"] = getMetaDataInfoFromIMG_MetaData(MetaData, r"ScalingYScale=[0-9]+[.]?[0-9]*")
            if SaveFullMetaData is True:
                self.CalibrationInfo["MetaData"] = MetaData

        elif self.Ext == '.spe':
            self.Data = getDataFromSPE(FilePath)
            self.CalibrationState = {"OD": getMetaDataInfoFromFilePath(FilePath, "OD[0-9]+") if OD is None else OD,
                                     "ExpTime": getMetaDataInfoFromSPE(FilePath, "ExpTime")
                                    }
            self.CalibrationOrigin = {         
                                    "ExpTime": self.CalibrationState["ExpTime"],
                                    "OD": self.CalibrationState["OD"]
                                     }
            self.CalibrationInfo = {
                                    "BG": None,
                                    "Calibration": None
                                   }
            # Addition of optional WaveLengths
            if getMetaDataInfoFromSPE(FilePath, "CenterWaveLength") != 0:
                self.Static_Parameters["CenterWaveLength"] = getMetaDataInfoFromSPE(FilePath, "CenterWaveLength")
                self.Static_Parameters["Grating"] = getMetaDataInfoFromSPE(FilePath, "Grating")
                self.CalibrationInfo["WaveLength"] = getMetaDataInfoFromSPE(FilePath, "WaveLength")


        elif self.Ext == 'csv':
            self.Data = getDataFromCSV(FilePath)
            self.CalibrationState = {"OD": getMetaDataInfoFromFilePath(FilePath, "OD[0-9]+") if OD is None else OD,
                                     "ExpTime": getMetaDataInfoFromFilePath(FilePath) if ExpTime is None else ExpTime
                                    }
            self.CalibrationOrigin = {
                                    "ExpTime": self.CalibrationState["ExpTime"],
                                    "OD": self.CalibrationState["OD"]
                                     }
            self.CalibrationInfo = {"BG": None,
                                    "Calibration": None
                                   }


def saveMES_ObjectDataAsCSV(MES_Object, SavePath):
    if os.path.isdir(SavePath):
        SavePath = os.path.join(SavePath, (MES_Object.Name + ".csv"))
    np.savetxt(SavePath, MES_Object.Data, delimiter=',')


def saveMES_ObjectASIMG(MES_Object, SavePath):
    assert(MES_Object.Ext == '.img'), "Data based on wrong Filetype, it has to be '.img' but it is" + MES_Object.Ext
    assert("MetaData" in MES_Object.CalibrationInfo.keys()), "MetaData is missing, Yo have to set SaveFullMetaData=True"
    if os.path.isdir(SavePath):
        SavePath = os.path.join(SavePath, (MES_Object.Name + ".img"))
    elif not os.path.isfile(SavePath):
        print("Given SavePath is neither DirectoryPath nor FilePath")
        raise
    File = open(SavePath, 'wb')
    BinaryBegin = b'IM\xdc\x11\x00\x03\x80\x02\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    BinaryMeta = bytearray(MES_Object.CalibrationInfo["MetaData"], 'utf8')
    BinaryData = bytearray(MES_Object.Data.astype('int32'))
    Binary = BinaryBegin + BinaryMeta + BinaryData
    File.write(Binary)
    File.close()


def DictToList(Dict):
    Keys = Dict.keys()
    return [(f, Dict[f]) for f in Keys]


def removeDoublesMES_ObjectsFromList(MES_Objects):
    return list(toolz.unique(MES_Objects, key=lambda w: tuple(DictToList(w.Parameters))))


def getSetOfDicts(ListOfDicts):
    return list(toolz.unique(ListOfDicts, key=lambda w: tuple(DictToList(w))))


# Calibration
# -- BASE Functions


def removeBG(Data_2D, BGData_2D, NoNegative=True):
    '''
    Takes two numpy Arrays of equal dimension and size and returns Difference
    '''
    Data_2D = Data_2D-BGData_2D
    if NoNegative is True:
        Data_2D[Data_2D < 0] = 0
    return Data_2D


def cutData_2D(Data_2D, Borders_2D):
    '''
    Extracts Submatrix of an given numpy 2D Array with the given Borders
    '''
    return Data_2D[Borders_2D[0]:Borders_2D[1], Borders_2D[2]:Borders_2D[3]]


def cutData(MES_Object, Borders_2D):
    MES_Object.Data = cutData_2D(MES_Object.Data, Borders_2D)
    return MES_Object


def calibrateExpTimeBase(Data_2D, ExpTime, normExpTime=100):
    Factor = normExpTime / ExpTime
    return Data_2D * Factor


def calibrateExpNumberBase(Data_2D, ExpNumber, normExpNumber=100):
    Factor = normExpNumber / ExpNumber
    return Data_2D * Factor


def Mean_Simple(Data_2D):
    return np.mean(np.ravel(Data_2D))


def Mean_OfOverRelTreshold(Data_2D, RelThreshold):
    Threshold = RelThreshold * max(np.ravel(Data_2D))
    Data = Data_2D.ravel()
    return np.mean(np.array([f for f in Data if f > Threshold]))


def normRowwise(Data_2D, ScalingFactor=1):
    Maxima = np.max(Data_2D, axis=(1), keepdims=True)
    Maxima[Maxima <= 0] = 1
    return Data_2D / Maxima * ScalingFactor


def convertWaveLengthToEnergy(WaveLength, n=1.0002702):
    """
    calculates the Energy for photons of a given wavelength
    Necesary Arguments:
    WaveLength:     Input Wavelength
    Optional Arguments:
    n:              real refractive index ofthe media (normally air). default is n=1.0002702. more specific values fpor air can be calculated by using the calculator on the website https://emtoolbox.nist.gov/Wavelength/Ciddor.asp
    """
    assert(WaveLength > 0 and n > 0), "Wavelength and n must be positive"
    return sc.h * sc.c / (WaveLength*sc.e*10**(-9)*n)

def linearizeWaveLength(MES_Object, CentralPixel, FitInterval,Unit="nm", n=1.0002702, IgnoreBordersOfCalibration=False, returnError = False):
    """
    NOT YET TESTED
    returns the parameter of a linear fit of the wavelenth calibration, in case of failure it returns None
    Necessary Arguments:
    - MES_Object: MES_Object which includes the WaveLength Calibration
    - CentralPixel: Central Pixel of the area where the fit should be done
    - FitInterval: Size of the FitInterval: The Interval then reachs from CentralPixel-FitInterval: CentralPixel+FitInterval
    Optional Arguments:
    - n: refractive Index of the used medium. defaul is 1.0002702 for air at 770nm.
    - IgnoreBordersOfCalibration: Bool to IgnoreIf the Fittingarea can be  assymetric in case parts of the original one are out of scope, Default is False
    - returnError: Bool to return also the error of the linearization. default is False
    """
    assert(Unit == "nm" or Unit=="eV"), "Only supported Units are either 'nm' or 'eV'"
    assert(MES_Object.Ext == '.spe'),"The Input MES_Object was not created from an spefile"
    assert("CenterWaveLength" in MES_Object.Static_Parameters.keys()),"The Input MES_Object was created from a file without a measured energy dimension"
    CalibrationTable = MES_Object.CalibrationInfo["WaveLength"]
    if IgnoreBordersOfCalibration is False:
        assert(CentralPixel-FitInterval >= 0 and CentralPixel+FitInterval < len(CalibrationTable)),"Parts of your FitInterval are of scope of the Calibrationfile"# Last Input of Len could be exchanged by the lenth of the calibrationfile if known
        StartPixel = CentralPixel-FitInterval
        EndPixel = CentralPixel+FitInterval
    else:
        StartPixel = max(CentralPixel-FitInterval, 0)
        EndPixel = min(CentralPixel-FitInterval, len(CalibrationTable)-1)
    if Unit=='eV':
        CalibrationTable = [convertWaveLengthToEnergy(f,n) for f in CalibrationTable]
    
    def linear(x,a,b):
        return a*x+b
    a_s = (CalibrationTable[EndPixel]-CalibrationTable[StartPixel])/(EndPixel-StartPixel)
    b_s = CalibrationTable[StartPixel] - StartPixel * a_s
    try:
        Parameters, Error = curve_fit(linear, CalibrationTable[StartPixel:EndPixel+1],p0=[a_s,b_s])
    except:
        print("CurveFit could not converge")
        return None

    if returnError is False:
        return np.array(Parameters)
    else:
        Error = np.sqrt(np.diag(Error))
        return uarray(Parameters, Error)

        

#--OBJECT ORIENTED Functions


def calibrateExpTime(MES_Object, normExpTime=100):
    assert("ExpTime" in MES_Object.CalibrationState)
    MES_Object.Data = calibrateExpTimeBase(MES_Object.Data, MES_Object.CalibrationState["ExpTime"], normExpTime)
    MES_Object.CalibrationState["ExpTime"] = normExpTime
    return MES_Object


def calibrateExpNumber(MES_Object, normExpNumber=100):
    assert("ExpNumber" in MES_Object.CalibrationState)
    MES_Object.Data = calibrateExpNumberBase(MES_Object.Data, MES_Object.CalibrationState["ExpNumber"], normExpNumber)
    MES_Object.CalibrationState["ExpNumber"] = normExpNumber
    return MES_Object


def remove_BG_correct(MES_Object, BG_Objects, NoNegative=True):
    assert (MES_Object.CalibrationState == MES_Object.CalibrationOrigin and MES_Object.CalibrationInfo["BG"] is None)
    for f in BG_Objects:
        if MES_Object.CalibrationState == f.CalibrationState and MES_Object.Static_Parameters == f.Static_Parameters:
            print(MES_Object.Path)
            MES_Object.Data = removeBG(MES_Object.Data, f.Data, NoNegative)
            MES_Object.CalibrationInfo["BG"] = f.Name
            return MES_Object
    print("Could not find BG for State", MES_Object.CalibrationState, MES_Object.Static_Parameters)
    raise


def createCalibrationMatrix(MES_Objects, MeanFunction, SavePath=None):
    # get all States and Connections: The Condition is that there are different States with the same Parameters. Additionally the Static Parameters (like TimeRange for StreakData) has to be the same for all data
    StateConnections = []
    assert all([(f.Parameters is not None) for f in MES_Objects]) and all([(f.Static_Parameters == MES_Objects[0].Static_Parameters) for f in MES_Objects])
    if all([f.CalibrationState == MES_Objects[0].CalibrationState for f in MES_Objects]):
        return {"Converter": [[MES_Objects[0].CalibrationState]], "Matrix": np.ones(1, 1)}
    for f in MES_Objects:
        for g in MES_Objects:
            if f.Parameters == g.Parameters and f.CalibrationState != g.CalibrationState:
                Factor = MeanFunction[0](f.Data, *MeanFunction[1]) / MeanFunction[0](g.Data, *MeanFunction[1])
                NewConnection = [f.CalibrationState, g.CalibrationState, Factor]
                if NewConnection not in StateConnections:
                    StateConnections.append(NewConnection)
                    StateConnections.append([g.CalibrationState, f.CalibrationState, 1/Factor])

    assert(StateConnections != []), "Your Data is not connected to each other, Calibration is impossible, THAT's YOUR FAULT"
    States = [dict(t) for t in set(tuple(sorted(d.items())) for d in list(np.array(StateConnections).T)[0])]
    States = sorted(States, key=itemgetter(*sorted(States[0].keys())))

    #simplifying the statedescriptions to use them as base for a Graph in networkx
    States = [States, list(range(0, len(States)))]
    StateConnectionsSimplified = []
    for i in range(0, len(StateConnections)):
        for j in range(0, len(States[0])):
            if StateConnections[i][0] == States[0][j]:
                start = States[1][j]
            if StateConnections[i][1] == States[0][j]:
                finish = States[1][j]
        StateConnectionsSimplified.append((start, finish, {"Factor": StateConnections[i][2]}))

    # create Graph from Data and check if Data is useable
    G = kx.DiGraph()
    G.add_nodes_from(States[1])
    G.add_edges_from(StateConnectionsSimplified)
    assert kx.is_strongly_connected(G), "Your Data is not completely connected to each other, YOUR FAULT"

    #create CalibrationMatrix by searching for shortest paths in graph
    CalibrationMatrix = np.identity(len(States[1]))
    for i in range(0, len(States[1])):
        for j in range(i + 1, len(States[1])):
            Way = kx.shortest_path(G, i, j)
            PathGraph = kx.path_graph(Way)
            CalibrationMatrix[j, i] = np.prod([G.edges[f]["Factor"] for f in PathGraph.edges()])
    for i in range(0, len(States[1])):
        for j in range(i + 1, len(States[1])):
            CalibrationMatrix[i, j] = 1 / CalibrationMatrix[j, i]
    CalibrationMatrix = CalibrationMatrix.T

    #Save and Return
    Calibration = {"Converter": States, "Matrix": CalibrationMatrix}
    if SavePath is not None:
        with open(SavePath, 'wb') as Handler:
            pickle.dump(Calibration, Handler)
    return Calibration


def SaveObjectAsFile(Object, SavePath):
    with open(SavePath, 'wb') as Handler:
        pickle.dump(Object, Handler)


def loadObjectFromFile(ObjectPath):
    with open(ObjectPath, 'rb') as Handler:
        return pickle.load(Handler) 

def getCalibrationFromMatrix(StartState, AimState, Calibration):
    Found = [False, False]
    for i in range(0, len(Calibration["Converter"][0])):
        if Calibration["Converter"][0][i] == StartState:
            StartState = Calibration["Converter"][1][i]
            Found[0] = True
            break
    for i in range(0, len(Calibration["Converter"][0])):
        if Calibration["Converter"][0][i] == AimState:
            AimState = Calibration["Converter"][1][i]
            Found[1] = True
            break
    assert all(Found)
    return Calibration["Matrix"][AimState, StartState]


def showAvaibleCalibrationStates(Calibration):
    for f in Calibration["Converter"][0]:
        print(f)


def showCalibrationMatrix(Calibration):
    np.set_printoptions(precision=2)
    print(Calibration["Matrix"])


def calibrate_MESObject_ToState(MES_Object, AimState, Calibration):
    MES_Object.Data = (MES_Object.Data).astype('float64') * getCalibrationFromMatrix(MES_Object.CalibrationState, AimState, Calibration)
    MES_Object.CalibrationState = AimState
    MES_Object.CalibrationInfo["Calibration"] = Calibration
    return MES_Object


# Profiles:


def getDataFromHPDTAProfile(FilePath, Delimiter=',', Type=float, AlsoReturnHeader=False):
    File = open(FilePath)
    Lines = File.readlines()
    DataLines = [f for f in Lines if f[0].isdigit()]
    DataLines = [(f.split(Delimiter)) for f in DataLines]
    #for f in DataLines:
    #    f[-1] = f[-1][:-2]
    for i in range(0, len(DataLines)):
        for j in range(0, len(DataLines[i])):
            if DataLines[i][j] == '':
                DataLines[i][j] = '0'
    DataLines = np.array(DataLines).astype(Type)
    if AlsoReturnHeader is True:
        Header = [f for f in Lines if not f[0].isdigit()]
        return [DataLines.T, Header]
    else:
        return [DataLines.T, None]


def writeHPDTAProfile(Data, Header, SavePath, Scaling=100000):
    File = open(SavePath, 'w')
    File.write("".join(Header))
    Data = Data.T
    Data[1] = Data[1]*Scaling
    Data = Data.T
    np.savetxt(File, Data, delimiter=',',fmt=['%.3f','%.2f'])
    File.close()


# Other Stuff (It could change its folder, so keep care)
def calcMagnifactionOfLensSystem(Lenses, PixelSize=20,WaveLength=770):
    """
    calculates the Magnificationfactor of a given lenssystem and returns the conversionfactor from px to µm for even number of lenses or to 1/µm for odd number of lenses
    Necessary Arguments:
    - Lenses:       List of Focusdistances of the lenssystem. the lenses must be ordererd like in the system. All lenses must be convex so f>0
    Optional Parameters:
    - PixelSize: Pixelsize of a pixel in the CCD. default is 20µm
    - WaveLength: approximate Wavelengt of the measuered Light. default is 770nm
    """
    assert(all([f>0 for f in Lenses])is True),"Alle Lenses have to be convex so f>0"
    if len(Lenses)%2 == 0: #lenses is even
        M = 1
        for i in range(0,len(Lenses)):
            if (i+1)%2 == 1:
                M = M / Lenses[i]
            else:
                M = M * Lenses[i]
        return PixelSize / M
    else:
        M = 1
        for i in range(0,len(Lenses)):
            if (i+1)%2 == 0:
                M = M / Lenses[i]
            else:
                M = M * Lenses[i]
        return (2*np.pi * PixelSize) / (WaveLength *10**(-3) *M)
