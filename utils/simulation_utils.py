import pyprismatic 
import pyprismatic as pr 
import os 
import pandas 
from typing import Union
from pathlib import Path
from . import general_utils as gen_utils
from . import dataframe_utils as df_utils
from . import simulation_utils as sim_utils


#### TAKE THE MASTER SIMULATION DATAFRAME AND SIMULATE THE IMAGES ####


def make_default_meta() -> pyprismatic.params.Metadata:
    """
    create the default pyprismatic simualtipnmetadata simulation 
    """
    meta = pr.Metadata(
    algorithm='multislice',
    E0=300,
    numSlices=10,
    sliceThickness=2,
    includeThermalEffects=True,
    numFP=1,
    interpolationFactorX=1,
    interpolationFactorY=1,
    potBound=2,
    probeDefocus=0,
    alphaBeamMax=8,
    probeSemiangle=2,
    realspacePixelSizeX=0.1,
    realspacePixelSizeY=0.1,
    scanWindowXMin=0.5,
    scanWindowXMax=0.5,
    scanWindowYMin=0.5,
    scanWindowYMax=0.5,
    numGPUs=1,
    #numStreamsPerGPU=12,
    alsoDoCPUWork=False,
    #numThreads=32,
    savePotentialSlices=True,
    saveProbe=True,
    save4DOutput=True,
    save3DOutput=False,
    save2DOutput=False,
    )

    return meta


def make_simulation_parameters_dictionary(row: pandas.Series,
                                        GPU:int = 1,
                                        no_pixels:Union[tuple,list] = (512,512)):
    """
    takes a pandas row and creates the simulation parameters into a dictionary
    """
    # create a dictionary of the parameters
    tmp_row = row.copy(deep=True)
    simulation_parms = {
        'probeSemiangle'        : tmp_row.probeSemiangle,
        'filenameOutput'        : str(tmp_row.filenameOutput),  # ensure it doesn't pass a pathlib object
        'filenameAtoms'         : str(tmp_row.structure_save_name),        # ensure it doesn't pass a pathlib object
        'probeSemiangle'        : tmp_row.probeSemiangle,
        'alphaBeamMax'          : tmp_row.probeSemiangle,
        'numFP'                 : tmp_row.numFP,
        'E0'                    : tmp_row.E0,
        'probeDefocus'          : tmp_row.probeDefocus,
        'sliceThickness'        : tmp_row.sliceThickness,
        'potBound'              : tmp_row.potBound,
        'algorithm'             : tmp_row.algorithm, # commented out 
        'randomSeed'            : tmp_row.simulation_seed,
        'numGPUs'               : GPU,
        'realspacePixelSizeX'   : tmp_row.realspacePixelSizeX,
        'realspacePixelSizeY'   : tmp_row.realspacePixelSizeY,
        'numSlices'             : tmp_row.numSlices 
        }

    return simulation_parms


def simulate_row(row: pandas.Series,
                GPU:int = 1,
                no_pixels:Union[tuple,list] = (512,512)):
    """
    simulate each row in the dataframe 

    accepts:
    row :       pandas.Series    : A row from the simulation dataframe
    GPU :       int              : number of GPUS per simualtion
    no_pixels:  tuple            : number of pxiels in output cbed 
    
    returns:
    None... but saves writes a simulation h5 file. 
    """
    
    
    # create a deep copy of the row
    tmp_row = row.copy(deep=True)

    # check if the file alredy exists if so stop here.
    if os.path.exists(tmp_row.filenameOutput):
        return None
    else: 
        pass

    # check the output folder exists if not make it 
    if not os.path.exists(Path(tmp_row.filenameOutput).parent):
        os.makedirs(Path(tmp_row.filenameOutput).parent)
    else:
        pass

    # create the simulation parameters 
    simulation_params = make_simulation_parameters_dictionary(tmp_row, GPU=GPU, no_pixels=no_pixels)

    # create the deafult pyprismatic meta object
    meta = make_default_meta()

    # update the meta object with parameters from the row
    for field, val in simulation_params.items():
            setattr(meta, field, val)

    #run the simulation
    meta.go(display_run_time=True, save_run_time=False)

    return None
    