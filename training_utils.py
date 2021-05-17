import pathlib
from pathlib import Path
import pandas
import pandas as pd
import numpy
import numpy as np
import h5py
from scipy import signal 
import os
from typing import Union


# These string manipulations should be replaced with more robust method earlier in the workflow 
def get_original_name(filenameAtoms:str):
    return filenameAtoms.split('/')[-1]

def get_chemical_name(filenameAtoms:str):
    original_name = filenameAtoms.split('/')[-1]
    return original_name.split('_')[1]

def get_rotation_name(filenameAtoms:str):
    original_name = filenameAtoms.split('/')[-1]
    return original_name.split('_')[-1][:-4]

def get_simulation_name(filenameOutput:str):
    return  filenameOutput.split('/')[-1]

def get_training_name(filenameOutput:str):    
    filename = filenameOutput.split('/')[-1]
    training_name = filename.split('_')[0]+'_training.h5'
    return training_name

def make_original_name(simulation_dataframe: pandas.DataFrame, str_name:str = 'orignal_name'):
    simulation_dataframe[str_name] = simulation_dataframe.filenameAtoms.map(get_original_name)
    return simulation_dataframe

def make_chemical_name(simulation_dataframe: pandas.DataFrame, str_name:str = 'chemical_name'):
    simulation_dataframe[str_name] = simulation_dataframe.filenameAtoms.map(get_chemical_name)
    return simulation_dataframe

def make_rotation_name(simulation_dataframe: pandas.DataFrame, str_name:str = 'rotation'):
    simulation_dataframe[str_name] = simulation_dataframe.filenameAtoms.map(get_rotation_name)
    return simulation_dataframe

def make_simulation_name(simulation_dataframe: pandas.DataFrame, str_name:str = 'simulation_name'):
    simulation_dataframe[str_name] = simulation_dataframe.filenameOutput.map(get_simulation_name)
    return simulation_dataframe

def make_training_name(simulation_dataframe: pandas.DataFrame, str_name:str = 'training_name'):
    simulation_dataframe[str_name] = simulation_dataframe.filenameOutput.map(get_training_name)
    return simulation_dataframe


def make_rotation_encoder(simulation_dataframe: pandas.DataFrame,
                          rotation_choices:list = None,
                          rotation_conditions:list = None,
                          default:str =None,
                          str_name:str ='rotation_type'):
    """
    Placeholder
    """
    if rotation_choices is None:
        rotation_choices = ['free', 'unrotated', 'zone']
    else: pass 
    
    if rotation_conditions is None:
        rotation_conditions = [simulation_dataframe.rotation.str.len() >3,
                                simulation_dataframe.rotation.str.contains('unrotated'), 
                                simulation_dataframe.rotation.str.len() ==3]
    else: pass
    
    simulation_dataframe[str_name] = np.select(rotation_conditions, rotation_choices, default=default)
    
    return simulation_dataframe


def augment_dataframe(simulation_dataframe: pandas.DataFrame, 
                      functions:list =[make_original_name,
                                       make_chemical_name, 
                                       make_rotation_name, 
                                       make_rotation_encoder,
                                       make_simulation_name,
                                       make_training_name] ):
    """
    Placeholder
    """
    for function in functions:
        simulation_dataframe = function(simulation_dataframe)
    
    return simulation_dataframe







def get_probe_and_cbeds(file_path: str):
    """
    Place holder
    
    """
    
    with h5py.File(file_path, 'r') as f:
        
        ### Potentials
        pots = [] # create empty list to append the potentials to
        # loop over all the potentials for each thickness
        for key in f['4DSTEM_simulation/data/realslices/'].keys():
            pots.append(f[f'4DSTEM_simulation/data/realslices/{key}/data'][...]) # append each potential
        pots = np.concatenate(pots, axis=0) # stack them into a giant stack each potential has differnet number of layers 
        
        
        ### CBEDS
        cbeds = [] # create empty list to append cbeds to
        for key in f['4DSTEM_simulation/data/datacubes/']:
            cbeds.append(f[f'4DSTEM_simulation/data/datacubes/{key}/data'][0,0,...])
        cbeds = np.swapaxes(np.stack(cbeds), 0, 2) # swapping the axis to be consistent with Colin's
        
        ### Probe
        probe = np.fft.fftshift(abs(f['4DSTEM_simulation/data/diffractionslices/probe/data'][...])**2)
        
        
    return pots, cbeds, probe

def get_thicknesses(example: pandas.Series, cbeds: numpy.array):
    """
    Placeholder
    """
    thicknesses = [i * example.numSlices * example.sliceThickness + (example.sliceThickness * example.numSlices)
               for i in range(cbeds.shape[2])]
    return thicknesses
    
def get_qx_qy(example: pandas.Series, cbeds: numpy.array):
    """
    Placeholder
    
    """
    pixel_size_AA = example.realspacePixelSizeX * 2
    N = [*cbeds.shape]
    qx = np.sort(np.fft.fftfreq(N[0], pixel_size_AA)).reshape((N[0], 1, 1))
    qy = np.sort(np.fft.fftfreq(N[1], pixel_size_AA)).reshape((1, N[1], 1))
    
    return qx, qy

def scale_probe(probe: numpy.array, max_val=1, ):
    """
    Placeholder
    """
    
    int_scale = np.divide(max_val, probe.max())
    probe *= int_scale
    
    return probe

def scale_cbeds(cbeds: numpy.array, probe: numpy.array, max_val=1):
    """
    Placeholder
    """
    int_scale = np.divide(max_val, probe.max())
    cbeds *= int_scale
    return cbeds


def get_qz_dataQZ_dataPot(example: pandas.Series,
                          cbeds: numpy.array, 
                          pots: numpy.array, 
                          thicknesses: list,
                          qz_sigma:float =1/20):
    """
    Placeholder    
    """
    N = [*cbeds.shape]
    wx = signal.windows.tukey(2*N[0]).reshape(2*N[0], 1, 1)
    wy = signal.windows.tukey(2*N[1]).reshape(1, 2*N[1], 1)
    scale = (1/4) / np.prod(N[:2])

    data_pot = np.zeros_like(cbeds)
    data_qz = np.zeros_like(cbeds)
    xAA = np.array([*[i for i in range(0, cbeds.shape[0]//2)], 
           *[j + cbeds.shape[0]*2 for j in range(-cbeds.shape[0]//2, 0)]])
    yAA = np.array([*[i for i in range(0, cbeds.shape[1]//2)], 
           *[j + cbeds.shape[1]*2 for j in range(-cbeds.shape[1]//2, 0)]])

    for index in range(cbeds.shape[-1]):
        #print(thicknesses[index]//meta.sliceThickness) 

        num_planes = thicknesses[index]//example.sliceThickness
        inds_range = np.array(thicknesses) // example.sliceThickness
        inds_range = inds_range[:index]
        wz = signal.windows.tukey(num_planes).reshape(1, 1, num_planes)
        qz =  np.fft.fftfreq(num_planes, example.sliceThickness).reshape((1, 1, num_planes))
        qz_filter = np.exp(-qz**2/(2*qz_sigma**2))

        pot_fft = np.abs(np.fft.fftn(pots[..., :num_planes] * (scale * wx *wy*wz)))*qz_filter


        assert len(pot_fft.shape) == 3

        data_pot[...,index] = np.fft.fftshift(np.sum(pot_fft[xAA][:,yAA],2))

        data_qz[...,index] = np.fft.fftshift(np.sum(pot_fft[xAA][:,yAA] * qz, 2))/data_pot[...,index]
    
    return qz, data_pot, data_qz

def h5_training_writer(output_filename: str, data_dict: dict):
    """
    writes the hdf5 outputfile with the training parameters
    
    Inputs: 
    output_filename (str) -> string with a path to write the name 
    data_dict (dict) -> training values to write to file
    
    
    Returns:
    None
    """
    
    with h5py.File(output_filename, 'w') as hf:
        # loop over the enteries in data_dict
        for key, val in data_dict.items():
            hf.create_dataset(key, data=val)
    return None

def save_training_data(thicknesses: list,
                       qx:numpy.array,
                       qy:numpy.array,
                       qz:numpy.array,
                       dataProbe:numpy.array,
                       dataMeas:numpy.array,
                       dataPots:numpy.array,
                       dataQz:numpy.array,
                       output_filename: str):
    """
    Placeholder
    """
    datasets_dict = {'thicknesses': thicknesses,
                 'qx': qx,
                 'qy': qy,
                 'qz': qz,
                 'dataProbe': dataProbe,
                 'dataMeas': dataMeas,
                 'dataPots': dataPots,
                 'dataQz': dataQz
    }
    
    h5_training_writer(output_filename, datasets_dict)
    
    return None

def convert_series_to_training_data(example:pandas.Series,
                                       training_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                                       data_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath]):
    """
    Placeholder 
    """
    #Should I add a try statement here? 
    
    
    
    # convert the paths to pathlib type  
    data_path = Path(data_path)
    training_path = Path(training_path)
    
    #make the training_path folder if it doenst exist 
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    else: pass
    
    #get the names full names for input and output
    input_name = data_path / example.simulation_name # output of the simulation path
    
    save_name = training_path / example.training_name
    print()
    print(example.training_name)
    
    print(save_name)
    #check that the file simulation exists and the training data doesnt.
    if os.path.exists(input_name) and not os.path.exists(save_name):
        
        # open simulation name and extract pots, cbeds and probe
        pots, cbeds, probe = get_probe_and_cbeds(input_name)

        #get cell thicknesses 
        thicknesses = get_thicknesses(example, cbeds)

        #get qx, qy
        qx,qy = get_qx_qy(example, cbeds)


        #scale probe and cbeds
        cbeds = scale_cbeds(cbeds, probe)
        probe = scale_probe(probe)

        #get qz, dataQz, and dataPot
        qz, data_pot, data_qz = get_qz_dataQZ_dataPot(example, cbeds, pots, thicknesses)

        #save the training data 
        save_training_data(thicknesses, qx, qy,qz, probe, cbeds, data_pot, data_qz, save_name)
    else:
        print(f'{input_name} does not exist')

def convert_dataframe_to_training_data(index: int,
                                       df:pandas.DataFrame,
                                       training_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                                       data_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath]):
    """
    DON'T USE
    
    Placeholder 
    """
    #Should I add a try statement here? 
    # I should add a check if file exists. 
    
    # get a specific example(row) from the dataframe
    example = df.iloc[index]
    
    # convert the paths to pathlib type  
    data_path = Path(data_path)
    training_path = Path(training_path)
    
    #make the training_path folder if it doenst exist 
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    else: pass
    
    #get the names full names for input and output
    input_name = data_path / example.simulation_name # output of the simulation path
    
    save_name = training_path / example.training_name
    print()
    print(example.training_name)
    
    print(save_name)
    #check that the file simulation exists and the training data doesnt.
    if os.path.exists(input_name) and not os.path.exists(save_name):
        
        # open simulation name and extract pots, cbeds and probe
        pots, cbeds, probe = get_probe_and_cbeds(input_name)

        #get cell thicknesses 
        thicknesses = get_thicknesses(example, cbeds)

        #get qx, qy
        qx,qy = get_qx_qy(example, cbeds)


        #scale probe and cbeds
        cbeds = scale_cbeds(cbeds, probe)
        probe = scale_probe(probe)

        #get qz, dataQz, and dataPot
        qz, data_pot, data_qz = get_qz_dataQZ_dataPot(example, cbeds, pots, thicknesses)

        #save the training data 
        save_training_data(thicknesses, qx, qy,qz, probe, cbeds, data_pot, data_qz, save_name)
    else:
        print(f'{input_name} does not exist or {example.training_name} already exists')
    