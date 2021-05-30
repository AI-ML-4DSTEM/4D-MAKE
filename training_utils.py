import enum
import pathlib
from pathlib import Path
from typing_extensions import ParamSpec
from ase import atoms
from numpy.core.getlimits import MachArLike
from numpy.core.overrides import array_function_from_dispatcher
import pandas
import pandas as pd
import numpy
import numpy as np
import h5py
from scipy import signal 
import os
from typing import Union
import csv
import itertools as it 

# These string manipulations should be replaced with more robust method earlier in the workflow - Should be reduant now
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


###### ALLL ABOVE SHOULD BE REDUDNAT ########



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
        # need to correct the shape of the probe (e.g. half the size of it)
        probe = np.fft.fftshift(abs(f['4DSTEM_simulation/data/diffractionslices/probe/data'][::2,::2])**2)
        
        
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
    None-Ewald Sphere    
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

def wavev(E):
    """
    Evaluate the relativistically corrected wavenumber of an electron with energy E.
    Energy E must be in electron-volts, see Eq. (2.5) in Kirkland's Advanced
    Computing in electron microscopy

    STOLEN FROM PYMULTISLICE (HAMISH BROWN REPO)
    """
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc

def load_parameters(path):
    """
    Function to load parameters from a CSV file.
    MODIFIED FROM ABTEM
    """
    # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    parameters = {}
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        keys = next(reader)
        for _, row in enumerate(reader):
            values = list(map(float, row))
            parameters[int(row[0])] = dict(zip(keys, values))
    return parameters

# I think this function isn't required now
# def _set_path(path):
    """
    Internal function to set the parametrization data directory.
    MODIFIED FROM ABTEM
    """
    #_ROOT = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_ROOT, 'data', path)

def load_kirkland_parameters(path:Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                             convert=True):
    """
    Function to load the Kirkland parameters (doi:10.1007/978-1-4419-6533-2).
    MODIFIED FROM ABTEM   
    """
    parameters = {}

    for key, value in load_parameters(path).items():
        a = np.array([value[key] for key in ('a1', 'a2', 'a3')])
        b = np.array([value[key] for key in ('b1', 'b2', 'b3')])
        c = np.array([value[key] for key in ('c1', 'c2', 'c3')])
        d = np.array([value[key] for key in ('d1', 'd2', 'd3')])
        if convert:
            a = np.pi * a
            b = 2. * np.pi * np.sqrt(b)
            c = np.pi ** (3. / 2.) * c / d ** (3. / 2.)
            d = np.pi ** 2 / d

        parameters[key] = np.vstack((a, b, c, d))

    return parameters

def kirkland_projected_fourier(k, p):
    """
    TAKEN FROM ABTEM
    redudant now
    """
    f = (4 * np.pi * p[0, 0] / (4 * np.pi ** 2 * k ** 2 + p[1, 0] ** 2) +
         np.sqrt(np.pi / p[3, 0]) * p[2, 0] * np.pi / p[3, 0] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 0]) +
         4 * np.pi * p[0, 1] / (4 * np.pi ** 2 * k ** 2 + p[1, 1] ** 2) +
         np.sqrt(np.pi / p[3, 1]) * p[2, 1] * np.pi / p[3, 1] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 1]) +
         4 * np.pi * p[0, 2] / (4 * np.pi ** 2 * k ** 2 + p[1, 2] ** 2) +
         np.sqrt(np.pi / p[3, 2]) * p[2, 2] * np.pi / p[3, 2] * np.exp(-np.pi ** 2 * k ** 2. / p[3, 2]))
    return f

def make_zone_vectors_training(min_vector:int=0, max_vector:int=3, 
                      max_number:int=None, seed:int=None):# -> list[tuple[int,int,int]]: Not sure how to do this correctlyVEC
    """
    makes zone axis vectors 
    ADD AN ERROR OR WARNING ABOUT IF MAX_NUMBER IS LESS THAN NO PERMUTATIONS (EDGE CASE)
    
    """
    #add something about seed being none
    if seed == None:
        seed = 42
    else:
        pass


    zones = np.array([i for i in it.product(range(min_vector, max_vector+1), repeat=3)])
    return zones


def atomFactor(atomID: int,
                q: float,
                params: dict) -> float:
    """[summary]

    Args:
        atomID (int): [description]
        q (float): [description]
        params (dict): [description]

    Returns:
        float: [description]
    """

    ap = params[atomID].ravel() # being lazy and I want to copy and paste from Colin
    q2 = q**2
    f = (
        ap[0] / (q2 + ap[1]) + 
        ap[2] / (q2 + ap[3]) + 
        ap[4] / (q2 + ap[5]) + 

        ap[6] * np.exp((-ap[7] * q2)) + 
        ap[8] * np.exp((-ap[9] * q2)) +
        ap[10] * np.exp((-ap[11] * q2))
        )
    return f 



def create_atomsFrac(super_cell: numpy.ndarray,
                    cell_size: Union[list,tuple,numpy.ndarray]) -> numpy.ndarray:
    """
    Create a structure with fractionalised coordinates, from super cell with real space coordinates
    accepts: 
    super_cell : numpy.ndarray Shape N x 4 [x, y, z, atom_id]
    cell_size : 3 unit cell vectors [x, y, z]

    x,y,z can be > 1 

    returns:
    atomsFrac :  numpy.ndarray Shape N x 4 [x', y', z', atom_id]
    x',y',z' <= 1   

    """
    # add an extra element to cell_size
    cell_vec = np.array([*cell_size,1])

    atomsFrac = super_cell / cell_vec

    return atomsFrac

    

def calcSF(atomsFrac:numpy.ndarray,
            latVec:numpy.ndarray,
            kMax:Union[int, float, numpy.ndarray],
            params:dict) -> dict:
    """
    TRANSLATING COLIN's MATLAB FUNCTION

    This script computes the 3D structure factors from atom positons and
    lattice vectors.
    """
    # not sure what this is doing in here, but placeholder for now  
    scale = 0.1 * 10

    # calculate inverse lattice vectors
    latInv = np.linalg.inv(latVec)
    q1 = latInv[0]
    q2 = latInv[1]
    q3 = latInv[2]

    # Find shortest vertex distance for tiling   
    qAll = np.array(
        [ 
            q1,
            q2,
            q3,
            q1+q2,
            q1-q2,
            q1+q3,
            q1-q3,
            q2+q3,
            q2-q3,
            q1+q2+q3,
            q1-q2+q3,
            q1+q2-q3,
            q1-q2-q3
        ]
        )


    qLength = np.sqrt(np.sum(qAll**2,1));
    qMin = min(qLength);
    numTile = round(kMax / qMin);
    
    # Tile the inverse lattice vectors
    v = np.arange(-numTile, numTile+1)
    # why are these swapped
    ya,xa,za = np.meshgrid(v,v,v) # Don't think this is needed anymore

    hkl = make_zone_vectors_training(-numTile, numTile) # outputs an array of zone vectors 
    # I don't get this part
    qVec = np.dot(hkl, latInv)
    
    # Crop inverse lattice vectors outside of kMax
    qVec2 = np.sum(qVec**2,1)
    mask = qVec2 > kMax**2
    hkl =  hkl[mask]
    qVec = qVec[mask]
    qVec2 = qVec2[mask]
    qVec1 = np.sqrt(qVec2);


    # create an empty array
    fAll= np.zeros(shape=(qVec1.shape[0], atomsFrac.shape[0]))
    
    # evaluate the single atom scattering factors
    # could I vectorize or do something smarter here? 
    for index in range(atomsFrac.shape[0]):
        fAll[...,index] = atomFactor(atomsFrac[index,-1].astype(int), qVec1, params=params)

    # calculate the structure factors
    qNum = hkl.shape[0]
    SF = np.zeros(shape=(qNum))
    for index in range(atomsFrac.shape[0]):
        SF = (SF +
                fAll[:, index] * np.exp(2j * np.pi * np.sum(hkl*atomsFrac[index,:3],1))) 
    
    # I think all these are sorted as 
    dataSF = {
                'hkl' : hkl,
                'qVec' : qVec,
                'qVec1' : qVec1,
                'SF' : SF
            }

    return dataSF








def get_qz_dataQZ_dataPot_new(example: pandas.Series,
                        cbeds: numpy.array, 
                        pots: numpy.array, 
                        thicknesses: list,
                        qx: numpy.ndarray,
                        qy: numpy.ndarray,
                        keV: float = 300,
                        qz_sigma: float =1/20,
                        qRadiusSF: float = 0.05,
                        kirkland_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath] = './kirkland.txt'):
    """
    Placeholder
    Ewald Sphere, analytical QZ
    TRANSLATED FROM COLIN'S MATLAB SCRIPT
    """
    # inital coordinates 
    N = [*cbeds.shape]
    # window size 
    wx = signal.windows.tukey(2*N[0]).reshape(2*N[0], 1, 1)
    wy = signal.windows.tukey(2*N[1]).reshape(1, 2*N[1], 1)
    # not 100% at the momeent 
    scale = (1/4) / np.prod(N[:2])

    # create empty arrays 
    data_pot = np.zeros_like(cbeds)
    data_qz = np.zeros_like(cbeds)
    # antialising cordiates 
    xAA = np.array([*[i for i in range(0, cbeds.shape[0]//2)], 
           *[j + cbeds.shape[0]*2 for j in range(-cbeds.shape[0]//2, 0)]])
    yAA = np.array([*[i for i in range(0, cbeds.shape[1]//2)], 
           *[j + cbeds.shape[1]*2 for j in range(-cbeds.shape[1]//2, 0)]])


    # initalize ewald sphere origin  
    eV = example.E0 * 1000
    kZ = 1 / wavev(eV)
    qxyzEwald = np.array([0, 0, kZ])

    qMax = np.max([np.max(np.abs(qx)), np.max(np.abs(qy))])
    # load the kirkland paramerters 
    params = load_kirkland_parameters(kirkland_path)
    
    # creat fractionalised unit cell
    atomsFrac = create_atomsFrac(example.super_cell, example.cell_size)
    # create 3x3 matrix of the lattice vectors 
    latVec = example.cell_size * np.eye(3)
    # calculate the structure factors returns a dict 
    dataSF = calcSF(atomsFrac, latVec, qMax, params)

    # SF peaks distance from Ewald sphere
    distEwald = np.sqrt(np.sum((dataSF['qVec'] - qxyzEwald)**2,1)) - kZ
    weightEwald = np.exp(-distEwald**2/(2*qz_sigma**2))

    # write peaks in order of further to closest to Ewald sphere
    # not sure if this should be a sorted array or just return the indicies 
    indsOrder = distEwald[::-1].sort() # sorted array
    # indsOrder = np.argsort(distEwald) # indicies of the sorted array needs

    # local coordinates
    dqx = qx[1] - qx[0] # get step size in x 
    dqy = qy[1] - qy[0] # get step size in y
    max_xvec = np.ceil(qRadiusSF/dqx)
    max_yvec = np.ceil(qRadiusSF/dqy)
    xVec = np.arange(-max_xvec, max_xvec + 1)[:, np.newaxis]
    yVec = np.arange(-max_yvec, max_yvec + 1)[np.newaxis, :]

    # 2D circular disk
    disk = xVec ** 2 + yVec ** 2 <= ((xVec.shape[0] - 1) / 2 + 0.5) ** 2

    # init some arrays 
    analyticVg = np.zeros_like(cbeds)
    analyticQz = np.zeros_like(cbeds)
    analyticPhase = np.zeros_like(cbeds)


    #### STOPPED HERE FOR NOW 
    cell_normalisation = np.arra    
    atomic_frac = example.super_cell / cell_normalisation
    example.super_cell
    



    # get kirkland parameters

    calcSF()


    # loop over all the cbed images 
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
    