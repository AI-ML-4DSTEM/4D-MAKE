
import qgrid
import pandas as pd
import matplotlib.pyplot as plt
import pandas 
from typing import Union
import pathlib
from pathlib import Path
import h5py
import numpy as np
import numpy



def change_path_root(row:pandas.Series, 
        base_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath]) -> Union[str, pathlib.PosixPath, pathlib.WindowsPath]:
    """[summary]

    Args:
        row (pandas.Series): [description]
        base_path (Union[str, pathlib.PosixPath, pathlib.WindowsPath]): [description]

    Returns:
        Union[str, pathlib.PosixPath, pathlib.WindowsPath]: [description]
    """
    tmp_row = row.copy(deep=True)
    base_path = Path(base_path)
    filename = Path(tmp_row.filenameOutput)
    filename = base_path / filename.parent.name / filename.name
    return filename


def get_pots_cbeds_and_probe(file_path):
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


def cbed_plotter(row:pandas.Series):
    
    file_path = row.filenameOutput
    pots, cbeds, probe = get_pots_cbeds_and_probe(file_path)
    
    # should do something clever to figure out the rows and columns
    shape = cbeds.shape[-1]
    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(10,10))
    axes = axes.ravel()
    for index, ax in enumerate(axes):
        ax.imshow(cbeds[...,index] ** 0.25)
        ax.axis('off')
        ax.set_title(f'{(index + 1) * 25} nm')
    plt.show()
    
    
def pots_plotter(row:pandas.Series, slice_thickness:int = 10):
    file_path = row.filenameOutput
    pots, cbeds, probe = get_pots_cbeds_and_probe(file_path)
    
    # should do something clever to figure out the rows and columns
    shape = pots.shape[-1]
    
    fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(10,10))
    axes = axes.ravel()
    for index, ax in enumerate(axes):
        ax.imshow(pots[:,:, index*slice_thickness:(index+1)*slice_thickness].sum(axis=2))
        ax.axis('off')
        ax.set_title(f'{index*slice_thickness}-{(index+1)*slice_thickness} nm')
    plt.show()
    
def probe_plotter(row:pandas.Series):
    file_path = row.filenameOutput
    pots, cbeds, probe = get_pots_cbeds_and_probe(file_path)
    
    fig, axes = plt.subplots(ncols=1, figsize=(3,3))
    
    axes.imshow(probe**0.25)
    axes.set_title('Amplitude')
    axes.axis('off')
    plt.show()