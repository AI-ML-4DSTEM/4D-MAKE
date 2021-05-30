import argparse
import json
from typing import Union
import pathlib
import types
from types import SimpleNamespace

def parse_prismatic_simulations_args() ->  argparse.Namespace:
    """
    parse the arguments for the pyPrismatic simulations
    """
    ### add 
    # dask switch 
    # skip steps if the simulation dataframe exits
    # add check if simulation exists  

    parser = argparse.ArgumentParser()

    # parse the arguments
    # required 
    parser.add_argument("-rot_df", "--rotation_master_df", type=str, required=True, 
                        help="path to the rotation master dataframe")
    parser.add_argument("-h5", "--h5_save_path", type=str, required=True,
                        help="path to where h5 simulation files will be saved")
    parser.add_argument("-pkl", "--simulation_dataframe_save_path", type=str, required=True,
                        help="path to where the simulation dataframe will be saved")
    # optional 
    # pick the type of dataframe to load 
    parser.add_argument("-df_type", "--dataframe_type", choices=['pandas', 'dask'], default='pandas', type=str,
                        help="pick the type of dataframe to be loaded, deafults to pandas")
    # pick the dataframe format
    parser.add_argument("-df_ex", "--df_extension", choices=['csv', 'parq', 'pkl'], default='pkl', type=str,
                        help="set the extension type to pick the reader, defualts to pickle")
    # add the master simulation path 
    parser.add_argument("-master_sim", "--master_simulation_path", )

    # simulation paramters 
    parser.add_argument("-gpu", "--num_gpus", default=1, type=int,
                        help="passes the number of gpus to be used for the simulations")
    parser.add_argument("-semi", "--semi_angles", nargs='+', default=None, type=int,
                        help="returns list of semiangles, defaults to None, which falls back to hardcoded list")
    parser.add_argument("-e0", "--beam_energy", nargs='+', default=None, type=int,
                        help="returns list of beam energies, deafults to None, which falls back to hardcoded list")
    parser.add_argument("-fp", "--num_frozen_phonons", nargs='+', default=None, type=int,
                        help="returns list of number of frozen phonons, defaults to none, which falls back to hardcoded list")
    # I thought this was an option but maybe not
    # parser.add_argument("--store_all_phonons", action='store_true',
    #                     help="switch to store each phonon sperately, defaults false, which stores average structure")
    parser.add_argument("-st", "--slice_thickness", default=2, type=float, 
                        help="set slice thickness")
    parser.add_argument("-pot", "--potential_bound", default=2, type=float, 
                        help="set potential bound")
    parser.add_argument("-defocus", "--probe_defocus", nargs='+', default=None, type=float,
                        help="set probe defocus, defaults to None, which falls back to hardcoded list")
    
    args = parser.parse_args()

    return args


def parse_json_file(json_path:Union[str, pathlib.PosixPath, pathlib.WindowsPath]) -> types.SimpleNamespace:
    """
    Parse the json file and return an args namespace similar to argpase
    This is the way I'm going to allow passing a lot of variable arguments to the functions
    Args:
        json_path (Union[str, pathlib.PosixPath, pathlib.WindowsPath]): path to the json file.

    Returns:
        types.SimpleNamespace: Returns simple namespace full of args 
    """

    with open(json_path) as file:
        args = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    
    return args