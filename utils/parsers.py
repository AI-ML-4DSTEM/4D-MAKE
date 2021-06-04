import argparse
import json
from typing import Union
import pathlib
import types
from types import SimpleNamespace

def parse_prismatic_simulations_args() ->  argparse.Namespace:
    """
    parse the arguments for the pyPrismatic simulations

    TODO 
    - add constraint that it requires at least 1 of n argumnets are passed.  
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--foo',action=.....)
        group.add_argument('--bar',action=.....)
    - add dask switch 
    - add skip steps if the simulation dataframe exits
    - add check if simulation exists  
    Returns:
    """


    parser = argparse.ArgumentParser()

    # parse the arguments
    # required 
    parser.add_argument("-h5", "--h5_save_path", type=str, required=True,
                        help="path to where h5 simulation files will be saved")
    parser.add_argument("-df_save", "--simulation_dataframe_save_path", type=str, required=True,
                        help="path to where the simulation dataframe will be saved")
    # optional 
    # pass the rotation master dataframe 
    parser.add_argument("-rot_df", "--master_rotation_df", type=str, 
                        help="path to the rotation master dataframe")
    # add the master simulation df path df
    parser.add_argument("-master_sim", "--master_simulation_df", type=str,
                        help="path to the master simulation dataframe" )
    # pick the type of dataframe to load 
    parser.add_argument("-df_type", "--dataframe_type", choices=['pandas', 'dask'], default='pandas', type=str,
                        help="pick the type of dataframe to be loaded, deafults to pandas")
    # pick the dataframe format
    parser.add_argument("-df_ex", "--df_extension", choices=['csv', 'parq', 'pkl'], default='pkl', type=str,
                        help="set the extension type to pick the reader, defualts to pickle")
    # path to load a json configuration file
    parser.add_argument("-json", "--json_path", type=str, default=None, help="path to the json configuration file")
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
    # a flag to run a slightly larger size 4 crystals
    parser.add_argument("--smoke_test", action="store_true", help="reduces the size of the intermediate dataframe to 4 rows")
    # a flag to run a slightly larger size 20 crystals
    parser.add_argument("--fire_test", action="store_true", help="reduces the size of the intermediate dataframe to 20 rows")
    
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

def dict_to_sns(d:dict) -> types.SimpleNamespace:
    """
    creates a SimpleNamespace object from a dictonary. Nested dictonaries not handled brillaintly

    sns.foo['bar']

    Args:
        d (dict): dictonary which can be nested to any degree 

    Returns:
        types.SimpleNamespace: [description]
    """
    return SimpleNamespace(**d)


def create_namespace_from_dict(d:dict) -> types.SimpleNamespace:
    """
    Creates a SimpleNamespace object from a dictionary, handles nested dictonaires nicely. 
    objects will be accessbile as sns.foo.bar 

    Args:
        dictionary (dict): [description]

    Returns:
        types.SimpleNamespace: [description]
    """

    # convert the dictionary to a string 
    json_object = json.dumps(d)

    # load the string with json parser 
    sns = json.loads(json_object, object_hook=dict_to_sns)

    return sns
