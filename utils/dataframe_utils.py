import pathlib
from typing import Union
import pathlib
from pathlib import Path
import pandas
import pandas as pd
import dask.distributed as dd
import hashlib
import pickle
import os
import itertools as it
from uuid import uuid4
from . import general_utils as  gen_utils 
import numpy
import numpy as np


def master_df_loader(master_df_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                    df_type: str, df_extension: str):
    """
    Placeholder
    There are two types of dataframe dask and pandas
    dask will be a path to a directory
    pandas will be to a path to a file 

    """
    if df_type == 'pandas':
        if df_extension == 'pkl':
            master_df = pd.read_pickle(master_df_path)
            return master_df
        elif df_extension == 'parq':
            master_df = pd.read_parquet(master_df_path)
            return master_df
        elif df_extension == 'csv':
            master_df = pd.read_csv(master_df_path)
            return master_df
        else: 
            print("YOU DID SOMETHING I DIDN'T THINK POSSIBLE CONGRATS BUT IT WON'T WORK")
    elif df_type == 'dask':
        if df_type == 'parq':
            master_df = dd.read_parquet(master_df_path)
            return master_df
        elif df_type == 'csv':
            master_df = dd.read_csv(master_df_path)
            return master_df
        else:
            print("YOU DID SOMETHING I DIDN'T THINK POSSIBLE CONGRATS BUT IT WON'T WORK")
    else:
        print("YOU DID SOMETHING I DIDN'T THINK POSSIBLE CONGRATS BUT IT WON'T WORK") 



def check_master_dataframe_for_duplicates(master_df: pandas.DataFrame,
                                        hash_keys:list = ['uvw', 'angles', 'mat_id']
                                        ) -> pandas.DataFrame:
    """ 
    check if a master dataframe has any duplicates, drop if they do.
    """
    # create a hash of the pickle dump for each row
    master_df['row_hash'] = master_df.apply(lambda row: hashlib.md5(pickle.dumps(
                               row[hash_keys])).hexdigest(), axis=1)
    
    # drop the duplicates
    master_df = master_df.drop_duplicates(subset=['row_hash'])
    
    return master_df



### CREATING THE INDIVIDUAL DATAFRAMES

def create_simulation_dataframe_from_series(row:pandas.Series,
                                semi_angles:list,
                                num_fps:list,
                                E0s:list,
                                probe_defocuses:list,
                                slice_thickness:list,
                                potential_bound:list,
                                h5_save_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                                dataframe_save_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                                algorithm:str = 'multislice',
                                no_pixels:Union[tuple,list,numpy.ndarray] = np.array(512,512),
                                numSlices:int = 10
                                ):

    """
    Placeholder
    takes a row from the master rotation dataframe and returns a dataframe with all the 
    different simulation parameters, each row returns an individual dataframe. 
    """
       

    # ensure paths are pathlib objects
    h5_save_path = Path(h5_save_path)
    dataframe_save_path = Path(dataframe_save_path)

    # create the seed for simualtion 
    # needs to be done inside the loop or all crystals will have them all
    # row['seed'] = hash_string_to_int(row.mat_id)

    # use row.uuid to create a sub directory
    sub_dir_name = str(row.uuid)
    h5_save_path = h5_save_path / sub_dir_name
    dataframe_save_path = dataframe_save_path / sub_dir_name
    
    # check the directories exist and make if not
    if not os.path.exists(h5_save_path):
        os.makedirs(h5_save_path)
    else:
        pass

    if not os.path.exists(dataframe_save_path):
        os.makedirs(dataframe_save_path)
    else:
        pass
    
    # create a dictionary to save the in.
    simulations_params = {}
    #loop over all the combinations
    for index, value in enumerate(it.product(
                                semi_angles,
                                num_fps,
                                E0s,
                                probe_defocuses,
                                slice_thickness,
                                potential_bound)):
        # copy the row so that it doesn't over write it
        tmp_row = row.copy(deep=True) 
        # create a uuid for the file name
        key = str(uuid4())
        # create the output name
        file_out = h5_save_path / f'{key}_prismatic.h5'

        tmp_row['filenameOutput'] = file_out
        tmp_row['probeSemiangle'] = value[0]
        tmp_row['alphaBeamMax'] = value[0] * 2
        tmp_row['numFP'] = value[1]
        tmp_row['E0'] = value[2]
        tmp_row['probeDefocus'] = value[3]
        tmp_row['sliceThickness'] = value[4]
        tmp_row['potBound'] = value[5]
        tmp_row['no_pixels'] = no_pixels
        tmp_row['numSlices'] = numSlices
        tmp_row['realspacePixelSizeX'] = tmp_row.cell_size[0] / tmp_row.no_pixels[0]
        tmp_row['realspacePixelSizeY'] = tmp_row.cell_size[1] / tmp_row.no_pixels[1]
        tmp_row['simulation_seed'] = gen_utils.hash_string_to_int(str(tmp_row.filenameOutput))
        tmp_row['algorithm'] = algorithm
        tmp_row['simulation_program'] = 'pyprismatic'
        
        # add it to the dictionary
        simulations_params[key] = tmp_row    
    
    # create the dataframe from the index 
    df = pd.DataFrame.from_dict(simulations_params, 'index')

    #save it as a pickle file for now, need to add option to parquet
    df.to_pickle(dataframe_save_path / 'simulation_dataframe.pkl')
    
    # might not want to return them in the end for dask reasons. 
    return None
    #return simulations_params, df


##### CREATING THE MASTER SIMULATION DATAFRAME #####

def create_master_simulation_dataframe(dataframe_save_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
                                        save_exten:str = 'pkl',
                                        df_exten:str = 'pkl',
                                        save_df:bool = True,
                                        return_df:bool = True):
    """
    collect all the simulation dataframes into one giant dataframe
    """
    
    dataframe_save_path = Path(dataframe_save_path)
    dfs = sorted(dataframe_save_path.glob(f'./**/simulation_dataframe.{df_exten}'))
    dfs = [pd.read_pickle(i) for i in dfs]
    master_df = pd.concat(dfs, ignore_index=True)
    
    if save_df:
        master_df.to_pickle(dataframe_save_path / f'master_simulation.{save_exten}')
    else: pass

    if return_df: 
        return master_df
    else: 
        return None

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