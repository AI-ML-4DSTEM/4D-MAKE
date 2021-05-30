from numpy.lib.financial import _nper_dispatcher
from numpy.lib.function_base import _parse_input_dimensions
from pandas.core import algorithms
from pandas.core.frame import DataFrame
import pyprismatic
import pyprismatic as pr
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import hashlib
import h5py
import argparse
import time
import itertools as it
import pandas as pd
import pandas
import numpy as np
import numpy 
import pathlib
import dask.dataframe as dd
from typing import Union
from uuid import uuid4
from pathlib import Path
import os 
import pickle
from types import SimpleNamespace
from utils import dataframe_utils as df_utils
from utils import simulation_utils as sim_utils
from utils import parsers
from utils import general_utils as gen_utils






####### DEFINE MAIN FUNCTION HERE #######
#def __main__():

# Parse the command line arguments
args = parsers.parse_prismatic_simulations_args()


# if json_path is passed then read from json file and return args
if args.json_path != None:
    # parse the json file to get args
    args2 = parsers.parse_json_file(args.json_path)
    # create dictionary with all the args from json and command Line 
    # JSON IS RETAINED OVER COMMAND LINE 
    args_dict = {**vars(args), **vars(args2)}
    # convert the dictionary back to a SimpleNamespace
    args = SimpleNamespace(**args_dict)
else:
    # just keep the command line parse arguments
    pass

# set the paths and esure they are pathlib objects
rot_master_df_path = Path(args.rotation_master_df)
h5_save_path = Path(args.h5_save_path)
simulation_dataframe_save_path = Path(args.simulation_dataframe_save_path)

# set the probe semi angles to be examined (mrad)
if args.semi_angles != None:
    semi_angles = args.semi_angles
else:
    semi_angles = [1, 2, 4]
# set the beam 
if args.beam_energy != None:
    E0s = args.beam_energy
else:
    E0s = [300]
# set the number of frozen phonon configurations
if args.num_frozen_phonons != None:
    num_fps = args.num_frozen_phonons
else:
    num_fps = [1]
# set the probe defocus 
if args.probe_defocus != None:
    probe_defocuses = args.probe_defocus
else:
    probe_defocuses = [0]
# set the slice thickness
slice_thickness = [args.slice_thickness]
# set the potential_bound
potential_bound = [args.potential_bound]

# do I want a non dask option as well? 

# if the number of gpus is zero skip dask cuda 
if args.num_gpus == 0: 
    # I need to set it as CPU only compute 
    pass

elif args.num_gpus >= 1:
    from dask_cuda import initialize 
    protocol = "ucx"
    enable_tcp_over_ucx = True
    enable_nvlink = True
    enable_infiniband = False

    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    cluster = LocalCUDACluster(local_directory="/tmp/USERNAME",
                            protocol=protocol,
                            enable_tcp_over_ucx=enable_tcp_over_ucx,
                            enable_infiniband=enable_infiniband,
                            enable_nvlink=enable_nvlink,
                            threads_per_worker=6,
                            #rmm_pool_size="35GB"
                        )
    client = Client(cluster)


if args.master_simulation_df != None:
    # load the dataframe with all the rotation names and structure paths
    master_rotation_df = df_utils.master_rotation_df_loader(
                    master_df_path=rot_master_df_path,
                    df_type=args.dataframe_type,
                    df_extension=args.df_extension
                    )


    # ensure its a dask_dataframe
    # do I want to add a dask switch here 
    if type(master_rotation_df) == pandas.core.frame.DataFrame:
        master_rotation_df = dd.from_pandas(master_rotation_df, npartions=)
    else:
        pass


    df_utils.check_master_dataframe_for_duplicates()


    res = master_rotation_df.apply(lambda row: 
                df_utils.create_simulation_dataframe_from_series(
                    row=row, 
                    semi_angles=semi_angles,
                    num_fps=num_fps,
                    E0s=E0s,
                    probe_defocuses=probe_defocuses, 
                    slice_thickness=slice_thickness,
                    potential_bound=potential_bound, 
                    h5_save_path='.',
                    dataframe_save_path='.'), axis=1, meta=master_rotation_df)

    res.compute(scheduler=client)

else:
    master_simulation_dataframe = df_utils.create_master_simulation_dataframe()





##### QUICK DASK COMMANDS ######
res = dask_master_df.apply(lambda row: create_simulation_dataframe_from_series(row=row, semi_angles=[1,2,3],num_fps=[1],E0s=[100,200,300],probe_defocuses=[0], slice_thickness=[2],potential_bound=[2], h5_save_path='.', dataframe_save_path='.'), axis=1, meta=dask_master_df)

res.compute(scheduler=client)

# applying the simulate function to it all 
ddf = dd.from_pandas(sim_df, npartitions=9) 
# this raises an error but it runs. I need to get a better handle on meta_data.
res = ddf.apply(simulate_row, gpu=0, axis=1, meta=ddf)   