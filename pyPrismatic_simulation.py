import pyprismatic
import pyprismatic as pr
from dask.distributed import Client
import pandas
import dask.dataframe as dd
from typing import Union
from pathlib import Path
from types import SimpleNamespace
from utils import dataframe_utils as df_utils
from utils import simulation_utils as sim_utils
from utils import parsers
from utils import general_utils as gen_utils






####### DEFINE MAIN FUNCTION HERE #######
def __main__():

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
    if args.master_rotation_df != None: 
        rot_master_df_path = Path(args.master_rotation_df) # no longer a required argument
    else:
        pass
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
        # should I add an option for non dask 

    elif args.num_gpus >= 1:
        # I need to add an option for GPU and pandas style gpu
        from dask_cuda import initialize, LocalCUDACluster  
        # These settings should be set using the json config
        # convert this to a function so I can pass the json args to it e.g. 
        # cuda_protocols(**args.dask_cuda_func.__todict__()), and have defaults set to these. 
        # these settings are system dependent so I need a way to have a sensible way to handle it
        # only over TCP is the safest and probably performant enough method. 
        protocol = "ucx"
        enable_tcp_over_ucx = True
        enable_nvlink = False # on local machine oif True I duplicate the crystals. 
        enable_infiniband = False

        initialize.initialize(
            create_cuda_context=True,
            enable_tcp_over_ucx=enable_tcp_over_ucx,
            enable_infiniband=enable_infiniband,
            enable_nvlink=enable_nvlink,
        )
        
        cluster = LocalCUDACluster(local_directory="/tmp/USERNAME", # this will throw and error on windows
                                protocol=protocol,
                                enable_tcp_over_ucx=enable_tcp_over_ucx,
                                enable_infiniband=enable_infiniband,
                                enable_nvlink=enable_nvlink,
                                threads_per_worker=6,
                                #rmm_pool_size="35GB" #  I think I need to change my docker base image to a later version of cuda 
                            )
        client = Client(cluster)

    # if the master simulation dataframe is passed we can use that 
    if args.master_simulation_df != None:
        # load the dataframe with all the rotation names and structure paths
        master_simulation_df = df_utils.master_df_loader(
                        master_df_path=args.master_simulation_df, # this looks wrong
                        df_type=args.dataframe_type,
                        df_extension=args.df_extension
                        )
        
        print(type(master_simulation_df))
        if args.smoke_test:
            master_simulation_df = master_simulation_df[:4]
            nparts = 4
        elif args.fire_test:
            master_simulation_df = master_simulation_df[:20]
            nparts = 4
        else:
            nparts = 40
    # if the simulation dataframe is not passed we need to create it 
    else:

        # load the rotation dataframe
        master_rotation_df = df_utils.master_df_loader(
                        master_df_path=args.master_rotation_df,
                        df_type=args.dataframe_type,
                        df_extension=args.df_extension)

        if args.smoke_test:
            master_rotation_df = master_rotation_df[:4]
            nparts = 4
        elif args.fire_test:
            master_rotation_df = master_rotation_df[:20]
            nparts = 4
        else:
            nparts = 40
        # ensure the rotation dataframe is a dask_dataframe
        # do I want to add a dask switch here 
        if type(master_rotation_df) == pandas.core.frame.DataFrame:
            master_rotation_df = dd.from_pandas(master_rotation_df, npartitions=nparts) # how to pick a sensible number here
        else:
            pass
        
        # for each row in the roation dataframe create the simualtion dataframe
        master_rotation_df['name'] = 'name'  # seeing if this fixes -- AttributeError: 'DataFrame' object has no attribute 'name'
        res = master_rotation_df.apply(lambda row: df_utils.create_simulation_dataframe_from_series(
                                        row=row, 
                                        E0s=E0s,
                                        semi_angles=semi_angles,
                                        probe_defocuses=probe_defocuses,
                                        num_fps=num_fps,
                                        slice_thickness=slice_thickness,
                                        potential_bound=potential_bound,
                                        h5_save_path=h5_save_path,
                                        dataframe_save_path=simulation_dataframe_save_path
                                        ), axis=1, meta=master_rotation_df)
        
        res.compute()
        
        # collect all the simulation dataframes together into a master dataframe
        master_simulation_df = df_utils.create_master_simulation_dataframe(
                                dataframe_save_path=simulation_dataframe_save_path)
        

        # ensure the simulation dataframe is a dask_dataframe
        # do I want to add a dask switch here 
        if type(master_simulation_df) == pandas.core.frame.DataFrame:
            master_simulation_df = dd.from_pandas(master_simulation_df, npartitions=4) # how to pick a sensible number here
        else:
            pass


        # # check the dataframe for duplicates
        # # issues with meta data, this add 'row_hash' str to the columnts
        # # I'm not 100% how to handle the meta data
        # # not sure on the functionality of this feature. 
        # master_simulation_df = master_simulation_df.map_partitions(lambda df:
        #                 df_utils.check_master_dataframe_for_duplicates(df),
        #                 meta=master_simulation_df).compute(scheduler=client)

    if type(master_simulation_df) == pandas.core.frame.DataFrame:
        master_simulation_df = dd.from_pandas(master_simulation_df, npartitions=4) # how to pick a sensible number here
    else:
        pass
    # Simulate the dataframe
    res = master_simulation_df.apply(lambda row: sim_utils.simulate_row(row), axis=1, meta=master_simulation_df)

    res.compute(scheduler=client)

    client.shutdown()

if __name__ == '__main__':
    __main__()

