# Import dependencies
from training_utils import *
from pathlib import Path
import pandas
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import scipy.signal as signal
import sys
import argparse
from dask.distributed import Client
from dask_mpi import initialize 
import dask.dataframe as dd
from mpi4py import MPI



def __main__():
    #pass command line args through argparse 
    parser = argparse.ArgumentParser()

    # command line arguments
    parser.add_argument("-pkl", "--pkl_path", type=str, help="Path to simulation dataframe pickle files.") # change to parquet files
    parser.add_argument("-data", "--data_path", type=str, help="Path to the simualtion h5 files.")
    parser.add_argument("-train", "--training_path", type=str, help="Path to where trainingfiles will be written.")

    args = parser.parse_args()
    
    # Simulation Dataframe paths 
    #pkl_path = Path('/home/arakowski/Documents/ML-AI-4DSTEM/meta_pkls') # should probably change to parquets. 
    pkl_path = Path(args.pkl_path)

    # Path with to simulation files
    #data_path = Path('/data1/4000_zone_rot_crystals')
    data_path = Path(args.data_path)

    # path to where training files written
    #train_path = Path('/home/arakowski/Documents/ML-AI-4DSTEM/data/4000_zone_rot_crystals_training')
    train_path = Path(args.training_path)

    #glob the dataframes and files
    #! these should be generalised to multiple folders 

    #dataframes 
    pkls = sorted(pkl_path.glob('sim*.pkl'))
    # simulation files
    files = sorted(data_path.glob('./**/*.h5'))

    #print(len(files))
    #Create the dataframe
    df = pd.concat([pd.read_pickle(i) for i in pkls]) #this will work for a single master df

    #Augment the dataframe with the missing columns
    df = augment_dataframe(df)
    #print(df.head(1))

    #initialize the dask MPI link 
    initialize(nthreads=2)
    # connect the local process to the remote worksers 
    client = Client()

    # need to find a  good way to pick the number of partitions get from MPI no process or aim for size of around 100 MB . 
    # maybe I should be using dask delayed or similar? 
    num_parts =  MPI.COMM_WORLD.Get_size() # this might be stupid. 
    
    #convert to a dask dataframe
    ddf = dd.from_pandas(df,  npartitions=num_parts) 

    #run the data creatation process
    res = ddf.map_partitions(lambda df: df.apply(convert_series_to_training_data, axis=1,
         training_path=train_path,
         data_path=data_path)).compute(scheduler=client)


if __name__ == "__main__":
    __main__()


"""

    for i in range(len(df)):
        print(i)
        # get a single entry
        example = df.iloc[i]
        
        #get simulation name
        #data_path = Path('/data1/4000_zone_rot_crystals')
        input_name = data_path / example.simulation_name
        
        if os.path.exists(input_name):
            # add check exists here. 

            #training_path = Path('/home/arakowski/Documents/ML-AI-4DSTEM/data/4000_zone_rot_crystals_training')
            save_name = train_path /  example.training_name
            
            
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
            save_training_data(thicknesses=thicknesses, qx=qx, qy=qy, qz=qz, dataProbe=probe, dataMeas=cbeds, dataPots=data_pot, dataQz=data_qz, output_filename=save_name)
        else:
            print(f'example {i} does not exist')
"""

