# Script for Simulation 12000 CBED patters from 4000 crystals in batches of 100.

# Import dependencies
import pyprismatic as pr 
import h5py 
import numpy as np
from pathlib import Path
from uuid import uuid4
import itertools as it
import time
from datetime import datetime
import pandas as pd
import sys
import os
import argparse
from simulation_utils import *


def __main__():
    #pass command line args through argparse 
    parser = argparse.ArgumentParser()

    # command line arguments
    parser.add_argument("-pkl", "--pkl_path", type=str, help="Path to simulation dataframe pickle files.") # change to parquet files
    parser.add_argument("-data", "--data_path", type=str, help="Path to the simualtion h5 files.")
    parser.add_argument("-train", "--training_path", type=str, help="Path to where trainingfiles will be written.")
    parser.add_argument("-batch", "--crystals_batch_size", type=int, default=100, help="number of crystals to do at a time ")
    parser.add_argument("-id", "--slurm_job_id", type=int, help="SLURM generated task ID")
    args = parser.parse_args()



    #set_save_path 
    out_path = Path(args.save_path)

    # Load the list of files 
    targets_df = pd.read_pickle(args.pkl_path)


    # I want to simulate 100 crystals, I'll use the job array to submit the job 

    SLURM_JOB_ID = int(args.slurm_job_id) # pass the I.D rather than the getting it from os 
    #SLURM_JOB_ID = 1 # pretending for now
    CRYSTALS_BATCH_SIZE = int(args.crystals_batch_size)
    lower_bound = SLURM_JOB_ID * CRYSTALS_BATCH_SIZE
    upper_bound = lower_bound + CRYSTALS_BATCH_SIZE

    in_files = [file for file in targets_df.iloc[lower_bound:upper_bound].file_path.to_list()]

    assert len(in_files) == CRYSTALS_BATCH_SIZE

    # hardcoded for convenience currently need a better solution 
    # I should pass these in via a file or create the dataframe first and then simulate
    semi_angles = [1, 2, 4]
    #max_angles *= semi_angles
    num_fps = [1]
    #realspace_pixel_size = [0.1, 0.05]
    tiling = [1]
    E0s = [300]

    #create an empty dictionary for the smulation values
    simulation_params = {}

    for i, val in enumerate(it.product(semi_angles,
                                    num_fps, 
                                    tiling,
                                    E0s,
                                    in_files)):
        key = str(uuid4())
        file_out = f'{str(out_path)}/{key}_prism.h5'
        simulation_params[key] = { 'probeSemiangle':val[0],
                                    'alphaBeamMax': val[0]*2,
                                    'numFP':val[1],
                                    'tileX':val[2],
                                    'tileY':val[2],
                                    'E0':val[3],
                                    'filenameAtoms':val[4],
                                    'filenameOutput':file_out,
                                }


            
            
    # Run the Simulations 
    #create empty dictionaries for storing experiments details
    output_dict = {}
    timings_dict = {}
    for key in simulation_params:
        
        #get the dictonary parameters for the indiviudal 
        local_dict = simulation_params[key]

        # set default values
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

        #change default values to those in the simulation dict
        for field, val in local_dict.items():
            setattr(meta, field, val)

        #
        output_dict[key] = make_meta_dict(meta)
        print(meta.filenameAtoms, meta.filenameOutput)#, meta.E0)
        #start a timer
        start = time.time()
        #start the simulation
        meta.go(display_run_time=True, save_run_time=False)
        end = time.time()
        
        timings_dict[key] = {'simulation_runtime':end-start,
                            'unix_time': end}


        
    df = pd.DataFrame.from_dict(output_dict, 'index').convert_dtypes()
    df.to_pickle(out_path/f'simulation_output_info_dataframe_{SLURM_JOB_ID}.pkl')
    timings_df = pd.DataFrame.from_dict(timings_dict, 'index').convert_dtypes()
    timings_df.to_pickle(out_path/f'timings_dataframe_{SLURM_JOB_ID}.pkl')
