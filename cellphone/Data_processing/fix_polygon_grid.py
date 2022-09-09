# This script is created by A. Biricz, last modified 03.01.2021.

## This script uses daily polygon encodings to 
# generate a fix encoding for the whole year

## Arguments:
# --source_folder: where processed polygons data located
# --target_folder: save folder (same as source by default)

# Example for running:
# python3 .py --source_folder '/media/Data_storage/Mobilcell/DayPolygonData/' --target_folder '/media/Data_storage/Mobilcell/DayPolygonData/'

import numpy as np
import pandas as pd
from itertools import product
import os
from tqdm import tqdm

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--source_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/DayPolygonData/' )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/'
target = os.path.abspath( args.target_folder ) + '/'

# already encoded daily data of polygons
files_pol_enc = np.array( sorted( [ i for i in 
                            os.listdir( source ) if 'encoded' in i ] ) )

# all unique ids of polygons for the whole year
tower_id_all_global = np.load( source+'unique-tower-id_all.npy' )[:,0] # 0th column contains the ids
tower_to_int_all = dict( zip( tower_id_all_global, np.arange( tower_id_all_global.shape[0] ) ) )

# This seems to be the global grid of rasters, same for every day!
start_x = 48262
end_x = 362968
start_y = 426468
end_y = 934214
num_x = int( ( end_x - start_x ) / 127 )
num_y = int( ( end_y - start_y ) / 127 )

# this raster encoding is universal then
raster_x = np.arange(start_x, end_x+127, 127, dtype=np.int32)
raster_y = np.arange(start_y, end_y+127, 127, dtype=np.int32)

# get coordinate vector
raster_coords = np.array( list(product( raster_x, raster_y )) )

# function that does the calculations for all days
def calc_tower_coords_day( source, files_pol_enc ):
    pol_enc = np.load( source + files_pol_enc )

    sort_idx = np.argsort( pol_enc[:,0] )
    pol_enc_tower = pol_enc[ sort_idx ]

    pol_enc_diff_idx = np.where( np.diff( pol_enc_tower[:,0] ) )[0]+1
    # insert first element (zero) ## otherwise left out!
    pol_enc_idx = np.insert(pol_enc_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    pol_enc_idx = np.append( pol_enc_idx, pol_enc.shape[0] )

    all_idx_tower = np.vstack( ( pol_enc_idx[:-1], pol_enc_idx[1:] ) ).T

    tower_id_all = []
    tower_coords_all = []
    for j in all_idx_tower:
        tower_id_all.append( pol_enc_tower[:,0][ j[0] ] )
        tower_coords_all.append( np.mean( raster_coords[ pol_enc_tower[:,1][ j[0]:j[1] ] ], axis=0 ) )
    tower_id_all = np.array( tower_id_all )
    tower_coords_all = np.array( tower_coords_all )

    return np.concatenate( (np.array([ tower_to_int_all[ k ] for k in tower_id_all ]).reshape(-1,1), 
                     tower_coords_all.astype(int) ), axis=1 )

# calling the function for the whole year
tower_coords_year = []
for s in tqdm( range( files_pol_enc.shape[0] ) ):
    tower_coords_year.append( calc_tower_coords_day( source, files_pol_enc[s] ) )

print("Collecting, sorting results..")
# collecting, sorting results
all_tower_coords_year = np.concatenate( tower_coords_year )
#all_tower_coords_year = np.sort( all_tower_coords_year, axis=0 )
sorting = np.argsort( all_tower_coords_year[:,0] ) # SORT ONLY w.r.t. FIRST COLUMN!
all_tower_coords_year = all_tower_coords_year[sorting]

# calculating fix coordinates (and do some analysis) for the towers for the whole year
all_tower_coords_year_diff_idx = np.where( np.diff( all_tower_coords_year[:,0] ) )[0]+1
all_tower_coords_year_diff_idx = np.insert(all_tower_coords_year_diff_idx, 0, 0, axis=0)
all_tower_coords_year_diff_idx = np.append( all_tower_coords_year_diff_idx, all_tower_coords_year.shape[0] )

all_tower_coords_year_idx = np.vstack( ( all_tower_coords_year_diff_idx[:-1], all_tower_coords_year_diff_idx[1:] ) ).T

print("Calculating statistics..")
tower_data_year = []
for l, j in enumerate( all_tower_coords_year_idx ):
    mean_val = np.mean( all_tower_coords_year[ j[0]:j[1], 1: ], axis=0 )
    std_val = np.std( all_tower_coords_year[ j[0]:j[1], 1: ], axis=0 )
    perc_val_10 = np.percentile( all_tower_coords_year[ j[0]:j[1], 1: ], 10, axis=0 )
    perc_val_50 = np.percentile( all_tower_coords_year[ j[0]:j[1], 1: ], 50, axis=0 )
    perc_val_90 = np.percentile( all_tower_coords_year[ j[0]:j[1], 1: ], 90, axis=0 )
    perc_val_99 = np.percentile( all_tower_coords_year[ j[0]:j[1], 1: ], 99, axis=0 )
    tower_data_year.append( np.concatenate( ( [ all_tower_coords_year[ j[0], 0 ] ], 
                                              mean_val, std_val, 
                                              perc_val_10, perc_val_50, 
                                              perc_val_90, perc_val_99, 
                                            ) 
                                          ).astype(np.int64) )
tower_data_year = np.array( tower_data_year )

print("Saving to file..")
# prepare data to be saved and used later on to build a fixed graph
int_to_towers = { v: k for k, v in tower_to_int_all.items() }

pd.DataFrame( np.concatenate( ( np.array([ int_to_towers[i] for 
                                           i in tower_data_year[:,0] ]).reshape(-1,1),
              tower_data_year ), axis=1 ), 
              columns=[ ['original_id', 'tower_id', 'mean_x',
              'mean_y', 'std_x', 'std_y', 'perc_10_x', 'perc_10_y',
              'perc_50_x', 'perc_50_y', 'perc_90_x', 'perc_90_y',
              'perc_99_x', 'perc_99_y' ] ] ).to_csv( target+'fixed_tower_locations.csv', 
                                                       index=False )

