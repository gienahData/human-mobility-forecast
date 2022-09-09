# This script is created by A. Biricz, last modified 27.02.2021.

## This script uses daily polygon encodings to 
# calculate cell overlap matrices for the whole year

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

## function to do whole process:
def calculate_overlap_matrix( file_enc ):
    
    # load one daily encoding of the cells (cell id, encoding)
    pol_enc = np.load( source + file_enc )
    
    ## calculation: how many cells each raster has

    # sort with the raster encoding
    sort_idx = np.argsort( pol_enc[:,1] )
    pol_enc_raster = pol_enc[ sort_idx ]

    # calculation of boundary indices (to speed up processing)
    pol_enc_diff_idx = np.where( np.diff( pol_enc_raster[:,1] ) )[0]+1
    # insert first element (zero) ## otherwise left out!
    pol_enc_idx = np.insert(pol_enc_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    pol_enc_idx = np.append( pol_enc_idx, pol_enc.shape[0] )

    # generate index array
    all_idx_raster = np.vstack( ( pol_enc_idx[:-1], pol_enc_idx[1:] ) ).T

    # calculate rasters that belongs to a cell (tower)
    rasters_all = []
    towers_all = []
    for i in all_idx_raster:
        rasters_all.append( pol_enc_raster[:,1][ i[0] ] )
        towers_all.append( pol_enc_raster[:,0][ i[0]:i[1] ] )

    raster_to_tower = dict( zip( rasters_all, towers_all ) )

    ## calculation: area of cells 
    # sort with cell encoding
    sort_idx = np.argsort( pol_enc[:,0] )
    pol_enc_tower = pol_enc[ sort_idx ]

    # calculation of boundary indices (to speed up processing)
    pol_enc_diff_idx = np.where( np.diff( pol_enc_tower[:,0] ) )[0]+1
    # insert first element (zero) ## otherwise left out!
    pol_enc_idx = np.insert(pol_enc_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    pol_enc_idx = np.append( pol_enc_idx, pol_enc.shape[0] )

    all_idx_tower = np.vstack( ( pol_enc_idx[:-1], pol_enc_idx[1:] ) ).T

    # all unique ids of polygons for the whole year
    tower_id_all_global = np.load( source+'unique-tower-id_all.npy' )[:,0] # 0th column contains the ids
    tower_to_int_all = dict( zip( tower_id_all_global, np.arange( tower_id_all_global.shape[0] ) ) )

    ## calcultion: overlap matrix, how many rasters are shared between different cells
    # preparation
    tower_id_all = []
    overlaps_counts_all = []
    overlaps_all = []
    for j in all_idx_tower:
        tower_id_all.append( pol_enc_tower[:,0][ j[0] ] )
        ov_uniq, ov_counts = np.unique( 
                                        np.concatenate( [ raster_to_tower[k] 
                                        for k in pol_enc_tower[:,1][ j[0]:j[1] ] ] ),
                                    return_counts=True )
        overlaps_all.append( ov_uniq ) 
        overlaps_counts_all.append( ov_counts )
    tower_id_all = np.array( tower_id_all )
    overlaps_all = np.array( overlaps_all )

    # place calculated numbers into final position
    overlap_matrix = np.zeros( (len(tower_to_int_all), len(tower_to_int_all)), 
                                dtype=np.uint16 )
    for n in range( tower_id_all.shape[0] ):
        line = tower_to_int_all[ tower_id_all[n] ]
        idx_to_add_at = [ tower_to_int_all[ q ] for q in overlaps_all[n] ]
        np.add.at(overlap_matrix[line], idx_to_add_at, overlaps_counts_all[n] )

    # normalize with diagonal values (diagonal elements are raster numbers of cells)
    norm_factor = np.ones( overlap_matrix.shape[0], dtype=np.uint16 )
    nonzero = overlap_matrix[ np.diag_indices(overlap_matrix.shape[0]) ] > 0
    norm_factor[nonzero] = overlap_matrix[ np.diag_indices(overlap_matrix.shape[0]) ][nonzero]
    overlap_matrix_normed = (overlap_matrix / norm_factor).astype(np.float16)

    return overlap_matrix_normed # overlap_matrix

for d in tqdm( range( files_pol_enc.shape[0] ) ):
    ovr = calculate_overlap_matrix( files_pol_enc[d] )
    
    # saving overlap matrix
    current_date = files_pol_enc[d].split('_')[-1][:8]
    savename = target+'overlap-matrix_normed_'+current_date
    # must be saved as compressed to spare disk space (1/1000!)
    np.savez_compressed( savename, ovr=ovr )