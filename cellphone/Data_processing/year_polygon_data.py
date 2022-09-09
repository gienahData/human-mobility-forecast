# This script is created by A. Biricz, last modified 17.03.2021.

## This script calculates unique tower ids

## Arguments:
# --source_folder: where events and polygons data located
# --target_folder: save folder

# Example for running:
# python3 .py --source_folder '/media/Data_storage/Mobilcell/Data/' --target_folder '/media/Data_storage/Mobilcell/DayPolygonData/'

import numpy as np
import pandas as pd
from itertools import product
import os
from tqdm import tqdm

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--source_folder", default='/mnt2/data/csv/' )
parser.add_argument("--target_folder", default='/mnt/DayPolygonData/' )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/'
target = os.path.abspath( args.target_folder ) + '/'

files_poligons = np.array( sorted([ i for i in os.listdir(source) if 'POLIGONS' in i]) )

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

# Investigate network id and its rasters for more days
towers_types = []
for i in tqdm( range(files_poligons.shape[0]) ):
    # load data
    poligons_path = source + files_poligons[i]
    poligons_df = pd.read_csv( poligons_path, delimiter=';' )
    
    # drop poligons outside of the country (was missing, caused index error!)
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovx.values < 366660, 
                                       poligons_df.eovx.values > 48210 ) ]
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovy.values < 934219, 
                                       poligons_df.eovy.values > 426341 ) ]

    # sort with network identifier
    poligons_df.sort_values( 'network_identifier', inplace=True )
    poligons_df.reset_index( inplace=True )

    pol_net_id = poligons_df.network_identifier.values
    pol_net_id_diff_idx = np.where( np.diff( pol_net_id ) )[0]+1
    pol_net_id_diff_idx = np.insert(pol_net_id_diff_idx, 0, 0, axis=0)

    u = poligons_df.network_identifier[ pol_net_id_diff_idx ].values, 
    t = poligons_df.network_element_type[ pol_net_id_diff_idx ].values
    
    towers_types.append( np.vstack( (u, t) ).T )
    
towers_types_numpy = np.concatenate(towers_types)
sort_idx = np.argsort( towers_types_numpy[:,0] )
towers_types_sorted = towers_types_numpy[sort_idx]

towers_types_diff_idx = np.where( np.diff( towers_types_sorted[:,0] ) )[0]+1
# insert first element (zero) ## otherwise left out!
towers_types_idx = np.insert( towers_types_diff_idx, 0, 0, axis=0 )
# insert last element (size of array) ## otherwise left out!
towers_types_idx = np.append( towers_types_idx, towers_types_sorted.shape[0] )
towers_types_idx = np.vstack( (towers_types_idx[:-1], towers_types_idx[1:]) ).T

type_to_num = dict( zip( ['A', 'B', 'C', 'D', 'E', 'U'], range(1,7) ) )

# save unique tower ids with its type (2G, 3G, ...)
towers_types_final = np.zeros( (towers_types_idx.shape[0], len(type_to_num)+1 ), dtype=np.int32 )
for n in range( towers_types_idx.shape[0] ):
    current = towers_types_sorted[ towers_types_idx[n,0]:towers_types_idx[n,1] ] 
    u, c = np.unique( current[:,1], return_counts=True )
    towers_types_final[n, 0] = current[0,0]
    for q in range(u.shape[0]):
        towers_types_final[n, type_to_num[ u[q] ] ] = c[q]

np.save( target+"unique-tower-id_all", towers_types_final )