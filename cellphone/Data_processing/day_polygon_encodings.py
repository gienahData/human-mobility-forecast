# This script is created by A. Biricz, last modified 03.01.2021.

## This script generates daily polygon encodings
# from the raw data

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

for i in tqdm( range(files_poligons.shape[0]) ):
    # load file
    poligons_path = source + files_poligons[i]
    poligons_df = pd.read_csv( poligons_path, delimiter=';' )
    
    # drop poligons outside of the country
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovx.values < 366660, 
                                       poligons_df.eovx.values > 48210 ) ]
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovy.values < 934219, 
                                       poligons_df.eovy.values > 426341 ) ]
    
    # calculate raster encodings
    poligons_df['eovx_num'] = ( (poligons_df.eovx - start_x) / 127 ).astype(int)
    poligons_df['eovy_num'] = ( (poligons_df.eovy - start_y) / 127 ).astype(int)
    poligons_df['eov_idx'] = poligons_df.eovx_num * (num_y+1) + poligons_df.eovy_num
    
    # save encoded data
    savename = target+'polygon_encoded_'+files_poligons[i].split('_')[-1][:8]
    save_df = poligons_df[ ["network_identifier", "eov_idx"] ]
    np.save( savename, save_df.values )
    
    # saving csv is very slow!!
    #save_df.to_csv( savename, index=False, )