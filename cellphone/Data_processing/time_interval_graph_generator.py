# This script is created by A. Biricz, 04.01.2021.

## This script uses the prepared short time interval
# daily data and generates undirected graphs 
# in shapes that can be fed into graph DL

## Arguments:
# --source_folder: where processed data of daily events located
# --source_pol_folder: where processed polygon files located
# --target_folder: save folder

# Example for running:
# python3 .py --source_folder '/media/Data_storage/Mobilcell/DayEventData/' --source_pol_folder '/media/Data_storage/Mobilcell/DayPolygonData/' --target_folder '/media/Data_storage/Mobilcell/TimeIntervalGraphs/'

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--start_idx", type=int, default=0, help="start with day 0")
parser.add_argument("--end_idx", type=int, default=365, help="end with day 365")
parser.add_argument("--source_folder", default='/mnt/DayEventData/' )
parser.add_argument("--source_pol_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/TimeIntervalGraphs/' )
parser.add_argument("--interval", type=int, default=60, help="time interval") ## CHANGED TO 60

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' #'/media/Data_storage/Mobilcell/DayEventData/'
source_pol = os.path.abspath( args.source_pol_folder ) + '/' #'/media/Data_storage/Mobilcell/DayPolygonData/'
target = os.path.abspath( args.target_folder ) + '/' #'/media/Data_storage/Mobilcell/TimeIntervalGraphs/'

files = np.array( sorted([ i for i in os.listdir(source) if 'daily-events-data_max-trip-min-'+str(args.interval) in i]) )
print(files.shape)

tower_info = pd.read_csv( source_pol+'fixed_merged_tower_locations_ovr-10_d-40.csv' ) ## CHANGED
sort_idx = np.argsort( tower_info.tower_id.values )
tower_info = tower_info.iloc[ sort_idx ]
tower_info.reset_index(inplace=True)

coords = np.unique( tower_info.iloc[:,2:], axis=0 )[:,1:]

time_splits = np.vstack( (np.arange(0,1470,int(args.interval))[:-1], np.arange(0,1470,int(args.interval))[1:]) ).T # CHANGED TO 60 minutes interval
#print('Time splits for a day:', time_splits)

def create_graph_and_save_to_disk( eq_interval, node_attrs_interval, tower_info, savename ):
    # get multiple movements as weights
    uq, counts = np.unique( np.concatenate( ( eq_interval[:,-2:], 
                                 np.ones(eq_interval.shape[0]).reshape(-1,1) ), 
                              axis=1 ).astype(np.uint16), 
              return_counts=True, axis=0 )
    
    # create edgelist
    V = np.concatenate( ( uq[:,:2], counts.reshape(-1,1) ), axis=1 )
    
    # create adjacency matrix and make graph undirected
    adj_mat = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.uint32 )
    adj_mat[ V[:,0], V[:,1] ] = V[:,2]
    adj_mat = adj_mat + adj_mat.T
    
    # get back the edgelist from the symmetric matrix
    idxs = np.argwhere( adj_mat )
    filt = idxs[:,0] < idxs[:,1]
    V_new = np.concatenate( (idxs[filt], adj_mat[ idxs[:,0][filt], idxs[:,1][filt] ].reshape(-1,1) ), axis=1 )
    
    # prepare to save edge and node attributes
    data_edge_index = V_new[:,:2].T # save info as the adjacency matrix, but in this compressed format
    data_edge_attr = V_new[:,-1].reshape(-1,1) # weights of movements as edge attributes
    data_x = np.concatenate( ( coords, node_attrs_interval.reshape(-1,1) ), axis=1 )
        
    # save to disk as compressed to avoid using large space
    np.savez_compressed( savename, data_edge_index=data_edge_index, 
                         data_edge_attr=data_edge_attr, data_x=data_x )

def process_a_day( source, target, filename, tower_info ):

    loaded = np.load( source+filename )
    
    ## individual trajectories with following attributes:
        # header = ['id', 'start_time_min', 'src_x', 'src_y', 'dst_x', 'dst_y', 
        #              'trip_time_min', 'dist_m', 'speed_ms', 'src', 'dst']
    eq_info = loaded['edge_data']
    
    ## resting event counts at each side:
    tower_datetime_events = loaded['node_data']

    ## current date
    date = filename.split('.npz')[0][-8:]

    for i in tqdm( range(time_splits.shape[0]) ): # go along the chosen time resolution (60 min step) ## CHANGED!

        # get current time intervals edge and node attributes
        eq_interval = eq_info[ np.in1d( eq_info[:,1],
                                    np.arange( time_splits[i,0], time_splits[i,1] ) ) ]
        node_attrs_interval = np.sum( tower_datetime_events[:,time_splits[i,0]:time_splits[i,1]], axis=1 )

        # path and filename of processed graph
        savename = target+'graph_'+date+'_'+str(i)+'_'+\
                'minutes-'+str(time_splits[i,0])+'-'+str(time_splits[i,1])
        
        # function call 
        if not os.path.exists(savename):
            create_graph_and_save_to_disk( eq_interval, node_attrs_interval, tower_info, savename )

# call this function like this:
# process_a_day( source, target, files[100], tower_info )
for q in range( args.start_idx, args.end_idx ):
    print('Current day:', q)
    process_a_day( source, target, files[q], tower_info )