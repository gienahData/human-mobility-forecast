# This script is created by A. Biricz, 29.11.2020, updated 02.04.2021.


## UPDATE: 02.04.2021. Fixed, merged nodes!

## 29.11.2020. This script generates daily graphs on the daily basis not(!) on the fix grid

## Arguments:
# --source_folder: where events and polygons data located
# --source_pol_folder: where processed polygon files located
# --target_folder: save folder
# --start_idx: first file's index
# --end_idx: last file's index

# Example for running:
# python3 .py --source_folder '/media/Data_storage/Mobilcell/Data/'  --source_pol_folder '/media/Data_storage/Mobilcell/DayPolygonData/' --target_folder '/media/Data_storage/Mobilcell/DayGraphData/' --start_idx 0 --end_idx 1

import numpy as np
import pandas as pd
from itertools import product
import os
from tqdm import tqdm

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--start_idx", type=int, default=0, help="start with day 0")
parser.add_argument("--end_idx", type=int, default=365, help="end with day 365")
parser.add_argument("--source_folder", default='/mnt2/data/csv/' )
parser.add_argument("--source_pol_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/DayGraphData/' )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
source_pol = os.path.abspath( args.source_pol_folder ) + '/' #'/media/Data_storage/Mobilcell/DayPolygonData/'
target = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'

files_events = np.array( sorted([ i for i in os.listdir(source) if 'EVENTS' in i]) )
files_events_cleaned = np.array( sorted([ i for i in os.listdir(source) if 'Events' in i]) )

def calculate_day_graph( source, target, events_path ):
    
    # load data
    print("loading input data")
    events_df = pd.read_csv( source+events_path, delimiter=';' )
    
    # load global, fix tower data
    towers_df = pd.read_csv( source_pol+'fixed_merged_tower_locations.csv' )

    # calculate tower encodings
    tower_id = np.unique( towers_df.original_id.values )
    tower_to_int = dict( zip( towers_df.original_id.tolist(), towers_df.tower_id.tolist() ) )

    tower_coords_all = dict( zip( towers_df.values[:,1].tolist(), 
                                  towers_df.values[:,2:4].tolist() ) )

    # add encodings to the dataframes
    print("encode ids of events")
    events_df["tower_idx"] = [ tower_to_int[i] for i in events_df.network_identifier.values ]
    
    # calculate event (equipment id) encodings
    event_id = np.unique( events_df.equipment_identifier.values )
    event_to_int = dict(zip( event_id, np.arange(event_id.shape[0]) ))
    
    # add event encodings to the dataframes
    events_df["event_idx"] = [ event_to_int[i] for i in events_df.equipment_identifier.values ]

    # partitioning the event dataframe to track individual events
    eq_diff_idx = np.where( np.diff(events_df.event_idx.values) )[0]+1
    # insert first element (zero) ## otherwise left out!
    eq_diff_idx = np.insert(eq_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    eq_diff_idx = np.append( eq_diff_idx, events_df.event_idx.values.shape[0] )

    # calculate trajectories of events (to which tower it connects to and when)
    eq_trajectories_towers = []
    eq_trajectories_time = []
    print("calculate trajectories of events..")
    for i in tqdm( range( eq_diff_idx.shape[0]-1 ) ):
        start_ = eq_diff_idx[i]
        end_ = eq_diff_idx[i+1]
        eq_trajectories_towers.append( events_df.tower_idx.values[ start_:end_ ] )
        eq_trajectories_time.append( events_df.event_datetime.values[ start_:end_ ] )
    
    # these arrays holds the information about the events
    eq_trajectories_towers = np.array( eq_trajectories_towers ) # array of variable length arrays!
    eq_trajectories_time = np.array( eq_trajectories_time ) # array of variable length arrays!
    
    # check for unique values and drop events that stand still for the whole day
    eq_trajectories_towers_uq = np.array([ np.shape(np.unique(i))[0] for i in eq_trajectories_towers ])
    eq_trajectories_time = eq_trajectories_time[ eq_trajectories_towers_uq > 1 ]
    eq_trajectories_towers = eq_trajectories_towers[ eq_trajectories_towers_uq > 1 ]
    
    # load datetime variable to code time as minutes
    dates_clock = np.loadtxt( "../Data/event_datetime.csv").astype(int) # time on clock
    dates_time = np.arange( 1440 ) # time in sec
    time_to_sec = dict( zip(dates_clock, dates_time) )

    # building the day graph, defining it with its edge list.. (it is paralellizable!)
    graph_edge_list_raw = []
    eps = 1e-6 # add small time to avoid division by zero
    print("building the graph, defining it with its edge list..")
    for curr in tqdm( range( eq_trajectories_towers.shape[0] ) ):
        
        # get indices
        eq_path = np.vstack( (eq_trajectories_towers[curr][:-1], eq_trajectories_towers[curr][1:]) ).T
        
        # filter if source and destination is the same
        filt = (eq_path[:,0] != eq_path[:,1])
        eq_path = eq_path[filt]
        
        # get time in minutes format
        eq_time_ = np.array( list( map(time_to_sec.get, eq_trajectories_time[curr]) ) )
        eq_time_min = eq_time_[1:] - eq_time_[:-1]
        eq_time_min = eq_time_min[filt] / 60 # this becomes hour for calc speed!
        eq_time_ = eq_time_[:-1][filt]
        
        src_coords = np.array([ tower_coords_all[ m ] for m in eq_path[:,0] ])
        dst_coords = np.array([ tower_coords_all[ n ] for n in eq_path[:,1] ])
        
        eq_dist_km = np.sqrt( np.sum( ( dst_coords - src_coords )**2, 1 ) ) / 1000
        
        eq_speed_kmh = (eq_dist_km/eq_time_min+eps).astype(int)
        filt = np.logical_and( eq_speed_kmh > 3, eq_speed_kmh < 180 )
        graph_edge_list_raw.append( np.concatenate( (eq_path[filt], 
                                                     np.ones(filt.sum()).reshape(-1,1) ), 
                                                   axis=1).astype(int) )
    
    # finalizing the edge list by removing duplicate lines and collecting edge weights
    print("finalizing the output..")
    graph_edge_list_raw = np.concatenate( graph_edge_list_raw )
    uq_, counts_ = np.unique(graph_edge_list_raw, axis=0, return_counts=True)
    graph_edge_list = np.concatenate( (uq_[:,:2], counts_.reshape(-1, 1)), axis=1 )
    
    print("saving results to disk..")
    # saving resulting dataframe to disc with coordinates of the source and destination towers
    savename = target + 'output_day-graph-by-edgelist_' + events_path.split('.csv')[0][-8:]+'.csv.gz'
    print(savename)
    pd.DataFrame( np.concatenate( ( graph_edge_list, 
                        np.array([ tower_coords_all[s] for s in graph_edge_list[:,0] ]), 
                        np.array([ tower_coords_all[t] for t in graph_edge_list[:,1] ]) ), 
                    axis=1  ), 
                  columns=[ ["src", "dst", "weight", 
                             "src_eovx", "src_eovy", 
                             "dst_eovx", "dst_eovy"] ] ).to_csv( savename, 
                                                                 index=False, 
                                                                 compression='gzip')

# call this function like this:
# calculate_day_graph( source, target, files_events[100] )
for q in range( args.start_idx, args.end_idx ):
    calculate_day_graph( source, target, files_events[q] )