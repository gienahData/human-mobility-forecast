# This script is created by A. Biricz 04.01.2021., modified 04.04.2021.

## This script generates output of 
# 30 (default) minutes long intervals  ## CHANGED for 60
# and does no interpolation at all!

## Arguments:
# --source_folder: where events and polygons data located
# --source_pol_folder: where processed polygon files located
# --target_folder: save folder

# Example for running:
# python3 .py --source_folder '/media/Data_storage/Mobilcell/Data/'  --source_pol_folder '/media/Data_storage/Mobilcell/DayPolygonData/' --target_folder '/media/Data_storage/Mobilcell/DayEventData/' --start_idx 0 --end_idx 1

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
parser.add_argument("--interval", type=int, default=60, help="time interval") ## CHANGED TO 60
parser.add_argument("--source_folder", default='/mnt2/data/csv/' )
parser.add_argument("--source_pol_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/DayEventData/' )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
source_pol = os.path.abspath( args.source_pol_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/DayPolygonData/'
target = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'

files_events = np.array( sorted([ i for i in os.listdir(source) if 'EVENTS' in i]) )

def prepare_time_interval_graph_data( source, source_pol, target, events_path ):

    ## PROCESS PREPARATION

    # load data
    print("loading input data..")
    events_df = pd.read_csv( source+events_path, delimiter=';', usecols=[0,1,2] )

    # load global, fix tower data
    towers_df = pd.read_csv( source_pol+'fixed_merged_tower_locations_ovr-10_d-40.csv' ) ## CHANGED FOR NEW MERGING DATA

    # calculate tower encodings
    tower_id = np.unique( towers_df.original_id.values )
    tower_to_int = dict( zip( towers_df.original_id.tolist(), towers_df.tower_id.tolist() ) )

    tower_coords_all = dict( zip( towers_df.values[:,1].tolist(), 
                                  towers_df.values[:,2:4].tolist() ) )

    # add encodings to the dataframes
    print("encode ids of towers and events..")
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

    # load datetime variable to code time as minutes
    dates_clock = np.loadtxt( "../Data/event_datetime.csv").astype(int) # time on clock
    dates_time = np.arange( 1440 ) # time in sec
    time_to_sec = dict( zip(dates_clock, dates_time) )


    ## PROCESS PART 1:

    # array for storing moving events and their info
    eq_info_all = np.zeros( ( np.sum([ len(i) for i in eq_trajectories_towers ]), 11 ), 
                            dtype=np.int32 )
    header = ['id', 'start_time_min', 'src_x', 'src_y', 'dst_x', 'dst_y', 
                    'trip_time_min', 'dist_m', 'speed_ms', 'src', 'dst']
    # array for storing standby event counts associated to a tower
    tower_datetime_events = np.zeros( ( max(tower_to_int.values())+1, dates_time.shape[0]), dtype=np.uint32 )

    counter = 0
    passed = 0
    eps = 1e-6 # add small time to avoid division by zero
    print('calculate nodes and edges of the mobility graph..')
    for curr in tqdm( range( eq_trajectories_towers.shape[0] ) ):
        
        # get the registered time points of the current equipment's trajectory in the absolute (minutes) scale
        eq_time_ = np.array( list( map(time_to_sec.get, eq_trajectories_time[curr]) ) )

        # filter if the registered time point at the source and destination towers are the same
        eq_timepoint = np.vstack( (eq_time_[:-1], eq_time_[1:]) ).T
        filt = (eq_timepoint[:,0] != eq_timepoint[:,1])
        filt = np.insert( filt, 0, True )
        
        # valid time points and events
        eq_time_ = eq_time_[filt]
        eq_tower = eq_trajectories_towers[curr][filt]
            
        # path of the equipment with tower ids, a motion is always between 2 towers
        eq_path = np.vstack( (eq_tower[:-1], eq_tower[1:]) ).T
        
        if eq_time_.shape[0] > 1:
        
            ## mutual part of calculations
            
            # get elapsed time for each path
            eq_elapsedtime_min = eq_time_[1:] - eq_time_[:-1]
            eq_elapsedtime_min = eq_elapsedtime_min # elapsed time in minutes!

            # source and destination towers's coordinates
            src_coords = np.array([ tower_coords_all[ m ] for m in eq_path[:,0] ])
            dst_coords = np.array([ tower_coords_all[ n ] for n in eq_path[:,1] ])

            # distance traveled in m
            eq_dist_m = np.sqrt( np.sum( ( dst_coords - src_coords )**2, 1 ) )

            # average travelling speed
            eq_speed_ms = (eq_dist_m/(eq_elapsedtime_min*60)+eps).astype(int)
            
            ## -- ##
            
            ## keep only standing equipments
            
            filt_standing = np.logical_and( eq_speed_ms > -1, eq_speed_ms < 1 ) 
            filt_standing = np.insert( filt_standing, 0, True )
            eq_time_standing = eq_time_[filt_standing]
            eq_tower_standing = eq_tower[filt_standing]

            # save valid standing event
            tower_datetime_events[ eq_tower_standing, eq_time_standing ] += 1
            
            ## -- ##        
            
            ## keep only moving equipments
            
            filt_moving = np.logical_and( eq_speed_ms > 0, eq_speed_ms < 50 )
            eq_time_moving = eq_time_[:-1]
            
            # saving calculated trajectory
            eq_info = np.concatenate( (
                                curr*np.ones( filt_moving.sum(), dtype=np.int32).reshape(-1,1),
                                eq_time_moving[filt_moving].reshape(-1,1), 
                                src_coords[filt_moving],
                                dst_coords[filt_moving],
                                eq_elapsedtime_min[filt_moving].reshape(-1,1),
                                eq_dist_m[filt_moving].reshape(-1,1), 
                                eq_speed_ms[filt_moving].reshape(-1,1),
                                eq_path[:,0][filt_moving].reshape(-1,1),
                                eq_path[:,1][filt_moving].reshape(-1,1) ), axis=1).astype(int)
            for l in range(eq_info.shape[0]):
                eq_info_all[counter] = eq_info[l]
                counter += 1

    
        # if an equipment has only 1 registered event it will be left out!
        else:
            passed += 1 # counts how many equipment has only 1 registered event

    # drop not needed elements from the end of the array
    eq_info_all = eq_info_all[:counter]

    ## PROCESS PART 2:

    # events that last longer than 60 minutes (default) could be further processed, but drop them for now ## CHANGED to 60
    filt_long_trips = eq_info_all[:, 6 ] > args.interval # 6th index means time
    eq_info_all = eq_info_all[ ~filt_long_trips ]

    # save output
    print('saving results to disk..')
    np.savez( target + 'output_daily-events-data_max-trip-min-'+str(args.interval)+'_' + events_path.split('.csv')[0][-8:],
              edge_data=eq_info_all, node_data=tower_datetime_events )


# call this function like this:
# prepare_time_interval_graph_data( source, target, files_events[100] )
for q in range( args.start_idx, args.end_idx ):
    print('working on', files_events[q])
    prepare_time_interval_graph_data( source, source_pol, target, files_events[q] )