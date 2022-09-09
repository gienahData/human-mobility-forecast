# this script is created by A. Biricz, 29.11.2020.

## Arguments:
# --source_folder: where events and polygons data located
# --target_folder: save folder
# --start_idx: first file's index
# --end_idx: last file's index

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
parser.add_argument("--target_folder", default='/mnt/DayGraphData/' )

args = parser.parse_args()


# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
target = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'

files_events = np.array( sorted([ i for i in os.listdir(source) if 'EVENTS' in i]) )
files_poligons = np.array( sorted([ i for i in os.listdir(source) if 'POLIGONS' in i]) )
files_events_cleaned = np.array( sorted([ i for i in os.listdir(source) if 'Events' in i]) )

def calculate_day_graph( source, target, poligons_path, events_path ):
    
    # load data
    print("loading input data")
    print(poligons_path)
    poligons_df = pd.read_csv( source+poligons_path, delimiter=';' )
    print(events_path)
    events_df = pd.read_csv( source+events_path, delimiter=';' )

    # drop poligons outside of the country
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovx.values < 366660, 
                                               poligons_df.eovx.values > 48210 ) ]
    poligons_df = poligons_df[ np.logical_and( poligons_df.eovy.values < 934219, 
                                               poligons_df.eovy.values > 426341 ) ]

    # create coordinate system for the rasters (much easier and faster to generate than search and match)
    start_x = poligons_df.eovx.values.min()
    start_y = poligons_df.eovy.values.min()
    end_x = poligons_df.eovx.values.max()
    end_y = poligons_df.eovy.values.max()
    num_x = int( ( end_x - start_x ) / 127 )
    num_y = int( ( end_y - start_y ) / 127 )
    raster_x = np.arange(start_x, end_x+127, 127, dtype=np.int32)
    raster_y = np.arange(start_y, end_y+127, 127, dtype=np.int32)
    
    # get coordinate vector
    raster_coords = np.array( list(product( raster_x, raster_y )) )
    raster_coords[:3], raster_coords.shape

    # calculate raster encodings
    poligons_df['eovx_num'] = ( (poligons_df.eovx - start_x) / 127 ).astype(int)
    poligons_df['eovy_num'] = ( (poligons_df.eovy - start_y) / 127 ).astype(int)
    poligons_df['eov_idx'] = poligons_df.eovx_num * (num_y+1) + poligons_df.eovy_num
    
    # calculate tower encodings
    tower_id = np.unique( poligons_df.network_identifier.values )
    uniq, counts = np.unique(poligons_df.eov_idx, return_counts=True)
    tower_to_int = dict(zip( tower_id, np.arange(tower_id.shape[0]) ))

    # add encodings to the dataframes
    poligons_df["tower_idx"] = [ tower_to_int[i] for i in poligons_df.network_identifier.values ]
    events_df["tower_idx"] = [ tower_to_int[i] for i in events_df.network_identifier.values ]
    
    # sort the smaller dataframe for much quicker searching
    poligons_df.sort_values( by='tower_idx', inplace=True )
    
    # calculate event (equipment id) encodings
    event_id = np.unique( events_df.equipment_identifier.values )
    event_to_int = dict(zip( event_id, np.arange(event_id.shape[0]) ))
    
    # add event encodings to the dataframes
    events_df["event_idx"] = [ event_to_int[i] for i in events_df.equipment_identifier.values ]

    # partitioning the event dataframe to track individual events
    eq_diff_idx = np.where( np.diff(events_df.event_idx.values) )[0]+1
    
    # calculate trajectories of events (to which tower it connects to and when)
    eq_trajectories_towers = []
    eq_trajectories_time = []
    print("calculate trajectories of events..")
    for i in range( eq_diff_idx.shape[0]-1 ):
        start_ = eq_diff_idx[i]
        end_ = eq_diff_idx[i+1]
        eq_trajectories_towers.append( events_df.tower_idx.values[ start_:end_ ] )
        eq_trajectories_time.append( events_df.event_datetime.values[ start_:end_ ] )
    
    # TO-CHECK: it is highly possible that the FIRST AND LAST EVENT IS DROPPED!
    
    # these arrays holds the information about the events
    eq_trajectories_towers = np.array( eq_trajectories_towers ) # array of variable length arrays!
    eq_trajectories_time = np.array( eq_trajectories_time ) # array of variable length arrays!
    
    # check for unique values and drop events that stand still for the whole day
    eq_trajectories_towers_uq = np.array([ np.shape(np.unique(i))[0] for i in eq_trajectories_towers ])
    eq_trajectories_time = eq_trajectories_time[ eq_trajectories_towers_uq > 1 ]
    eq_trajectories_towers = eq_trajectories_towers[ eq_trajectories_towers_uq > 1 ]

    # partitioning the poligon dataframe to locate individual towers
    tower_diff_idx = np.where( np.diff(poligons_df.tower_idx.values) )[0]+1
    # insert first element (zero) ## otherwise left out!
    tower_diff_idx = np.insert(tower_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    tower_diff_idx = np.append( tower_diff_idx, poligons_df.tower_idx.values.shape[0] )

    # collect all rasters for the towers
    tower_rasters = []
    print("collect all rasters for the towers..")
    for i in range( tower_diff_idx.shape[0]-1 ):
        start_ = tower_diff_idx[i]
        end_ = tower_diff_idx[i+1]
        tower_rasters.append( poligons_df.eov_idx.values[ start_:end_ ] )
    tower_rasters = np.array(tower_rasters)

    # calculate coordinates of the towers (and its error with std for now)
    tower_coords_all = []
    tower_std_all = []
    print("calculate coordinates of the towers..")
    for i in range( tower_rasters.shape[0] ):
        tower_coords_all.append( np.mean( raster_coords[ tower_rasters[i] ], 0 ) )
        tower_std_all.append( np.std( raster_coords[ tower_rasters[i] ], 0 ) )  
    tower_coords_all = np.array( tower_coords_all )
    tower_std_all = np.array( tower_std_all )
    
    # load datetime variable to code time as minutes
    dates_clock = np.loadtxt( "event_datetime.csv").astype(int) # time on clock
    dates_time = np.arange( 1440 ) # time in sec
    time_to_sec = dict( zip(dates_clock, dates_time) )

    # building the day graph, defining it with its edge list.. (it is paralellizable!)
    graph_edge_list_raw = []
    eps = 1e-6 # add small time to avoid division by zero
    print("building the graph, defining it with its edge list..")
    for curr in range( eq_trajectories_towers.shape[0] ):
        # get indices
        eq_path = np.vstack( (eq_trajectories_towers[curr][:-1], eq_trajectories_towers[curr][1:]) ).T
        # filter if source and destination is the same
        filt = (eq_path[:,0] != eq_path[:,1])
        eq_path = eq_path[filt]
        eq_time_ = np.array( list( map(time_to_sec.get, eq_trajectories_time[curr]) ) )
        eq_time_min = eq_time_[1:] - eq_time_[:-1]
        eq_time_min = eq_time_min[filt] / 60
        eq_dist_km = np.sqrt( np.sum( (tower_coords_all[ eq_path[:,1] ] - \
                          tower_coords_all[ eq_path[:,0] ])**2, 1 ) ) / 1000
        eq_speed_kmh = (eq_dist_km/eq_time_min+eps).astype(int)
        filt = np.logical_and( eq_speed_kmh > 0, eq_speed_kmh < 180 )
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
                                    tower_coords_all[ graph_edge_list[:,0] ], 
                                    tower_coords_all[ graph_edge_list[:,1] ] ), 
                                  axis=1  ), 
                  columns=[ ["src", "dst", "weight", 
                             "src_eovx", "src_eovy", 
                             "dst_eovx", "dst_eovy"] ] ).to_csv( savename, 
                                                                 index=False, 
                                                                 compression='gzip')

# call this function like this:
# calculate_day_graph( source, target, files_poligons[100], files_events[100] )
for q in range( args.start_idx, args.end_idx ):
    calculate_day_graph( source, target, files_poligons[q], files_events[q] )
