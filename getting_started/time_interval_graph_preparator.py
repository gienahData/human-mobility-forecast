# this script is created by A. Biricz, last modified 31.12.2020.

#python3 time_interval_graph_preparator.py --start_idx 100 --end_idx 101 --source_folder '/media/Data_storage/Mobilcell/Data/' --target_folder '/media/Data_storage/Mobilcell/DayEventData/'


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
parser.add_argument("--target_folder", default='/mnt/DayEventData/' )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
target = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'

files_events = np.array( sorted([ i for i in os.listdir(source) if 'EVENTS' in i]) )
files_poligons = np.array( sorted([ i for i in os.listdir(source) if 'POLIGONS' in i]) )
files_events_cleaned = np.array( sorted([ i for i in os.listdir(source) if 'Events' in i]) )



def prepare_time_interval_graph_data( source, target, poligons_path, events_path ):

    # load data
    print("loading input data")
    poligons_df = pd.read_csv( source+poligons_path, delimiter=';' )
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

    # partitioning the poligon dataframe to locate individual towers
    tower_diff_idx = np.where( np.diff(poligons_df.tower_idx.values) )[0]+1
    # insert first element (zero) ## otherwise left out!
    tower_diff_idx = np.insert(tower_diff_idx, 0, 0, axis=0)
    # insert last element (size of array) ## otherwise left out!
    tower_diff_idx = np.append( tower_diff_idx, poligons_df.tower_idx.values.shape[0] )

    # collect all rasters for the towers
    tower_rasters = []
    print("collect all rasters for the towers..")
    for i in tqdm( range( tower_diff_idx.shape[0]-1 ) ):
        start_ = tower_diff_idx[i]
        end_ = tower_diff_idx[i+1]
        tower_rasters.append( poligons_df.eov_idx.values[ start_:end_ ] )
    tower_rasters = np.array(tower_rasters)

    # calculate coordinates of the towers (and its error with std for now)
    tower_coords_all = []
    tower_std_all = []
    print("calculate coordinates of the towers..")
    for i in tqdm( range( tower_rasters.shape[0] ) ):
        tower_coords_all.append( np.mean( raster_coords[ tower_rasters[i] ], 0 ) )
        tower_std_all.append( np.std( raster_coords[ tower_rasters[i] ], 0 ) )  
    tower_coords_all = np.array( tower_coords_all )
    tower_std_all = np.array( tower_std_all )

    # load datetime variable to code time as minutes
    dates_clock = np.loadtxt( "event_datetime.csv").astype(int) # time on clock
    dates_time = np.arange( 1440 ) # time in sec
    time_to_sec = dict( zip(dates_clock, dates_time) )


    ## Processing PART 1:

    eq_info_all = np.zeros( ( np.sum([ len(i) for i in eq_trajectories_towers ]), 9 ), 
                            dtype=np.int32 )
    counter = 0
    eps = 1e-6 # add small time to avoid division by zero
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
        
        # calculate speed and filter with it
        eq_dist_km = np.sqrt( np.sum( (tower_coords_all[ eq_path[:,1] ] - \
                        tower_coords_all[ eq_path[:,0] ])**2, 1 ) ) / 1000
        eq_speed_kmh = (eq_dist_km/eq_time_min+eps).astype(int)
        filt = np.logical_and( eq_speed_kmh > 3, eq_speed_kmh < 180 )
        
        # saving calculated trajectory
        #eq_info = pd.DataFrame( 
        eq_info = np.concatenate( (
                            curr*np.ones( filt.sum(), dtype=np.int32).reshape(-1,1),
                            eq_time_[filt].reshape(-1,1), 
                            tower_coords_all[ eq_path[:,0] ][filt], 
                            tower_coords_all[ eq_path[:,1] ][filt], 
                            eq_time_min[filt].reshape(-1,1)*60,
                            eq_dist_km[filt].reshape(-1,1)*1000, 
                            eq_speed_kmh[filt].reshape(-1,1)/3.6 ), axis=1).astype(int) #,
    #columns=['id',start_time_min','src_x','src_y','dst_x', 'dst_y','trip_time_min', 'dist_m', 'speed_ms'])
        
        for l in range(eq_info.shape[0]):
            eq_info_all[counter] = eq_info[l]
            counter += 1

    # drop not needed elements from the end of the array
    eq_info_all = eq_info_all[:counter]

    ## Processing PART 2:

    # events that last longer than 15 minutes should be further processed
    filt_long_trips = eq_info_all[:, 6 ] > 15 # 6th index means time
    eq_info_all_long = eq_info_all[ filt_long_trips ]

    # first remove these longer events from the list
    eq_info_all = eq_info_all[ ~filt_long_trips ]


    # events that last longer than 65 minutes will not be interpolated!
    filt_long_trips = eq_info_all_long[:, 6 ] < 65 # 6th index means time
    eq_info_all_long = eq_info_all_long[ filt_long_trips ]


    # linear interpolation of the longer events to match the time scale of 15 minutes
    to_add_all = np.zeros( ( np.sum([ len(i) for i in eq_trajectories_towers ]), 9 ), 
                            dtype=np.int32 )

    counter2 = 0
    for q in tqdm( range( eq_info_all_long.shape[0] ) ):
        eq_current = eq_info_all_long[q] # select a trip

        # request info of the current movement
        eq_id = eq_current[0]
        start_time = eq_current[1]
        trip_time = eq_current[6]
        end_time = start_time + trip_time
        src_coords = eq_current[2:4]
        dst_coords = eq_current[4:6]
        dist = eq_current[7]
        speed = eq_current[8]
        num_to_interp = np.int( np.ceil( eq_current[6] / 15 ) ) # CHANGED TO CEIL!

        # do linear interpolation (coordinates and time between need to be calc.)
        time_samples = np.linspace( start_time, end_time, num_to_interp+1 ).astype(int)
        coords_samples = np.linspace( src_coords, dst_coords, num_to_interp+1 ).astype(int)

        # index array for consecutive events
        idx_samples = np.vstack( (np.arange(coords_samples.shape[0])[1:], 
                                np.arange(coords_samples.shape[0])[:-1]) ).T
        
        # just add speed from previously calculated trip
        speed_samples = np.ones(num_to_interp, dtype=np.int32)*speed # speed remains the same!

        # divide original distance into parts
        dist_samples = (np.ones(num_to_interp)*dist/num_to_interp).astype(int)
    
        # putting all that together and collect data
        to_add = ( np.concatenate( (  
                    eq_id*np.ones(dist_samples.shape[0], dtype=np.int).reshape(-1,1),
                    time_samples[:-1].reshape(-1,1), 
                    coords_samples[:-1], coords_samples[1:], 
                    np.diff(time_samples).reshape(-1,1), 
                    dist_samples.reshape(-1,1), 
                    speed_samples.reshape(-1,1)), axis=1) )

        for l in range(to_add.shape[0]):
            to_add_all[ counter2 ] = to_add[l]
            counter2 += 1
    
    # drop not needed elements from the end of the array
    to_add_all = to_add_all[:counter2]
    eq_info_final = np.concatenate( (eq_info_all, to_add_all), axis=0 )

    # saving results to disk
    #savename = target + 'output_daily-events-data_' + events_path.split('.csv')[0][-8:]+'.csv.gz'
    #pd.DataFrame( eq_info_final,
    #    columns=['id','start_time_min','src_x','src_y',
    #            'dst_x', 'dst_y','trip_time_min', 'dist_m', 
    #            'speed_ms']).to_csv( savename, index=False, compression='gzip' )
        
    # saving as numpy array is much-much faster!
    np.save( target + 'output_daily-events-data_' + \
             events_path.split('.csv')[0][-8:],
             eq_info_final )


# call this function like this:
# prepare_time_interval_graph_data( source, target, files_poligons[100], files_events[100] )
for q in range( args.start_idx, args.end_idx ):
    prepare_time_interval_graph_data( source, target, files_poligons[q], files_events[q] )


