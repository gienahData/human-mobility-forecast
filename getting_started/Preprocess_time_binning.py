import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Locate source folder
source = '/mnt2/data/csv/'

# Locate filenames that needs to be processed
files_events = np.array( sorted( [ i for i in os.listdir( source ) if 'Events' in i  ] ) )
#files_events = np.array( [ i for i in files_events if 'EVENTS' in i ] )

# Define time bins to split original dataframe into -> this setting is for 5 minutes
abstimes = np.linspace( 0, 1440, int(1440/5)+1 ).astype(int)

def time_in_minutes( time ):
    '''
    Converts the event datetime to absolute scale coded in minutes from 0:00 until 24:00.
    '''
    num_str = str(time)
    num_digits = len(num_str)
    #print(num_str, num_digits)
    if num_digits < 3:
        #print( '2', time )
        return time
    elif num_digits == 3:
        #print( '3', int(num_str[0])*60 + int(num_str[1:]) )
        return int(num_str[0])*60 + int(num_str[1:])
    elif num_digits == 4:
        #print( '4', int(num_str[:2])*60 + int(num_str[2:]) )
        return int(num_str[:2])*60 + int(num_str[2:])
    else:
        print("error!", num_str)


def time_binning( source_path, csv_name ):
    '''
    Time binning and saving file in the same format as the original dataframe.
    '''
    # Load data
    events_df = pd.read_csv( source_path+csv_name, delimiter=';', compression='gzip' )
    events_df.columns = ["event_datetime", "equipment_identifier", "network_identifier", "event_type", "event_direction","device_tac"]
    
    # Calculation of absolute time
    event_abstime = np.zeros( events_df.event_datetime.shape[0], dtype=np.int64 )
    event_datetime = events_df.event_datetime.values
    
    print('Calculation of absolute times...')
    for i in tqdm( range( events_df.event_datetime.shape[0] ) ):
        event_abstime[i] = time_in_minutes( event_datetime[i] )
    
    # Saving calculated absolute times to the last column of the dataframe
    #events_df["event_abstime"] = event_abstime
    
    # Sort array with absolute times
    #sort_idx = np.argsort( events_df.event_abstime.values )
    #events_df = events_df.loc[sort_idx]
    #events_df.reset_index(inplace=True)
    #print(events_df.head(100))

    # Filter out events that fall into this short time interval
    print('Calculation of time binned csv...')
    for i in tqdm( range( 0, abstimes.shape[0]-1 ) ):
        time_bin_idx = np.logical_and( abstimes[i] <= event_abstime, abstimes[i+1] > event_abstime )
        events_df_time_binned = events_df[ time_bin_idx ]
        #events_df_time_binned.reset_index(inplace=True)
        savename = csv_name.split('.csv')[0]+'_'+str(abstimes[i])+'.csv'+csv_name.split('.csv')[1]
        events_df_time_binned.to_csv( source+'time_binned/'+savename, index=False, header=False, compression='gzip', sep=';' )

# Do process for all input files
for k in range( 0, files_events.shape[0] ):
    time_binning( source, files_events[k] )
