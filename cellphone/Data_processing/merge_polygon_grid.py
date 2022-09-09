# This script is created by A. Biricz, 30.03.2021.

## This script uses previously calculated fixed locations
#  and the overlap matrices to merge individual cells into 
#  "supercells" thats positions will be the tower positions 
#  and the nodes of the mobility graph 

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
import networkx as nx

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--source_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/DayPolygonData/' )


args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/'
target = os.path.abspath( args.target_folder ) + '/'

# Settings of merging individual cells
## meaning: at least 10% overlapping cells are merged, then distance is checked among them
threshold_ovr = 0.10 # fraction of overlap (this is the first criterium)
threshold_dist = 40*127 #127*10 # distance in m (this comes second) (roughly 5000 m)

tower_loc_df = pd.read_csv( source+'fixed_tower_locations.csv' )

# all unique ids of polygons for the whole year
tower_id_all_global = np.load( source+'unique-tower-id_all.npy' )[:,0] # 0th column contains the ids
tower_to_int_all = dict( zip( tower_id_all_global, np.arange( tower_id_all_global.shape[0] ) ) )
int_to_tower = dict( zip(tower_loc_df.tower_id, tower_loc_df.original_id) )
int_to_pos = dict( zip(tower_loc_df.tower_id, tower_loc_df.iloc[:,2:4].values.tolist() ) )

# list overlap matrix filenames
files_ovr = np.array( sorted( [ i for i in os.listdir( source ) if 'overlap' in i] ) )

## uncomment if fresh run
# array for storing the "global" overlap matrix
#ovr_mat_global = np.zeros( ( len(tower_to_int_all), len(tower_to_int_all) ), np.float16)

# loading all daily overlap matrices
#print("Loading and summing daily overlap matrices...")
#for t in tqdm( range( files_ovr.shape[0] ) ):
#    ovr_mat_global += np.load( source + files_ovr[t] )['ovr']
#ovr_mat_global = np.save(source+'ovr_mat_global.npy')

# uncomment for shortcut if file exist on disk
ovr_mat_global = np.load(source+'ovr_mat_global.npy')

# normalize the global overlap matrix by the diagonal values
norm_factor = np.ones( ovr_mat_global.shape[0], dtype=np.float32 )
nonzero = ovr_mat_global[ np.diag_indices(ovr_mat_global.shape[0]) ] > 0
norm_factor[nonzero] = ovr_mat_global[ np.diag_indices(ovr_mat_global.shape[0]) ][nonzero]
ovr_mat_normed = ovr_mat_global / norm_factor

def euclidean_distance( x, y ):
    return np.sqrt( np.sum( (x-y)**2 ) )

ovr_mat = ovr_mat_normed
ovr_mat_thrs_idxs = np.argwhere( ovr_mat > threshold_ovr )

ovr_diff_idx = np.where( np.diff( ovr_mat_thrs_idxs[:,0] ) )[0]+1
# insert first element (zero) ## otherwise left out!
ovr_diff_idx = np.insert(ovr_diff_idx, 0, 0, axis=0)
# insert last element (size of array) ## otherwise left out!
ovr_diff_idx = np.append( ovr_diff_idx, ovr_mat_thrs_idxs.shape[0] )
# generation of index array
ovr_mat_indexing = np.vstack( ( ovr_diff_idx[:-1], ovr_diff_idx[1:] ) ).T

print('Merging cells...')
merge_idxs_all = []
for i in tqdm( range(ovr_mat_indexing.shape[0]) ):
    # get current tower indices, where distance needs to be checked
    curr_idxs = ovr_mat_thrs_idxs[ ovr_mat_indexing[i,0]:ovr_mat_indexing[i,1] ]
    curr_idxs = curr_idxs[ curr_idxs[:,0] != curr_idxs[:,1] ]
    
    if len(curr_idxs) > 0:
        # current line (current cell to measure distance to)
        base_pos = np.array( int_to_pos[ curr_idxs[0,0] ] )
        check_pos = np.array( [ int_to_pos[q] for q in curr_idxs[:,1] ] )
        dists = np.array( [ euclidean_distance(base_pos, s ) for s in check_pos ] )
        merge_idxs = curr_idxs[ np.argwhere( dists < threshold_dist ), 1]
        if len(merge_idxs) > 0:
            merge_idxs = np.insert( merge_idxs, 0, i )
            merge_idxs_all.append( merge_idxs )

print("Building a graph and looking for subgraphs to do merging...")
# build a binary matrix based on overlap and distance calculated previously
ovr_mat_int = np.zeros( ovr_mat.shape, dtype=np.uint16 )
for n in merge_idxs_all:
    line = n[0]
    idx_to_add_at = n[1:]
    np.add.at( ovr_mat_int[line], idx_to_add_at, 1 )

# accept only the symmetric values in this binary matrix
# (A and B cells must both overlap with each other to be merged)
merge_idxs_symm = []
for a in tqdm( range( ovr_mat_int.shape[0] ) ):
    inline_idx_symm = np.argwhere( np.logical_and( ovr_mat_int[a] > 0, ovr_mat_int[:,a] > 0 ) ).flatten()
    if inline_idx_symm.shape[0] > 0:
        merge_idxs_symm.append( np.insert( inline_idx_symm, 0, a ) )

# build final matrix
ovr_mat_int = np.zeros( ovr_mat.shape, dtype=np.uint16 )
for n in merge_idxs_symm:
    line = n[0]
    idx_to_add_at = n[1:]
    np.add.at( ovr_mat_int[line], idx_to_add_at, 1 )

# make it binary to be able to treat as an adjacency matrix of a graph
ovr_mat_to_graph = (ovr_mat_int > 0)*1

# build a graph in order to find fully connected subgraphs
# which is the key step in merging individual cells
G = nx.from_numpy_matrix(ovr_mat_to_graph)

print( 'Graph nodes and edges: ', len( G.nodes() ), len( G.edges() ) )

# get all of the connected nodes in all of the subgraphs
group_idxs = [ list(c) for c in sorted(nx.connected_components(G),  key=len, reverse=True) ]

# prepare to save calculated groups in the same format
print("Saving output to disk...")
groups_save = np.zeros( len(int_to_tower), dtype=np.uint16 )
for n in range( len(group_idxs) ):
    idx_to_add_at = group_idxs[n]
    np.add.at( groups_save, idx_to_add_at, n )

cell_groups = np.vstack( (tower_loc_df.original_id.values,
                          tower_loc_df.tower_id.values,
                          tower_loc_df.mean_x.values, 
                          tower_loc_df.mean_y.values, 
                          groups_save) ).T

sort_idx = np.argsort( cell_groups[:,-1] )
cell_groups = cell_groups[sort_idx]

diff_idx = np.where( np.diff( cell_groups[:,-1] ) )[0]+1
# insert first element (zero) ## otherwise left out!
diff_idx = np.insert( diff_idx, 0, 0, axis=0 )
# insert last element (size of array) ## otherwise left out!
diff_idx = np.append( diff_idx, cell_groups.shape[0] )
cell_groups_indexing = np.vstack( ( diff_idx[:-1], diff_idx[1:] ) ).T

cell_groups_coords = np.zeros( (cell_groups.shape[0], 2), dtype=np.uint32 )
for l in cell_groups_indexing:
    current_coords = np.round( np.mean( cell_groups[ l[0]:l[1], 2:4 ], axis=0 ), 0 ).astype(np.uint32)
    cell_groups_coords[ l[0]:l[1] ] = current_coords

cell_groups_save = np.concatenate( (cell_groups[:,[0, 4]], cell_groups_coords), axis=1 )
sort_idx = np.argsort( cell_groups_save[:,0] )
cell_groups_save = cell_groups_save[ sort_idx ]

pd.DataFrame( cell_groups_save, 
              columns=['original_id', 'tower_id', 
                       'mean_x', 'mean_y'] ).to_csv( target+'fixed_merged_tower_locations_ovr-10_d-40.csv', 
                                                       index=False )