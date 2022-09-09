# Created by A. Biricz 01.12.2020., then revisited, corrected, updated 03.04.2021.

## Arguments:
# --source_folder: where events and polygons data located
# --target_folder: save folder
# --source_pol_folder: load universal grid from here
# --start_idx: first file's index
# --end_idx: last file's index

## python3 day_graph_analytics_global_undirected.py --start_idx 0 --end_idx 1 --source_folder /media/Data_storage/Mobilcell/DayGraphData/ --source_pol_folder /media/Data_storage/Mobilcell/DayPolygonData/ --target_folder /media/Data_storage/Mobilcell/DayGraphAnalytics_unweighted/ --unweighted 1

# Load packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import tqdm
import json

import collections
from random import choice
import copy

from graph_tool.all import *

# for running with different arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--start_idx", type=int, default=0, help="start with day 0")
parser.add_argument("--end_idx", type=int, default=365, help="end with day 365")
parser.add_argument("--source_folder", default='/mnt/DayGraphData/' )
parser.add_argument("--source_pol_folder", default='/mnt/DayPolygonData/' )
parser.add_argument("--target_folder", default='/mnt/DayGraphAnalytics/' )
parser.add_argument("--unweighted", type=int, default=0, required=False )

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
source_pol = os.path.abspath( args.source_pol_folder ) + '/' #'/media/Data_storage/Mobilcell/DayPolygonData/'
destination = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'
unweighted = args.unweighted

files = sorted([ i for i in os.listdir(source) if '.csv.gz' in i ])
files = np.array( sorted([ i for i in files if 'output_' in i ]) )

# Load universal, merged, fixed tower locations
tower_df = pd.read_csv( source_pol+'fixed_merged_tower_locations.csv' )
sort_idx = np.argsort( tower_df.tower_id.values )
tower_df = tower_df.iloc[ sort_idx ]
tower_df.reset_index(inplace=True)
coords = np.unique( tower_df.iloc[:,2:], axis=0 )[:,1:]

# Function definitions

def create_day_graph_from_csv( path_to_file ):
    V = pd.read_csv( path_to_file, delimiter=',' ).values

    # create adjacency matrix and make graph undirected
    adj_mat = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.uint32 )
    adj_mat[ V[:,0], V[:,1] ] = V[:,2]
    adj_mat = adj_mat + adj_mat.T 

    # filter out isolated nodes
    adj_mat = adj_mat[ np.ix_( ~np.all( adj_mat == 0, axis=1 ), ~np.all( adj_mat == 0, axis=0 ) ) ]

    if unweighted:
        adj_mat = (adj_mat > 0 )*1 # create binary adjacency matrix
        print('All weights are transformed to binary')

    G = nx.from_numpy_matrix(adj_mat)
    print( "Graph nodes, edges: ", len( G.nodes() ), len( G.edges() ) )
    
    return G

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, str):
        # Encode the key as utf-8
        key = key.encode('utf-8', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('utf-8', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)
        
    #If key is a byte value, decode it to string
    try:
        key = key.decode('utf-8')
    except AttributeError:
        pass

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in list(nxG.graph.items()):
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in list(data.items()):            
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in list(data.items()):            
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)
            
            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in list(data.items()):
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in list(data.items()):
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG


def calculate_num_vertices(g):
    return { 'value': len(list(g._Graph__vertex_properties['id'])) }


def calculate_num_edges(g):
    return { 'value': len(list(g._Graph__edge_properties['weight'])) }


def calculate_assortativity(g):
    value, var = assortativity( g, 'total', eweight=g._Graph__edge_properties['weight'] )
    return {'value': value, 'variance': var }


def calculate_scalar_assortativity(g):
    value, var = scalar_assortativity( g, 'total', eweight=g._Graph__edge_properties['weight'] )
    return {'value': value, 'variance': var }
    

def calculate_pseudo_diameter(g):
    value, _ = pseudo_diameter(g, weights=g._Graph__edge_properties['weight'])
    return {'value': int(value) }


def calculate_min_spanning_tree(g):
    return {'num_edges_involved': sum( list(min_spanning_tree(g, weights=g._Graph__edge_properties['weight'])) ) }


def calculate_vertex_percolation(g):
    """
    Largest connected components relative size : fraction of removed vertices
    """
    vertices = sorted( [v for v in g.vertices()], key=lambda v: v.out_degree() )
    sizes, comp = vertex_percolation(g, vertices)
    np.random.shuffle(vertices)
    sizes2, comp = vertex_percolation(g, vertices)
    
    # fractions to evaluate at
    vert_fracs = np.array( [ 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 
                                0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                                0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 
                                0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 
                                0.996, 0.997, 0.998, 0.999, 1.0 ] )
                   
    # generate indices
    all_idx = ( (sizes.shape[0]-1)*np.ones(vert_fracs.shape[0])*vert_fracs ).astype(int)

    #left_idx = np.arange( 0,  int(sizes.shape[0]/10), int(sizes.shape[0]/100) )
    #middle_idx = np.arange( int(sizes.shape[0]/10), int(sizes.shape[0]*9/10), int(sizes.shape[0]/10) )
    #right_idx = np.arange( int(sizes.shape[0]*9/10), int(sizes.shape[0]), int(sizes.shape[0]/100) )
    #rightmost_idx = np.arange( int(sizes.shape[0]*990/1000), int(sizes.shape[0]), int(sizes.shape[0]/1000) )
    #rightmost_idx = np.append( np.arange( int(sizes.shape[0]*990/1000), int(sizes.shape[0]), int(sizes.shape[0]/1000) ), sizes.shape[0]-1 )
    #all_idx = np.concatenate( (left_idx, middle_idx, right_idx ) ) #, rightmost_idx) )
    #print( all_idx.shape, all_idx )

    #vert_fracs_left = np.round( left_idx / sizes.shape[0], 2 )
    #vert_fracs_middle = np.round( middle_idx / sizes.shape[0], 1 )
    #vert_fracs_right = np.round( right_idx / sizes.shape[0], 2 )
    #vert_fracs_rightmost = np.round( rightmost_idx / sizes.shape[0], 3 )
    #vert_fracs = np.concatenate( (vert_fracs_left, vert_fracs_middle, vert_fracs_right ) ) #, vert_fracs_rightmost) )
    
    all_idx = sizes.shape[0]-1-all_idx # make idx to mean the removed fraction
    #print(vert_fracs.shape, vert_fracs)

    #print( vert_fracs.shape, len( sizes[all_idx]/len(vertices) ) )
    # at fraction of removed vertices the relative size of the largest connected comp.
    items_direct = dict( zip( vert_fracs, sizes[all_idx]/len(vertices) ) )
    items_random = dict( zip( vert_fracs, sizes2[all_idx]/len(vertices) ) )
    return { 'directed': items_direct, 'random': items_random }


def calculate_edge_percolation(g):
    """
    Largest connected components relative size : fraction of removed edges
    """    
    # the (largest) connected component is given by the number of connected nodes
    edges = sorted([(e.source(), e.target()) for e in g.edges()],
                   key=lambda e: e[0].out_degree() * e[1].out_degree())
    vertices = sorted( [v for v in g.vertices()], key=lambda v: v.out_degree() )
    sizes, comp = edge_percolation(g, edges)
    np.random.shuffle(edges)
    sizes2, comp = edge_percolation(g, edges)

    # fractions to evaluate at
    edge_fracs = np.array( [ 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 
                                0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                                0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 
                                0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 
                                0.996, 0.997, 0.998, 0.999, 1.0 ] )
                                
    # generate indices
    all_idx = ( (sizes.shape[0]-1)*np.ones(edge_fracs.shape[0])*edge_fracs ).astype(int)
    #print(all_idx)

    # generate indices
    #left_idx = np.arange( 0,  int(sizes.shape[0]/10), int(sizes.shape[0]/100) )
    #middle_idx = np.arange( int(sizes.shape[0]/10), int(sizes.shape[0]*9/10), int(sizes.shape[0]/10) )
    #right_idx = np.arange( int(sizes.shape[0]*9/10), int(sizes.shape[0]), int(sizes.shape[0]/100) )
    #rightmost_idx = np.append( np.arange( int(sizes.shape[0]*990/1000), int(sizes.shape[0]), int(sizes.shape[0]/1000) ), sizes.shape[0]-1 )
    #all_idx = np.concatenate( (left_idx, middle_idx, right_idx ) ) #, rightmost_idx) ) )
    #print(all_idx.shape, all_idx)

    #edge_fracs_left = np.round( left_idx / sizes.shape[0], 2 )
    #edge_fracs_middle = np.round( middle_idx / sizes.shape[0], 1 )
    #edge_fracs_right = np.round( right_idx / sizes.shape[0], 2 )
    #edge_fracs_rightmost = np.round( rightmost_idx / sizes.shape[0], 3 )
    #edge_fracs = np.concatenate( (edge_fracs_left, edge_fracs_middle, edge_fracs_right ) ) #, edge_fracs_rightmost) ) )

    all_idx = sizes.shape[0]-1-all_idx # make idx to mean the removed fraction
    #print(edge_fracs.shape, edge_fracs)

    #print( edge_fracs.shape, len( sizes[all_idx]/len(vertices) ) )
    # at fraction of removed edges the relative size of the largest connected comp.
    items_direct = dict( zip( edge_fracs, sizes[all_idx]/len(vertices) ) )
    items_random = dict( zip( edge_fracs, sizes2[all_idx]/len(vertices) ) )
    return { 'directed': items_direct, 'random': items_random }


def calculate_global_clustering(g):
    values, num_triangs, num_triples = global_clustering(g, weight=g._Graph__edge_properties['weight'], ret_counts=True)
    return { 'value': values[0], 'std': values[1], 
             'number_of_triangs': int(num_triangs), 'number_of_triples': int(num_triples) }

def calculate_avg_outdegrees(g):
    A = adjacency(g) 
    return { 'value': np.mean( np.sum(A, axis=1) ) }

def calculate_avg_indegrees(g):
    A = adjacency(g) 
    return { 'value': np.mean( np.sum(A, axis=0) ) }

# Running calculations

print('Running calculations..')
for f in range( args.start_idx, args.end_idx ):
    # global metrics in json format!
    savename = destination+'graph_global_attributes_undirected_'+files[f].split('.')[0][-8:]+'.json'
    if not os.path.exists( savename ):

        #read graph from csv file
        print(savename)
        print('Loading graph from file')
        day_csv = pd.read_csv( source+files[f] )
        G = create_day_graph_from_csv( path_to_file=source+files[f] )
        # convert to Graph-tool graph object
        gtG = nx2gt(G)
        # get random graph with configuration model
        print('Generating random graph with configuration model')
        gtG_rnd = copy.deepcopy(gtG)
        random_rewire( gtG_rnd, parallel_edges=True, self_loops=True )
        
        print('Calculating global graph properties')
        dict_to_dump = {} # every attribute will be written to this
        
        dict_to_dump['num_vertices_graph'] = calculate_num_vertices(gtG)
        dict_to_dump['num_vertices_config'] = calculate_num_vertices(gtG_rnd)
        
        dict_to_dump['num_edges_graph'] = calculate_num_edges(gtG)
        dict_to_dump['num_edges_config'] = calculate_num_edges(gtG_rnd)
        
        dict_to_dump['assortativity_graph'] = calculate_assortativity(gtG)
        dict_to_dump['assortativity_config'] = calculate_assortativity(gtG_rnd)
        
        dict_to_dump['scalar_assortativity_graph'] = calculate_scalar_assortativity(gtG)
        dict_to_dump['scalar_assortativity_config'] = calculate_scalar_assortativity(gtG_rnd)
        
        dict_to_dump['pseudo_diameter_graph'] = calculate_pseudo_diameter(gtG)
        dict_to_dump['pseudo_diameter_config'] = calculate_pseudo_diameter(gtG_rnd)
        
        dict_to_dump['min_spanning_tree_graph'] = calculate_min_spanning_tree(gtG)
        dict_to_dump['min_spanning_tree_config'] = calculate_min_spanning_tree(gtG_rnd)
        
        dict_to_dump['global_clustering_graph'] = calculate_global_clustering(gtG)
        dict_to_dump['global_clustering_config'] = calculate_global_clustering(gtG_rnd)
        
        dict_to_dump['vertex_percolation_graph'] = calculate_vertex_percolation(gtG)
        dict_to_dump['vertex_percolation_config'] = calculate_vertex_percolation(gtG_rnd)
        
        dict_to_dump['edge_percolation_graph'] = calculate_edge_percolation(gtG)
        dict_to_dump['edge_percolation_config'] = calculate_edge_percolation(gtG_rnd)

        dict_to_dump['avg_outdegrees_graph'] = calculate_avg_outdegrees(gtG)
        #dict_to_dump['avg_outdegrees_config'] = calculate_avg_outdegrees(gtG_rnd)

        dict_to_dump['avg_indegrees_graph'] = calculate_avg_indegrees(gtG)
        #dict_to_dump['avg_indegrees_config'] = calculate_avg_indegrees(gtG_rnd)

        with open(savename, 'w', encoding='utf-8') as f:
            json.dump( dict_to_dump, f, ensure_ascii=False, indent=4)
        
        #print(dict_to_dump)
    else:
        print('Already processed, skipping!')
