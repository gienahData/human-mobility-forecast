# Created by A. Biricz 01.12.2020., then revisited, corrected, updated 03.04.2021.

## Arguments:
# --source_folder: where events and polygons data located
# --target_folder: save folder
# --source_pol_folder: load universal grid from here
# --start_idx: first file's index
# --end_idx: last file's index


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

args = parser.parse_args()

# Locate files
source = os.path.abspath( args.source_folder ) + '/' # '/mnt2/data/csv/' #'/media/Data_storage/Mobilcell/Data/'
source_pol = os.path.abspath( args.source_pol_folder ) + '/' #'/media/Data_storage/Mobilcell/DayPolygonData/'
target = os.path.abspath( args.target_folder ) + '/' # '/mnt/DayGraphData/'  #'/media/Data_storage/Mobilcell/DayGraphData/'

files = sorted([ i for i in os.listdir(source) if '.csv.gz' in i ])
files = np.array( sorted([ i for i in files if 'output_' in i ]) )

# Load universal, merged, fixed tower locations
tower_df = pd.read_csv( source_pol+'fixed_merged_tower_locations.csv' )
sort_idx = np.argsort( tower_df.tower_id.values )
tower_df = tower_df.iloc[ sort_idx ]
tower_df.reset_index(inplace=True)
coords = np.unique( tower_df.iloc[:,2:], axis=0 )[:,1:]

# Function definitions
#     
def create_day_graph_from_csv( path_to_file ):
    V = pd.read_csv( path_to_file, delimiter=',' ).values

    # create adjacency matrix and make graph undirected
    adj_mat = np.zeros( (coords.shape[0], coords.shape[0]), dtype=np.uint32 )
    adj_mat[ V[:,0], V[:,1] ] = V[:,2]
    adj_mat = adj_mat + adj_mat.T

    G = nx.from_numpy_matrix(adj_mat)
    print( "Graph nodes, edges: ", len( G.nodes() ), len( G.edges() ) )
    
    # adj mat is symm. -> completely zero rows and cols are at the same 1D indices
    zeros_idxs = np.arange( adj_mat.shape[0] )[ ~np.all( adj_mat == 0, axis=1 ) ]

    return G, zeros_idxs


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


def calculate_betweenness(g):
    vertex, edge = betweenness(g)
    return list(vertex)#, list(edge)


def calculate_closeness(g):
    vertex = closeness(g)
    return list(vertex)
    
    
def calculate_pagerank(g):
    vertex = pagerank(g)
    return list(vertex)


def calculate_eigenvector(g):
    largest_num, vertex = eigenvector(g)
    return list(vertex)


def calculate_katz(g):
    vertex = katz(g)
    return list(vertex)
    
    
def calculate_local_clustering(g):
    vertex = local_clustering(g)
    return list(vertex)
    
      
def calculate_motifs(g):
    mots, counts = motifs( g, k=3 )
    return mots, counts


for f in range( args.start_idx, args.end_idx ):
    # node_id, geospatial coordinates, graph metrics
    savename = target+'graph_vertex_attributes_undirected_'+files[f].split('.')[0][-8:]+'.csv'
    if not os.path.exists( savename ):
        
        print('Loading graph')
        # read graph from csv file
        day_csv = pd.read_csv( source+files[f] )
        G, zeros_idxs = create_day_graph_from_csv( path_to_file=source+files[f] )
        # convert to Graph-tool graph object
        gtG = nx2gt(G)
        
        print('Calculating local metrics')
        vertex_attrs = []
        vertex_attrs.append( calculate_betweenness(gtG) )
        vertex_attrs.append( calculate_closeness(gtG) )
        #vertex_attrs.append( calculate_pagerank(gtG) )
        vertex_attrs.append( calculate_eigenvector(gtG) )
        #vertex_attrs.append( calculate_katz(gtG) )
        vertex_attrs.append( calculate_local_clustering(gtG) )
        
        print('Saving results to disk')
        vertex_attrs = np.array(vertex_attrs).T
        vertex_attrs = np.concatenate( ( np.arange(coords.shape[0]).reshape(-1,1), 
                                         coords, vertex_attrs ), axis=1 )[zeros_idxs]
        vertex_df = pd.DataFrame( vertex_attrs, 
                                  columns=( ['node_id', 'eovx', 'eovy', 
                                             'betweenness', 'closeness', 
                                             'eigenvector', "local_clustering"] ) )
        vertex_df.to_csv( savename, index=None )
    else:
        print('Already processed, skipping!')