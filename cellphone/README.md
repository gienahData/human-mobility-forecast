# cellphone
Mobility analysis and forecast based on cellphone data. 

### Generate daily mobility graphs: 
Use `day_graph_generator.py` with proper arguments given. 

Standing objects and high speed movements are filtered. The nodes in the network are the towers (2G, 3G, 4G cells), the egdes are movements from towers to towers (cells to cells), their weights are integer numbers of registered moving object. The position of the towers are calculated from their area that is given by rasters. It is very likely that these areas overlap.

![Alt text](./polygon_overlap.png?raw=true "overlap of polygons")

### Investigate polygon grid and fix tower positions for the whole year:
Use `year_polygon_data.py`, `day_polygon_encodings.py` and `fix_polygon_grid.py` in this order with proper arguments given. The result is a file with this header:

original_id,tower_id,mean_x,mean_y,std_x,std_y,perc_10_x,perc_10_y,perc_50_x,perc_50_y,perc_90_x,perc_90_y,perc_99_x,perc_99_y
71892,0,250859,557285,4,2,250851,557281,250862,557287,250862,557287,250862,557287
72307,1,243343,558377,4,9,243342,558374,243342,558374,243345,558383,243355,558407
72922,2,247254,549716,5,2,247247,549714,247258,549716,247258,549724,247258,549724
73325,3,248646,551068,3,3,248646,551066,248646,551066,248652,551072,248652,551081

### Prepare data to construct mobility graph of short time intervals:
Use `time_interval_graph_preparator.py` with proper arguments given.

Generates data for creating short time (15 minutes) interval mobility graphs and interpolates longer events to match this timescale. Until now, linear interpolation is used and only the movements between 15-65 minutes are interpolated, the longer ones are omitted.

Due to the interpolation the coordinates of events will fall between the original towers which makes it hard (not trivial) to create a graph as input with nodes that are fixed globally (in reality, the towers are fixed to a position!). Reassigning these events to the towers could be a way to go, but for now let's stick to the following:

Consider a time scale of 30 minutes and omit the longer events (approx. 10-20% of all events), so interpolation is not needed. For this, use `time_interval_graph_preparator_30_mins_fix_grid.py` with proper arguments given.

An example plot for this graph (approx. 43k nodes, 673k edges):

![Alt text](./motion_graph_snaphot_30min.png?raw=true "30 min interval mobility graph")

### Short-time interval graph generation:
Use `time_interval_graph_generator.py` with proper arguments given. This will create output graphs with edgelist and node features in a format matching "Data handling of graphs" [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html).

### Graph analytics:
Use `day_graph_analytics_global.py` and `day_graph_analytics_local.py` with proper arguments given.

Calculate global and local graph metrics on the previously generated daily mobility graphs.


### Data exploration and manipulation notebooks:
- `Raw_data_investigation.ipynb`: loads raw polygon and event files and investigates them.

- `Raw_data_processing.ipynb`: generates daily graphs and creates plots of a particular date. This is the experimental version of the `day_graph_generator.py` script.

- `Raw_data_processing_time_invervals.ipynb`: gerenates short time interval mobility graphs and interpolates longer events to match this timescale. This is the experimental version of the `time_interval_graph_preparator.py` script.

- `Raw_data_processing_time_invervals_30_mins_fix_grid.ipynb`: gerenates short time (30 minutes) interval mobility graphs without interpolation. This is the experimental version of the `time_interval_graph_preparator_30_mins_fix_grid.py` script.

- `Raw_data_processing_time_invervals_30_mins_fix_grid_node_attrs.ipynb`: gerenates short time (30 minutes) interval node attributes of the mobility graphs without interpolation. This is the experimental version of the ...


- `Polygon_grid_save_yearly_data_to_disk.ipynb`: loads all raw polygon files and encodes their data for further processing. This is the experimental version of the `day_polygon_encodings.py` script.

- `Polygon_grid_exploration.ipynb`: loads all raw polygon files to investigate raster stability related to a tower throughout the year and calculates overlap matrices of towers to measure the precision of object localization. Results (globally fixed tower positions) are saved to disk at the end. This is the experimental version of the `year_polygon_data.py` and `fix_polygon_grid.py` scripts.

- `DayEvent_postprocessing.ipynb`: this is the experimental version of the `time_interval_graph_generator.py` script.

### To-Do:
- Deep learning: graph encoder, decoder, RNN for time predictions.
- Approximate the error of event localization and take it into consideration.
