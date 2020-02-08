##############################################################################
# fire_station_geolocator.R
#
# Red Cross Fire Alarm Phase 2, 2019
# S. Gianfortoni
#
# This script matches fire stations to their census block, and pulls census block information (including centroid coordinates).
#
# Inputs:
#     State-based TIGER/LINE shapefile, state Census Block 2017:
#           Pulled using the tigris package.
#
#     FD_locations.csv:
#        Address and long/lat of all fire stations in US
#

# Outputs:
#     FD_locations_blockgroups.csv: Fire station csv with census block geo_id included
#     Census_Blocks.csv: Csv of all census block groups and metadata
#############################################################################

library(data.table)
library(tigris)
library(rgdal)
library(here)
library(geosphere)
options(digits = 15)


fire_stations <- fread('FD_Locations.csv')


# create census block column

fire_stations$census_block <- NA
fire_stations$census_block <- as.character(fire_stations$census_block)

# create list of state abbreviations
state_list <- unique(fire_stations$STATE)

# create empty data.table for census blocks
block_information <- data.table()

# loop by state

for (state in state_list) {
    
    try(
    # get census block groups by state
        census_blocks <- block_groups(state = state)
    )
    
    # filter dataframe by state
    stations_filtered <- fire_stations[STATE == state]
    
    # set at shapefile
    coordinates(stations_filtered) <- ~ Longitude + Latitude
    
    # map spatial dataframes
    proj4string(stations_filtered) <- proj4string(census_blocks)
    
    # match spatial dataframes to blocks
    fire_blocks <- over(stations_filtered, census_blocks)
    
    # assign block group geoid
    fire_stations[STATE == state]$census_block <- fire_blocks$GEOID
    
    # save state ge
    block_information <- rbind(block_information, census_blocks@data)
}


# write.csv(fire_stations, 'FD_Locations_Blockgroups.csv', row.names = F)
# write.csv(block_information, 'census_blocks.csv', row.names = F)

