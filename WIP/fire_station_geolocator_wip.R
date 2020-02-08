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


####################################################################################################
fire_stations <- fread('FD_Locations_Blockgroups.csv')
block_information <- fread('census_blocks.csv')

####################################################################################################
# find closest fire station from each block group (as the crow flies) distance

# work in progress

min(block_information$INTPTLAT)
max(block_information$INTPTLAT)

min(block_information$INTPTLON)
max(block_information$INTPTLON)


min(fire_stations$Latitude)
max(fire_stations$Latitude)

min(fire_stations$Longitude)
max(fire_stations$Longitude)

# lat and long to numeric
block_information$INTPTLON <- as.numeric(block_information$INTPTLON)
block_information$INTPTLAT <- as.numeric(block_information$INTPTLAT)

# set buffer to limit number of stations searched. each unit is about 70 miles each way
lat_buffer <-  2
lon_buffer <- 2

# fire station ID
block_information$closest_station_ID <- NA_character_

for (i in 1:nrow(block_information)) {
    # filter fire station data to 140 miles each way
    close_stations <- fire_stations[Latitude < block_information[i, INTPTLAT] + lat_buffer & 
                          Latitude > block_information[i, INTPTLAT] - lat_buffer & 
                          Longitude < block_information[i, INTPTLON] + lon_buffer & 
                          Longitude > block_information[i,INTPTLON] - lon_buffer, ]
    
    # find closest station
    best_dist <- 100000
    best_id <- NA_character_
    
    
    #loop to match best distance
    for (j in 1:nrow(close_stations)) {
        dist <- distm(block_information[i, c('INTPTLON','INTPTLAT')],
                      close_stations[j, c('Longitude','Latitude')])
        
        if (is.na(dist)) next
        
        if (dist < best_dist) {
            best_dist <- dist
            best_id <- close_stations$"Unique ID"[j]
        }
    }
    block_information$closest_station_ID[i] <- best_id
    print(paste(i, best_id))
}

# turn into function

findClosestStation <- function(lon, lat) {
    close_stations <- fire_stations[Latitude < lat + lat_buffer & 
                                        Latitude > lat - lat_buffer & 
                                        Longitude < lon + lon_buffer & 
                                        Longitude > lon - lon_buffer, ]
    
    # find closest station
    best_dist <- 100000
    best_id <- NA_character_
    
    try(
    #loop to match best distance
        for (j in 1:nrow(close_stations)) {
            dist <- distm(c(lon, lat),
                          close_stations[j, c('Longitude','Latitude')])
            
            if (dist < best_dist) {
                best_dist <- dist
                best_id <- close_stations$"Unique ID"[j]
            }
        }
    )
    return(best_id)
}

# apply to block table

block_information$closest_station_ID <- mapply(findClosestStation,
                                              block_information$INTPTLON,
                                              block_information$INTPTLAT)
                                              
    
 # write.csv(block_information, 'census_blocks.csv', row.names = F)    
    