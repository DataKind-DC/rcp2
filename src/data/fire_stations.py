"""Data processing for the fire stations data.

Add details about:

- top level functions.
- resulting data & metadata.
- source data.
- processing overview.
- processing diagram.

"""
import geopandas
import pandas as pd
import shapely
from src import utils
from src.data import raw


# Path to processed fire station data.
PATH = utils.DATA["processed"] / "fire-stations.csv"


def process():
    """Process the raw fire stations data.
    
    Returns:
        pandas.DataFrame: Processed fire station data.
    """
    stations = raw.read_fire_stations()
    tracts = raw.read_shapefiles()[["GEOID10", "geometry"]]
    
    # Create points for each lon-lat pair.
    geometry = [] 
    for coord in zip(stations.Longitude, stations.Latitude):
        geometry.append(shapely.geometry.Point(coord))
    
    # Add points to convert fd to a geodataframe.
    stations = geopandas.GeoDataFrame(stations,
                                      crs="EPSG:4269",
                                      geometry=geometry)
    
    # Find the Tract geoid for each station.
    return (geopandas.sjoin(stations, tracts, op="within", how="left")
                     .drop("geometry", axis=1)
                     .pipe(lambda x: pd.DataFrame(x)))


if __name__ == "__main__":
    # Process the fire stations data.
    stations = process()
    
    # Write processed data to disk.
    stations.to_csv(PATH)