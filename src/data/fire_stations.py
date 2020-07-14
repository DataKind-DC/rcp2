"""Data processing for the fire stations data.

The processed fire station data may be useful for analyzing the locations and
high-level attributes for fire stations across the US.

This module includes several top-level functions:

- :func:`src.data.fire_stations.process` processes raw fire station data.
- :func:`src.data.fire_stations.read` reads processed fire station data.

Run this module as a script to process fire station data from raw inputs and
save the processed data to the default data directory. ::

  $ python -m src.data.fire_stations

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

    This function reads raw fire station data and appends column *GEOID10*,
    which identifies the 2010 Census tract that each station belongs to. The
    calculation occurs in three steps.

    1. Identify fire station locations from latitude and longitude.
    2. Identify 2010 Census tract boundaries from raw shape files.
    3. Perform a spatial left-join, matching stations to surrounding tracts.

    The resulting pandas.DataFrame has the same row count as the raw data.

    Returns:
        pandas.DataFrame: Processed fire station data.
    """
    stations = raw.read_fire_stations()
    tracts = raw.read_shapefiles()[["GEOID10", "geometry"]]
    
    # Create points for the lon-lat pair at each fire station.
    geometry = calculate_points(stations.Longitude,
                                stations.Latitude,
                                crs=tracts.crs)
    
    # Append the points to the fire station data.
    stations = geopandas.GeoDataFrame(stations, geometry=geometry)

    # Find the Tract geoid for each station.
    result = geopandas.sjoin(stations, tracts, op="within", how="left")

    # Return a pandas.DataFrame without geometries.
    return pd.DataFrame(result).drop("geometry", axis=1)


def read(process_if_needed=False):
    """Read processed fire stations data.

    If `process_if_needed` is True and the processed fire station data are not
    found on disk, then this function will process the fire stations data from
    raw data and return the result.

    Args:
        process_if_needed (bool): Process the data if data not found.

    Returns:
        pandas.DataFrame: Processed fire station data.
    """
    dtype = {"GEOID10": str}
    if process_if_needed:
        try:
            result = pd.read_csv(PATH, dtype=dtype)
        except FileNotFoundError:
            result = process()
    else:
        result = pd.read_csv(PATH, dtype=dtype)
    return result


def calculate_points(long, lat, *args, **kwargs):
    """Calculate points in space from longitude and latitude.

    Args:
        long (array-like): Longitude values.
        lat (array-like): Latitude values.
        args: Positial arguments passed to geopandas.GeoSeries.
        kwargs: Keyword arguments passed to geopandas.GeoSeries.

    Returns:
        geopandas.GeoSeries: Points for the coordinates given.
    """
    geometry = []
    for coord in zip(long, lat):
        geometry.append(shapely.geometry.Point(coord))
    return geopandas.GeoSeries(geometry, *args, **kwargs)


if __name__ == "__main__":
    # Process the fire stations data.
    stations = process()
    
    # Write processed data to disk.
    stations.to_csv(PATH)