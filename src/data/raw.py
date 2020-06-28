"""Utilities for working with raw data."""
import geopandas
import pandas as pd
from src import utils


# Map state names to state FIPS codes.
STATES = {
    'AL': '01',
    'AK': '02',
    'AZ': '04',
    'AR': '05',
    'CA': '06',
    'CO': '08',
    'CT': '09',
    'DE': '10',
    'FL': '12',
    'GA': '13',
    'HI': '15',
    'ID': '16',
    'IL': '17',
    'IN': '18',
    'IA': '19',
    'KS': '20',
    'KY': '21',
    'LA': '22',
    'ME': '23',
    'MD': '24',
    'MA': '25',
    'MI': '26',
    'MN': '27',
    'MS': '28',
    'MO': '29',
    'MT': '30',
    'NE': '31',
    'NV': '32',
    'NH': '33',
    'NJ': '34',
    'NM': '35',
    'NY': '36',
    'NC': '37',
    'ND': '38',
    'OH': '39',
    'OK': '40',
    'OR': '41',
    'PA': '42',
    'RI': '44',
    'SC': '45',
    'SD': '46',
    'TN': '47',
    'TX': '48',
    'UT': '49',
    'VT': '50',
    'VA': '51',
    'WA': '53',
    'WV': '54',
    'WI': '55',
    'WY': '56',
    'AS': '60',
    'GU': '66',
    'MP': '69',
    'PR': '72',
    'VI': '78',
    'DC': '11',
}


class BadPathError(Exception):
    """An error for invalid paths."""
    pass


def read_shapefiles(states=None, fips=None, glob=None):
    """Read raw 2010 Census tract shapefiles.
    
    - Each shapefile has tracts for a state.
    - From the 2010 Census.
    - Select only one filter.
    
    Args:
        states (list): Two-letter state abbreviation strings.
        fips (list): Two-digit state FIPS code strings.
        glob (str): A glob expression (e.g., "*.shp").
        
    Returns:
        geopandas.DataFrame: Shapes for the states of interest.
    """
    # Directory with raw shapefiles.
    datadir = utils.DATA["shapefiles-census"]
    
    # Template for shapefile names.
    fname = "tl_2010_{code}_tract10.shp"
    
    # Count the nmber of filter arguments.
    n_filters = sum([arg != None for arg in (states, fips, glob)])

    # Handle calls with too many arguments.
    if n_filters > 1:
        raise ValueError("Only one of [states/fips/glob] may be selected.")
    
    # Identify the paths to read from, defaulting to all shapefiles.
    elif n_filters == 0:
        paths = sorted(datadir.glob("*.shp"))
        
    elif fips:
        paths = [datadir / fname.format(code=code) for code in fips]
        
    elif states:
        paths = []
        for state in states:
            code = STATES[state]
            paths.append(datadir / fname.format(code=code))
            
    elif glob:
        paths = sorted(datadir.glob(glob))

    # Handle bad path names and non-shapefiles. 
    for path in paths:
        if not path.exists():
            raise BadPathError(f"File {path} not found")
        if not path.suffix == ".shp":
            raise BadPathError(f"File {path} doesn't have extension '.shp'")
        
    # Read the data.
    chunks = []
    for path in paths:
        chunks.append(geopandas.read_file(path))
    return pd.concat(chunks)    


def read_fire_stations():
    """Read the raw fire station data."""
    path = utils.DATA["master"] / "Fire Station Location Data.csv"
    return pd.read_csv(path)
