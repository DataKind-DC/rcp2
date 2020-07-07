"""Utilities for working with raw data.

The following top-level functions can be useful for reading raw data.

- :func:`src.data.raw.read_shapefiles` reads 2010 Census tract shapefiles.
- :func:`src.data.raw.read_fire_stations` reads fire station data.

"""
import geopandas
import pandas as pd
import numpy as np
from src import utils
import os


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

    The project raw data includes 56 shapefiles with 2010 Census tract
    polygons. Each file corresponds to a US state, territory, etc.

    This function facilitates reading and stacking any number of these
    shapefiles into a single geopandas.GeoDataFrame.

    The caller can specify the shapefiles to use with a list of two-letter
    state abbreviations, a list of two-digit FIPS codes, or a glob for the file
    names of interest. The caller can only use one of these filters in a call.

    Args:
        states (list): Two-letter state abbreviation strings.
        fips (list): Two-digit state FIPS code strings.
        glob (str): A glob expression (e.g., "*.shp").
        
    Returns:
        geopandas.GeoDataFrame: Shapes for the states of interest.
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
    """Read raw fire station data.

    Returns:
        pandas.DataFrame: The fire station data.
    """
    path = utils.DATA["master"] / "Fire Station Location Data.csv"
    return pd.read_csv(path)

def ingest_raw_nfirs_data(data_dir, output_dir, year):
    """Ingest single year of raw nfirs data, perform basic cleaning, merging, and filtering to 
    generate one years worth of nfirs data ready to be geocoded.
    
    Args:
        data_dir: nfirs directory with one year of data
        output_dir: output directory filepath
        year: year corresponding to nfirs data
        
    Returns:
        pandas dataframe of cleaned nfirs data (not geocoded yet)
    """
    
    # Read tables and switch columns to lower case
    basic = pd.read_csv(os.path.join(data_dir, 'basicincident.txt'), sep = '^', encoding = 'latin-1', low_memory = False)
    address = pd.read_csv(os.path.join(data_dir, 'incidentaddress.txt'), sep = '^', encoding = 'latin-1', low_memory = False)
    fire = pd.read_csv(os.path.join(data_dir, 'fireincident.txt'), sep = '^', encoding = 'latin-1', low_memory = False)
    
    basic.columns = basic.columns.str.lower()
    address.columns = address.columns.str.lower()
    fire.columns = fire.columns.str.lower()
    
    # Columns to merge the 3 datasets on
    merge_cols = ['state','fdid','inc_date','inc_no','exp_no']
    
    # Drop duplicates based on those merge columns. For nfirs 2016, there were 110 duplicates
    # dropped from the basic table, 65 from the address table, and 5 from the fire table. 
    basic = basic.drop_duplicates(merge_cols)
    address = address.drop_duplicates(merge_cols)
    fire = fire.drop_duplicates(merge_cols)
    
    # Subset the basic data by inc_type and prop_use values which correspond to home fires
    inc_type_vals = [111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
    prop_use_vals = ['419','429']
    
    mask1 = basic['inc_type'].isin(inc_type_vals)
    mask2 = basic['prop_use'].isin(prop_use_vals)

    basic = basic[mask1 & mask2]
    
    # Left join the address and fire tables to the basic table
    df = (basic.merge(address, how = 'left', on = merge_cols, indicator = 'address_merge')
      .merge(fire, how = 'left', on = merge_cols, indicator = 'fire_merge')
     )
    
    # Convert the address to a datetime object
    df['inc_date'] = pd.to_datetime(df['inc_date'].astype(str).str.zfill(8), format = '%m%d%Y')
        
    ### Combine the street address parts into a single address field
    # Clean address parts
    address_parts = ['num_mile','street_pre','streetname','streettype','streetsuf']
    for part in address_parts:
        df[part] = df[part].fillna('').astype(str).str.upper()

    # Some streetnames included the street_pre as part of the field (i.e. N N 21st st, or E E John Blvd). This
    # line replaces street_pre with '' if that is the case
    df['street_pre'] = np.where(df['street_pre'] == df['streetname'].str.split(' ').str[0], '', df['street_pre'])

    # Combines and cleans the address parts into a single address field
    df['address'] = df['num_mile'] + ' ' + df['street_pre'] + ' ' + df['streetname'] + ' ' + df['streettype'] + ' ' +\
                    df['streetsuf']
    df['address'] = df['address'].str.replace('\s+',' ', regex=True).str.strip()
    
    # Subset the data by only those records which have a zip code
    df = df[df['zip5'].notnull()]
    
    # Fill null values for state (which corresponds to the state the fire department is in) with the state_id (which corresponds
    # to the state where the fire occurred. 99% of the time these are the same). Do the same for state_id using state.
    # In 2016 there were 19 null values for state, and 4 for state_id
    df['state_id'] = df['state_id'].fillna(df['state'])
    df['state'] = df['state'].fillna(df['state_id'])
    
    # Fill null values for oth_inj and oth_death with 0. Assumption is that if there were really an injury or death, these 
    # fields would have been filled out. 
    df['oth_inj'] = df['oth_inj'].fillna(0)
    df['oth_death'] = df['oth_death'].fillna(0)
    
    # Fill null values for prop_loss and cont_loss with 0. Assumption is that if there were really a large property 
    # loss or content loss then these fields would have been filled out. 
    df['prop_loss'] = df['prop_loss'].fillna(0)
    df['cont_loss'] = df['cont_loss'].fillna(0)
    
    # Calculate the total loss column
    df['tot_loss'] = df['prop_loss'] + df['cont_loss']
    
    # Convert fdid column to str, and left pad with zeros to match documentation
    df['fdid'] = df['fdid'].astype(str).str.zfill(5)

    # Create st_fdid column with unique identifier for each fire department in the country
    df['st_fdid'] = df['state'] + '_' + df['fdid']
    
    # Zero pad dept_sta column to align with documentation
    df['dept_sta'] = (df['dept_sta'].astype(str)
                      .str.zfill(3)
                      .replace('nan',np.nan))
    
    # Capitalize cities
    df['city'] = df['city'].astype(str).str.upper()
    
    # Subset the data by the columns we've selected for further use
    usecols = ['state','fdid','st_fdid','dept_sta','inc_date','inc_no','exp_no','inc_type','prop_use','address','city','state_id',
              'zip5','oth_inj','oth_death','prop_loss','cont_loss','tot_loss','detector','det_type','det_power',
              'det_operat','det_effect','det_fail','aes_pres','aes_type','aes_oper','no_spr_op','aes_fail']
    
    df = df[usecols]
    
    return(df)