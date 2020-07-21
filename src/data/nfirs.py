import pandas as pd
import numpy as np
import os
from src import utils

def ingest_raw_nfirs_data(data_dir):
    """Ingest single year of raw nfirs data, perform basic cleaning, merging, and filtering to 
    generate one years worth of nfirs data ready to be geocoded.
    
    Args:
        data_dir: nfirs directory with one year of data
        
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
    
    mask1 = basic['inc_type'].isin(inc_type_vals)
    mask2 = basic['prop_use'].str.startswith('4')

    basic = basic[mask1 & mask2]
    
    # Left join the address and fire tables to the basic table
    df = (basic.merge(address, how = 'inner', on = merge_cols)
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
    
    # Replace erroneous zip codes with null values
    erroneous_zip_codes = ['00000','11111','22222','99999']
    df['zip5'] = df['zip5'].replace(erroneous_zip_codes, np.nan)
    
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
    
    # Convert to str and zero pad inc_no and exp_no
    df['inc_no'] = df['inc_no'].str.zfill(7)
    df['exp_no'] = df['exp_no'].astype(str).str.zfill(3)

    # Create id column which is just combination of the primary key fields from nfirs
    df['unique_id'] = df['state'] + '_' + df['fdid'] + '_' + df['inc_date'].astype(str) + '_' + df['inc_no'] + '_' + df['exp_no']
    
    # Subset the data by the columns we've selected for further use
    usecols = ['state','fdid','st_fdid','dept_sta','inc_date','inc_no','exp_no','inc_type','prop_use','address','city','state_id',
              'zip5','oth_inj','oth_death','prop_loss','cont_loss','tot_loss','detector','det_type','det_power',
              'det_operat','det_effect','det_fail','aes_pres','aes_type','aes_oper','no_spr_op','aes_fail','unique_id']
    
    df = df[usecols]
    
    return(df)

def ingest_all_nfirs():
    
    """This function calls the ingest_raw_nfirs_data function on each
    directory within data/raw/nfirs, and saves the cleaned output files to
    data/interim/nfirs, ready to be geocoded.
    """
    
    raw_nfirs_path = utils.DATA["raw"] / 'nfirs'
    interim_nfirs_path = utils.DATA['interim'] / 'nfirs'

    for year in os.listdir(raw_nfirs_path):
        year_path = os.path.join(raw_nfirs_path, year)

        if not os.path.isdir(year_path):
            continue

        output_name = f'nfirs_cleaned_{year}.csv'
        output_path = os.path.join(interim_nfirs_path, output_name)

        df = ingest_raw_nfirs_data(year_path)
        df.to_csv(output_path, index = False)
    
    return

if __name__ == "__main__":
    ingest_all_nfirs()
