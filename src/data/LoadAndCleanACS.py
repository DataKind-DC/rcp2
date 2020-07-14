# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

def LoadACS():

    p = Path.cwd()
    data_path = p.parent.parent / 'Data' / 'Master Project Data'   
    ACS_path = data_path  / 'ACS 5YR Block Group Data.csv'
    ACS = pd.read_csv(ACS_path, dtype = {'GEOID':'object'},
      index_col = 1)
    return ACS 


def CleanACS(ACS):
    # ACS Munging
    
    # Ensures GEOID variable is in the correct format and sets it as the dataframe index
    ACS['GEOID'] = ACS['GEOID'].str[2:]
    
    # Create ACS dataframes without NY or ACS dataframes with only NY to see how baseline model changes
    #ACS = ACS[ACS['GEOID'].str[0:2] != "36"]
    #ACS = ACS[ACS['GEOID'].str[0:2] == "36"]
    
    ACS.set_index(['GEOID'],inplace = True)

    # Removes extraneous features (i.e. non-numeric) in the dataframe
    if 'Unnamed: 0' in ACS.columns:
        ACS.drop('Unnamed: 0','columns',inplace= True)
    
    if 'NAME' in ACS.columns:
        ACS.drop('NAME','columns',inplace= True)
    
    if 'inc_pcincome' in ACS.columns:
        ACS.drop('inc_pcincome','columns',inplace= True)
    
    # Drop all total count columns in ACS and keeps all percentage columns
    cols = ACS.columns.to_list()
    for col in cols:
        if  col.find('tot') != -1 : 
            ACS.drop(col,'columns', inplace = True)
    
    
    # Integer indexing for all rows, but gets rid of county_name, state_name, and in_poverty
    ACS = ACS.iloc[:,3:]
    
    # Remove missing values from dataframe
    ACS.replace([np.inf, -np.inf], np.nan,inplace = True)
    ACS.dropna(inplace = True)
    return ACS



if __name__ == '__main__':
   
    ACS = LoadACS()
    ACS = CleanACS(ACS)
