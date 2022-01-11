from pathlib import Path
import numpy as np
import pandas as pd
from src import utils


class ACSData():
    # TODO: typechecking
    
    def __init__(self,year = 2016,level = 'block_group'):

        self.file_name = utils.DATA['acs'] / "acs_{}_data.csv".format(year)
        self.level = level
        self.data = None
        self.tot_pop = None
        self.Load()
        self.Clean(self.data)
        self.Munge(self.data,self.tot_pop, self.level)

    
    def Load(self):
        self.data = pd.read_csv(self.file_name, dtype = {'GEOID':'object'}, index_col = 1)

    def Clean(self,ACS):
    
        ## Cleans ACS data 
        #  'ACS' - ACS variable from LoadACS
        #  'self.level' - geography level to munge the data to
        #            levels can be found in utils.GEOID
        #            #Note: this can function can only aggregate data    

        # Ensures GEOID variable is in the correct format and sets it as the dataframe index
        ACS.reset_index(inplace = True)
        ACS['GEOID'] = ACS['geoid'].str[2:]
        
        ACS.set_index(['GEOID'],inplace = True)

        ACS.drop('geoid','columns',inplace =True)

        
        # Removes extraneous features (i.e. non-numeric) in the dataframe
        if 'Unnamed: 0' in ACS.columns:
            ACS.drop('Unnamed: 0','columns',inplace= True)
        
        if 'NAME' in ACS.columns:
            ACS.drop('NAME','columns',inplace= True)
        
        if 'inc_pcincome' in ACS.columns:
            ACS.drop('inc_pcincome','columns',inplace= True)
        

        
        self.tot_pop = ACS[['tot_population']].groupby('GEOID').sum()
        # Drop all total count columns in ACS and keeps all percentage columns
        cols = ACS.columns.to_list()
        for col in cols:
            if  col.find('tot') != -1 : 
                ACS.drop(col,'columns', inplace = True)
        
        

       
        # Remove missing values from dataframe
        ACS.replace([np.inf, -np.inf], np.nan,inplace = True)
        ACS.dropna(inplace = True)

        
        self.data = ACS

    
    def Munge(self,ACS,tot_pop,level='block_group'):

        ## ACS Munging
        
        ACS.drop(ACS.loc[:, 'state':'in_poverty'], inplace = True, axis = 1)
        
        #education adjustment 
        ACS['educ_less_12th'] =  ACS.loc[:,'educ_nursery_4th':'educ_12th_no_diploma'].sum(axis =1 )
        ACS['educ_high_school'] =  ACS.loc[:,'educ_high_school_grad':'educ_some_col_no_grad'].sum(axis =1 )
        ACS.drop(ACS.loc[:, 'educ_nursery_4th':'educ_some_col_no_grad'], inplace = True, axis = 1)

        # house age adjustment 
        ACS['house_yr_pct_before_1960'] =ACS.loc[:,'house_yr_pct_1950_1959':'house_yr_pct_earlier_1939'].sum(axis =1 )
        ACS['house_yr_pct_after_2000'] = ACS.loc[:, 'house_yr_pct_2014_plus':'house_yr_pct_2000_2009'].sum(axis = 1 )
        ACS['house_yr_pct_1960_2000'] = ACS.loc[:, 'house_yr_pct_1990_1999':'house_yr_pct_1960_1969'].sum(axis = 1 )
        ACS.drop(ACS.loc[:, 'house_yr_pct_2014_plus':'house_yr_pct_earlier_1939'], inplace = True, axis = 1)
        
        # housing Price adjustment
        ACS['house_val_less_50K']=ACS.loc[:,'house_val_less_10K':'house_val_40K_50K'].sum(axis =1 )
        ACS['house_val_50_100K']=ACS.loc[:,'house_val_50K_60K':'house_val_90K_100K'].sum(axis =1 )
        ACS['house_val_100K_300K']=ACS.loc[:,'house_val_100K_125K':'house_val_250K_300K'].sum(axis =1 )
        ACS['house_val_300K_500K']=ACS.loc[:,'house_val_300K_400K':'house_val_400K_500K'].sum(axis =1 )
        ACS['house_val_more_500K'] = ACS.loc[:,'house_val_500K_750K':'house_val_more_2M'].sum(axis = 1)
        ACS.drop(ACS.loc[:, 'house_val_less_10K':'house_val_more_2M'], inplace = True, axis = 1)
        
        ACS['race_pct_black_or_amind'] = ACS.loc[:,'race_pct_black'] \
                                       + ACS.loc[:,'race_pct_amind']

        ACS['pct_alt_heat'] =  ACS.loc[:,'heat_pct_fueloil_kerosene']  \
                       + ACS.loc[:,'heat_pct_coal']   \
                       + ACS.loc[:,'heat_pct_wood']   \
                       + ACS.loc[:,'heat_pct_bottled_tank_lpgas']

        
        
        
        self.data = ACS

        
        # munge to appropriate level 

        if  self.level =='block_group':
            #ACS data already at block_group level
            self.tot_pop = tot_pop
        else:
            Data = self.data 
            Data = Data.multiply(tot_pop['tot_population'],axis= 'index')
        
            Data.index , tot_pop.index  = Data.index.str[0:utils.GEOID[level]], \
                                    tot_pop.index.str[0:utils.GEOID[level]]

            Data, tot_pop = Data.groupby(Data.index).sum(), \
                    tot_pop.groupby(tot_pop.index).sum()

            self.data = Data.divide(tot_pop['tot_population'],axis = 'index')
            self.tot_pop = tot_pop


class NFIRSData():
    
    def __init__(self,level,tot_pop,sev=False, min_loss = 10000):
        self.file_name = utils.DATA['master'] /'NFIRS Fire Incident Data.csv'
        self.tot_pop = tot_pop
        self.level = level
        self.severeFiresOnly = sev
        self.data = None
        self.fires = None
        self.top10 = None
        self.severeFire = None
        self.min_loss = min_loss
        self.Load()
        # self.Clean(self.data)
        # munge to appropriate level 
        self.Munge(self.data, self.tot_pop,self.level, self.min_loss)

    def set_sev_loss(self, min_loss):
        self.min_loss = min_loss
        nfirs = self.data
        nfirs['severe_fire'] = 'not_sev_fire'
        sev_fire_mask = (nfirs['oth_death'] > 0) | (nfirs['oth_inj'] > 0) | (nfirs['tot_loss'] >= self.min_loss)
        nfirs.loc[sev_fire_mask,'severe_fire'] = 'sev_fire'
        nfirs['min_loss'] = np.where(nfirs['tot_loss']>=self.min_loss,'had_min_loss','no_min_loss')
        self.data = nfirs

        return



    def Load(self):
        cols_to_use = ['state','fdid','inc_date','oth_inj','oth_death','prop_loss',
               'cont_loss','tot_loss','geoid']

        # Specify particular data type for geoid column
        col_dtypes = {'geoid':str}

        # utils.DATA['master']  / self.file_name

        #Read in NFIRS dataframe
        Data_path =  self.file_name
        
        Data =  pd.read_csv(Data_path,
                    dtype = col_dtypes,
                    usecols = cols_to_use,
                    encoding='latin-1')
        self.data = Data
        
    

    def Munge(self, nfirs, tot_pop, level, min_loss):
        #NFIRS Munging

        #Convert inc_date column values to python datetime type
        nfirs['inc_date'] = pd.to_datetime(nfirs['inc_date'], infer_datetime_format=True)



        # Ensure correct calculation of tot_loss column 
        nfirs['tot_loss'] = nfirs['prop_loss'] + nfirs['cont_loss']

        # # Create mask for new severe fire variable
        # sev_fire_mask = (nfirs['oth_death'] > 0) | (nfirs['oth_inj'] > 0) | (nfirs['tot_loss'] >= 10000)

        # # By default assigns values of severe fire column as not severe
        # nfirs['severe_fire'] = 'not_sev_fire'

        # # Applies filter to severe fire column to label the severe fire instances correctly
        # nfirs.loc[sev_fire_mask,'severe_fire'] = 'sev_fire'

        self.set_sev_loss(min_loss)

        # Create new NFIRS variables based on specified thresholds of existing variables in dataframe
        nfirs['had_inj'] = np.where(nfirs['oth_inj']>0,'had_inj','no_inj')
        nfirs['had_death'] = np.where(nfirs['oth_death']>0,'had_death','no_death')
        

        # Extract just the numeric portion of the geoid
        nfirs['geoid'] =  nfirs['geoid'].str.strip('#_')

        # Add a year column to be used to groupby in addition to geoid
        nfirs['year'] = nfirs['inc_date'].dt.year.astype('str')
        nfirs.set_index('geoid',inplace = True)


        # package  
        self.data = nfirs
#-------------------------
        nfirs = self.data
        L = utils.GEOID[level]
        # shorten geoid to desired geography
        nfirs.index = nfirs.index.str[0:L]

       # subset to severe fires if requested 
        if self.severeFiresOnly:
            nfirs = nfirs[nfirs['severe_fire'] == 'sev_fire' ]


        # create a list of number of fires per year for each geography
        fires = pd.crosstab(nfirs.index, nfirs['year'])
        
        # Grab total population values pulled from ACS dataframe and assign to each census block in NFIRS dataframe
        fires = fires.merge(tot_pop, how = 'left', left_index = True, right_index = True)
        fires.index = fires.index.rename('geoid')


        # Remove resulting infinity values and zeros following merge 
        # note: We keep resulting NA's as NA's to show gaps in data collection 
        # use NA tolerant algo or change or add line to drop all rows with NAs
        fires.replace([np.inf, -np.inf,0], np.nan,inplace = True)

        # drop rows with low population count
        fires = fires[fires['tot_population'] >= 50 ] 

        # population adjustment to fires per_n_people 
        per_n_people = 1000
        min_year,max_year = nfirs['year'].min(), nfirs['year'].max()
        fires.loc[:,min_year:max_year] = fires.loc[:,min_year:max_year].div(fires['tot_population'], axis = 'index') * per_n_people

        # remove population
        fires.drop('tot_population',axis = 1, inplace = True)

        # find top decile in terms of number of adjusted fires each year
        top10 = fires > fires.quantile(.9)

        self.fires = fires
        self.top10 = top10
