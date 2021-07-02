from pathlib import Path
import numpy as np
import pandas as pd
from src import utils
import sys
# looks for acs 2016 data in Data/acs_2016/acs_2016_features.csv
# looks for nfirs data in Data/NFIRS Fire Incident Data.csv
# pass in desired geolevel as argument

class genericDataSet:
    def  __init__(self, level = 'block_group'):

        #self.file_name= [] provide in subclasses  
        #self.tot_pop = []
        self.level = level
        self.LoadAndClean()

    def LoadAndClean(self):
        Data_path = self.file_name
        Data =  pd.read_csv(Data_path, dtype = {'GEOID':'object'},\
        index_col = 1)
        print(Data)
        self.CleanData(Data) 


    def MungeData(self,tot_pop,level='block'):


        Data = self.data 
        Data = Data.multiply(tot_pop['tot_population'],axis= 'index')
     
        Data.index , tot_pop.index  = Data.index.str[1:utils.GEOID[level]], \
                                  tot_pop.index.str[1:utils.GEOID[level]]

        Data, tot_pop = Data.groupby(Data.index).sum(), \
                   tot_pop.groupby(tot_pop.index).sum()

        self.data = Data.divide(tot_pop['tot_population'],axis = 'index')
        self.tot_pop = tot_pop

    
    def CleanData(self):
        pass


    
class ACSData(genericDataSet):
    def __init__(self,level):
        self.file_name = '../../Data/acs_2016/acs_2016_features.csv'
        super().__init__(level)


    def CleanData(self,ACS):

        ## Cleans and munges ACS data 
        #  'ACS' - ACS variable from LoadACS
        #  'self.level' - geography level to munge the data to
        #            levels can be found in utils.GEOID
        #            #Note: this can function can only aggregate data    

        # Ensures GEOID variable is in the correct format and sets it as the dataframe index
        ACS['GEOID'] = ACS['geoid'].str[2:]
        
        ACS.set_index(['GEOID'],inplace = True)

        # Removes extraneous features (i.e. non-numeric) in the dataframe
        if 'Unnamed: 0' in ACS.columns:
            ACS.drop('Unnamed: 0','columns',inplace= True)
        
        if 'NAME' in ACS.columns:
            ACS.drop('NAME','columns',inplace= True)
        
        if 'inc_pcincome' in ACS.columns:
            ACS.drop('inc_pcincome','columns',inplace= True)
        
        
        tot_pop = ACS[['tot_population']].groupby('GEOID').sum()
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
        
        ## ACS Munging
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

        # package 
        self.data = ACS
        
        # munge to appropriate level 

        if  self.level =='block_group':
            #ACS data already at block_group level
            self.tot_pop = tot_pop
        else:
            self.MungeData(tot_pop,self.level)
    





class NFIRSdata(genericDataSet):
    
    def __init__(self,level,tot_pop):
        self.file_name = '../../Data/NFIRS Fire Incident Data.csv'
        self.tot_pop = tot_pop
        self.level = level
        self.LoadAndClean()




    def LoadAndClean(self):
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

        self.CleanData(Data)


             


    def CleanData(self,nfirs):
        #NFIRS Munging

        #Convert inc_date column values to python datetime type
        nfirs['inc_date'] = pd.to_datetime(nfirs['inc_date'], infer_datetime_format=True)



        # Ensure correct calculation of tot_loss column 
        nfirs['tot_loss'] = nfirs['prop_loss'] + nfirs['cont_loss']

        # Create mask for new severe fire variable
        sev_fire_mask = (nfirs['oth_death'] > 0) | (nfirs['oth_inj'] > 0) | (nfirs['tot_loss'] >= 10000)

        # By default assigns values of severe fire column as not severe
        nfirs['severe_fire'] = 'not_sev_fire'

        # Applies filter to severe fire column to label the severe fire instances correctly
        nfirs.loc[sev_fire_mask,'severe_fire'] = 'sev_fire'

        # Create new NFIRS variables based on specified thresholds of existing variables in dataframe
        nfirs['had_inj'] = np.where(nfirs['oth_inj']>0,'had_inj','no_inj')
        nfirs['had_death'] = np.where(nfirs['oth_death']>0,'had_death','no_death')
        nfirs['10k_loss'] = np.where(nfirs['tot_loss']>=10000,'had_10k_loss','no_10k_loss')

        # Extract just the numeric portion of the geoid
        nfirs['geoid'] =  nfirs['geoid'].str.strip('#_')

        # Add a year column to be used to groupby in addition to geoid
        nfirs['year'] = nfirs['inc_date'].dt.year.astype('str')
        nfirs.set_index('geoid',inplace = True)


        # package 
        self.data = nfirs

        # munge to appropriate level 
        self.mungeData(self.tot_pop, self.level, self.data)
        # if  self.level =='block_group':
        #     #ACS data already at block_group level
        #     pass
        # else:
        #     self.MungeData(self.tot_pop,self.level)


    def mungeData(self,tot_pop,level, nfirs):

        GEOID = { 
            'state': 2,
            'county': 5,
            'tract' : 11,
            'block_group' : 12
            }

        l = GEOID[level]


        fires = pd.crosstab(nfirs.index, nfirs['year'])
        
        block_fires = fires.copy()
        fires.index = [f[0:l] for f in fires.index]
        fires.index.name = 'geoid'
        grouped_fires = fires.groupby(by='geoid').sum()

        block_tot_pop = tot_pop.copy()
        tot_pop.index = [p[0:l] for p in tot_pop.index]
        tot_pop.index.name = 'geoid'
        grouped_tot_pop = tot_pop.groupby(by='geoid').sum()

        final = grouped_fires.divide(grouped_tot_pop['tot_population'], axis='index')

        top10 = final > final.quantile(.9)

        self.final = final
        self.top10 = top10


if __name__ == "__main__":

    level = str(sys.argv[1])
    ACSLoader =  ACSData('block_group')
    ACS = ACSLoader.data


    tot_pop = ACSLoader.tot_pop
    NFIRSDataLoader = NFIRSdata(level,tot_pop)
    fires = NFIRSDataLoader.final
    top10 = NFIRSDataLoader.top10