from pathlib import Path
import numpy as np
import pandas as pd
from src import utils


class ARCPData():
    # american red cross preparedness data 
    def __init__(self, ACS, file_name = 'ARC Preparedness Data.csv', level = 'block_group'):
        self.data = None
        self.file_name = utils.DATA['master'] / file_name 
        self.Load()
        self.standardizeColumnNames(ACS, level)

    def Load(self):

        self.data = pd.read_csv(self.file_name)

    def standardizeColumnNames(self, ACS, level):
        """
        Standardizes column names
        """

        df = self.data
        df.columns = map(str.lower, df.columns)
        df.columns = df.columns.str.replace(', ', '_')
        df.columns = df.columns.str.replace('-', '_')
        df.columns = df.columns.str.replace('/', '_')
        df.columns = df.columns.str.replace('(', '_')
        df.columns = df.columns.str.replace(')', '_')
        df.columns = df.columns.str.replace(' ', '_')
        df.dropna(inplace = True)
        # trim geoid leading saftey marks
        if level == 'block_group':
            df['geoid'] = df['geoid'].str[2:]
        elif level == 'tract':
            df['geoid'] = df['geoid'].str[2:-1]
        elif level == 'county':
            df['geoid'] = df['geoid'].str[2:-7]
        else:
            print('invalid level')

        df = df[df['geoid'].isin(ACS.tot_pop.index)]

        self.data = df   
    

class ACSData():
    # TODO: typechecking
    
    def __init__(self,year = 2016,level = 'block_group', pop_thresh = 0):

        self.file_name = utils.DATA['acs'] / "acs_{}_data.csv".format(year)
        self.level = level
        self.data = None
        self.tot_pop = None
        self.tot_pop_bg = None
        self.pop_thresh = pop_thresh
        self.Load()
        self.Clean(self.data)
        self.Munge(self.data,self.tot_pop, self.pop_thresh, self.level)

    
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
        
        #if 'inc_pcincome' in ACS.columns:
        #    ACS.drop('inc_pcincome','columns',inplace= True)
        

        
        self.tot_pop = ACS[['tot_population']].groupby('GEOID').sum()
        # Drop all total count columns in ACS and keeps all percentage columns
        #cols = ACS.columns.to_list()
        #print(cols)
        #for col in cols:
        #    if  col.find('tot') != -1 : 
        #        print(col)
        #        ACS.drop(col,'columns', inplace = True)
        
        

       
        # Remove missing values from dataframe
        ACS.replace([np.inf, -np.inf], np.nan,inplace = True)
        #ACS.dropna(inplace = True)

        
        self.data = ACS

    
    def Munge(self,ACS,tot_pop, pop_thresh,level='block_group'):

        ## ACS Munging
        
        #ACS.drop(ACS.loc[:, 'state':'in_poverty'], inplace = True, axis = 1)
        #print(ACS.columns)
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

        
        
        #print(ACS.columns)
        self.data = ACS

        
        # munge to appropriate level 
        self.tot_pop_bg = tot_pop
        if  self.level =='block_group':
            #ACS data already at block_group level
            self.tot_pop = tot_pop
        else:
            Data = self.data 
            operations = pd.read_csv(utils.DATA['data']/'geo_levels.csv', skiprows = 1)
            Data_level = Data.copy()
            Data_original = Data.copy()
            for i in Data_level.columns:
                a = operations.loc[operations['variable_name'] == i]
                op = a['operator'].values
                if (op == 'pct') | (op =='div'):
                    Data_level[i] = Data_original[i] * Data_original[a['argument1'].item()]
            Data_level.index = Data_level.index.str[0:utils.GEOID[level]]

            Data_level_num = Data_level.groupby(Data_level.index).sum()
            Data_level_str = Data_level.groupby(Data_level.index).first()
            for i in Data_level_num.columns:
                a = operations.loc[operations['variable_name'] == i]
                op = a['operator'].values
                if (op == 'pct') | (op =='div'):
                    Data_level_num[i] = Data_level_num[i].divide(Data_level_num[a['argument1'].item()],axis = 'index')
            Data_level = Data_level_num
            Data_level['state'] = Data_level_str['state']
            cols = Data_level.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            Data_level = Data_level[cols]
            self.data = Data_level
            self.tot_pop = Data_level[['tot_population']].groupby('GEOID').sum()
        #only get geoids with population greater than user defined value
        self.tot_pop = self.tot_pop[self.tot_pop['tot_population']>=self.pop_thresh]
        self.data = self.data[self.data.index.isin(self.tot_pop.index)]

class SVIData():
    # TODO: typechecking
    # level and year are fixe
    def __init__(self,ACS, tot_pop_bg, level = 'block_group'):

        self.file_name = utils.DATA['svi'] / "SVI Tract Data.csv"
        self.data = None
        self.level = level
        self.tot_pop_bg = tot_pop_bg
        self.Load()
        self.Clean(ACS, level)
        
    def Load(self):
        self.data = pd.read_csv(self.file_name, encoding='ISO-8859-1')
        self.data['Tract'] = self.data['GEOID'].str[2:]
    
    def Clean(self, ACS, level):
        if level == 'county':
            tract_pop = self.tot_pop_bg
            tract_pop.index = tract_pop.index.str[0:utils.GEOID['tract']]
            tract_pop = tract_pop.groupby(tract_pop.index).sum()
            data = self.data
            data.index = data['Tract']
            data = data[['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']]
            data = tract_pop.merge(data, how = 'left', left_index = True, right_index = True)
            print(data)
            print(data.isnull().values.sum())
            data[['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']] = data[['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']].multiply(data['tot_population'], axis='index')
            data.drop('tot_population',axis = 1, inplace = True)
            print(data)
            data.index = data.index.str[0:utils.GEOID['county']]
            data = data.groupby(data.index).sum()
            data = ACS[['tot_population']].merge(data, how = 'left', left_index = True, right_index = True)
            data[['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']] = data[['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']].divide(data['tot_population'], axis='index')
            data['inc_pct_poverty'] = ACS['inc_pct_poverty']
            data.drop('tot_population',axis = 1, inplace = True)
            self.data = data
        else:
            if level == 'block_group':
                ACS['Tract'] = ACS.index.str[:-1]
            elif level == 'tract':
                ACS['Tract'] = ACS.index
            else:
                print('invalid level')  
            ACS['geos'] = ACS.index
            merged = ACS.merge(self.data, how = 'left', left_on = 'Tract' , right_on ='Tract')
            merged.set_index('geos', inplace=True)
            cols = ['inc_pct_poverty','RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3','RPL_THEME4']
            self.data = merged[cols]
                    

        self.data = self.data.replace(-999,np.nan)
            
  
class NFIRSData():
    
    def __init__(self,level,tot_pop,pop_thresh = 0, sev=False, min_loss = 10000):
        self.file_name = utils.DATA['master'] /'NFIRS Fire Incident Data.csv'
        self.tot_pop = tot_pop
        self.level = level
        self.severeFiresOnly = sev
        self.pop_thresh = pop_thresh
        self.data = None
        self.fires = None
        self.top10 = None
        self.severeFire = None
        self.min_loss = min_loss
        self.Load()
        # self.Clean(self.data)
        # munge to appropriate level 
        self.Munge(self.data, self.tot_pop,self.level, self.min_loss, self.pop_thresh)

    def set_sev_loss(self, min_loss):
        self.min_loss = min_loss
        nfirs = self.data
        nfirs['severe_fire'] = 'not_sev_fire'
        sev_fire_mask = (nfirs['oth_death'] > 0) | (nfirs['oth_inj'] > 0) | (nfirs['tot_loss'] >= self.min_loss) | (nfirs['tot_units_affected'] > 1)
        nfirs.loc[sev_fire_mask,'severe_fire'] = 'sev_fire'
        nfirs['min_loss'] = np.where(nfirs['tot_loss']>=self.min_loss,'had_min_loss','no_min_loss')
        self.data = nfirs

        return



    def Load(self):
        cols_to_use = ['state','fdid','inc_date','oth_inj','oth_death','prop_loss',
               'cont_loss','tot_loss','tot_units_affected','geoid']

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
        
    

    def Munge(self, nfirs, tot_pop, level, min_loss, pop_thresh):
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
            nfirs_geos = nfirs.index.unique()
            nfirs_sev = nfirs[nfirs['severe_fire'] == 'sev_fire' ]

            fires = pd.crosstab(nfirs_sev.index, nfirs_sev['year'])
            # ensure no geographies were lost in restriction
            missing_geos = nfirs_geos.difference(fires.index)
            fires = fires.reindex(fires.index.append(missing_geos ) )

        else:
            # create a list of number of fires per year for each geography
            fires = pd.crosstab(nfirs.index, nfirs['year'])
        
        # Grab total population values pulled from ACS dataframe and assign to each census block in NFIRS dataframe
        #fires = fires.merge(tot_pop, how = 'left', left_index = True, right_index = True)
        #change order to keep ACS geoids
        fires = tot_pop.merge(fires, how = 'left', left_index = True, right_index = True)
        #fires = tot_pop.merge(fires, how = 'right', left_index = True, right_index = True)
        fires.index = fires.index.rename('geoid')


        # Remove resulting infinity values and zeros following merge 
        # note: We keep resulting NA's as NA's to show gaps in data collection 
        # use NA tolerant algo or change or add line to drop all rows with NAs
        fires.replace([np.inf, -np.inf,0], np.nan,inplace = True)

        # drop rows with low population count
        fires = fires[fires['tot_population'] >= pop_thresh ] 

        # population adjustment to fires per_n_people 
        per_n_people = 1000
        min_year,max_year = nfirs['year'].min(), nfirs['year'].max()
        fires_noAdjustment = fires.copy()
        fires.loc[:,min_year:max_year] = fires.loc[:,min_year:max_year].div(fires['tot_population'], axis = 'index') * per_n_people
        
        # remove population
        fires.drop('tot_population',axis = 1, inplace = True)

        # find top decile in terms of number of adjusted fires each year
        top10 = fires > fires.quantile(.9)

        self.fires = fires
        self.fires_noAdjustment = fires_noAdjustment
        self.top10 = top10
