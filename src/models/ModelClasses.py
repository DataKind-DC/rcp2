
from pathlib import Path
from src import utils
from src.data import DataLoaders

import numpy as np
import pandas as pd

import xgboost as xgb 

import pickle



    


class ModelBaseClass:
    def __init__(self):
        pass
    def save(self):
        pass
    def load(self):
        pass


class FireRiskModels():
    def  __init__(self,Modeltype ='severity',Level = 'block_group',ACS_year = '2016',
    mode = 'Production'):

        try:
            assert( mode in ['Production','Development'],'mode must be either Production or Development' )
        except AssertionError as msg:
            print(msg)

        #self.file_name= [] provide in subclasses  
        #self.tot_pop = []
        self.level = Level
        self.type = Modeltype
        ACSdata = DataLoaders.ACSData(ACS_year, Level)
        self.ACS = ACSdata.data
        self.tot_pop = ACSdata.tot_pop 
        nfirs = DataLoaders.NFIRSData(Level, self.ACSdata.tot_pop)
        self.fires = nfirs.fires
        self.top10 = nfirs.top10
        self.mode = mode

    def train(self):
        pass
    def test(self):
        pass

    def predict(self):
        pass
    
    @staticmethod
    def resample_df(X,y,upsample=True,seed = SEED):
        from sklearn.utils import resample
        # check which of our two classes is overly represented 
        if np.mean(y) > .5:
            major,minor = 1,0
        else:
            major,minor = 0, 1
        
        # Add Class feature to dataframe equal to our existing dependent variable
        X['Class'] = y
        
        df_major = X[X.Class == major ]
        df_minor = X[X.Class == minor ]
        

        if upsample:      
        
            df_minor_resampled = resample(df_minor,
                                        replace = True,
                                        n_samples = df_major.shape[0], 
                                        random_state = seed)
        
        
    
            combined = pd.concat([df_major,df_minor_resampled])
            
            # Debug
            #print('minor class {}, major class {}'.format(df_minor_resampled.shape[0],
                                                        #df_major.shape[0]))
        
            
        else: # downsample
            
            df_major_resampled = resample(df_major,
                                        replace = False,
                                        n_samples = df_minor.shape[0],
                                        random_state = seed)
            
            
            combined = pd.concat([df_major_resampled,df_minor])
            
            #print('minor class {}, major class {}'.format(df_minor.shape[0],
                                                        #df_major_resampled.shape[0]))


        
        
        y_out = combined['Class']
        X_out = combined.drop('Class', axis =1)
        return X_out , y_out



class SmokeAlarmModels:
    def  __init__(self,ARC,ACS):
        self.arc = ARC
        self.acs = ACS
        self.models = {}
        
    def train(self):

        self.trainStatisticalModel()
        self.trainDLmodel()


    def trainStatisticalModel(self):



        # single level models 
        for geo in ['state','county','tract','block_group'] :
            df = self.createSingleLevelSmokeAlarmModel(geo) 
            name = 'SmokeAlarmModel' + geo + '.csv'
            df.index.name = 'geoid'
            df.index =  '#_' + df.index 
            out_path =  utils.DATA['model-outputs'] /'Smoke_Alarm_Single_Level' / name
            df.to_csv(out_path)
            self.models[geo] = df
        
        self.createMultiLevelSmokeAlarmModel()
        
    





    def trainDLModel(self):
        pass

    def CreateConfidenceIntervals(self,num_surveys,percentage):
        # this function takes the cleaned data and adds a confidence interval 

        z =	1.960 # corresponds to 95% confidence interval
    
        CI =  z * np.sqrt(
                     (percentage * (100 - percentage) ) / 
                      num_surveys  )

        return CI



    def createSingleLevelSmokeAlarmModel(self,geo_level):
# This function takes the arc data  into a dataset containing the percentage 
# and number of smoke detectors by census geography
#
# Inputs 
# arc-  the arc dataset
#
# geo_level- String var indcating what census geography to aggregate on. current levels are:
# State,County,Tract,Block_Group
#
# The resultant dataset will have the following values:
#
#   num_surveys - total number of surveys conducted
#
#   detectors_found -   houses with at least one smoke detector in the home
#
#   detectors_workding - houses with at least one tested and working smoke detector in the home
#
#   Note: for variables the suffixes 
#       _total- indicates raw counts 
#        _prc  - indicates percentage: (_total / num_surveys * 100)
#
        df = self.arc.copy()
        acs = self.acs.copy()
        # dict with relevant length of GEOID for tract geography
        geo_level_dict = {'State':2,'County':5,'Tract':11,'Block_Group':12}
        
        df['geoid'] = df['geoid'].str[: geo_level_dict[geo_level]]
        acs.index =  acs.index.str[:geo_level_dict[geo_level]]
        acs.drop_duplicates(inplace = True)
        ## binarize pre_existing_alarms and _tested_and_working
        #  values will now be: 0 if no detectors present and 1 if any number were present
        df['pre_existing_alarms'].where(df['pre_existing_alarms'] < 1, other = 1, inplace = True) 
        df['pre_existing_alarms_tested_and_working'].where(
                                                            df['pre_existing_alarms_tested_and_working'] < 1,
                                                                other = 1, 
                                                                inplace = True)

        ## create detectors dataset
        # This happens by grouping data both on pre_existing alarms and then _tested_and working alarms 
        # and then merging the two into the final dataset

        detectors =  df.groupby('geoid')['pre_existing_alarms'].agg({np.size ,
                                                                    np.sum,
                                                                    lambda x: np.sum(x)/np.size(x)* 100 })

        detectors.rename({'size':'num_surveys','sum':'detectors_found_total','<lambda_0>':'detectors_found_prc'},
                        axis =1,
                        inplace = True)

        detectors['detectors_found_prc'] = detectors['detectors_found_prc'].round(2)
        
    
        
        d2 =  df.groupby('geoid')['pre_existing_alarms_tested_and_working'].agg({np.size,np.sum, 
                                                                                lambda x: np.sum(x)/np.size(x)* 100 })
        
        d2.rename({'size':'num_surveys2','sum':'detectors_working_total','<lambda_0>':'detectors_working_prc'},
                        axis =1,
                        inplace = True)

        
        d2['detectors_working_prc'] = d2['detectors_working_prc'].round(2)
        

        detectors = detectors.merge(d2,how = 'left', on ='geoid')

        detectors['detectors_found_CI'] = self.CreateConfidenceIntervals(detectors['num_surveys'].values,
                                                                    detectors['detectors_found_prc'].values )
                                                                    
        detectors['detectors_working_CI'] = self.CreateConfidenceIntervals(detectors['num_surveys'].values,
                                                                    detectors['detectors_working_prc'].values )  
        
        
        
        
        
        # rearrange columns 
        column_order = ['num_surveys',	
                        'detectors_found_total',
                        'detectors_found_prc', 
                        'detectors_found_CI',
                        'detectors_working_total',
                        'detectors_working_prc',
                        'detectors_working_CI']
        
        detectors = detectors[column_order]
        
        detectors = detectors[~pd.isna(detectors.index)]
    # fix block model to ensure blocks that weren't visited are added to the model 
        detectors = detectors.reindex(detectors.index.union(acs.index.unique()),fill_value = 0)
        detectors = detectors[~pd.isna(detectors.index)]
    

    # test if there are missing values in resultant 

        return detectors   


    def createMultiLevelSmokeAlarmModel(self):

        if 'Block_Group' not in self.models.keys():
            self.createSingleLevelSmokeAlarmModel(self,'Block_Group') 

        block_data = self.models['Block_Group'].copy()
        all_IDS = block_data.index
        # subset block data to contain only data from Geographies with 30 or more surveyes
        block_data =  block_data[block_data['num_surveys'] >= 30]
        block_data['geography'] = 'block_group'
        block_data.index.name = 'geoid'
        # find which ids still need processing
        remaining_ids = all_IDS[~all_IDS.isin(block_data.index)]
        remaining_ids = remaining_ids.to_frame()
        remaining_ids = remaining_ids.rename({0:'geoid'},axis = 1)

        MultiLevelModel = block_data


        for  geo,geo_len,df in [
            ('tract', 11, self.models['Tract'].copy() ),
            ('county', 5, self.models['County'].copy() ),
            ('state', 2, self.models['State'].copy() )
            ]:

            # find all remaining ids that are not in the block data 
            
            geo_index = remaining_ids

            # set up data index 
            
            geo_index['temp_geoid'] = geo_index.index.str[:geo_len]
            geo_index = geo_index.set_index('geoid')
            
            # create data set at one level
            geo_data = geo_index.merge(df, how = 'left', right_index = True, left_on = 'temp_geoid')
            geo_data = geo_data[geo_data['num_surveys'] > 30] 
            geo_data = geo_data.drop('temp_geoid',axis = 1 )
            geo_data['geography'] = geo
            # add to multilevel index
            MultiLevelModel = MultiLevelModel.append(geo_data)
            
            # update remaining_ids
            
            remaining_ids = remaining_ids[~remaining_ids.index.isin(MultiLevelModel.index)]
            del geo_index, geo_data


        MultiLevelModel = MultiLevelModel.reset_index()
        MultiLevelModel['geoid'] = '#_' + MultiLevelModel['geoid']    
        self.models['MultiLevel'] = MultiLevelModel
        
        out_path =  utils.DATA['model-outputs'] / 'SmokeAlarmModelMultiLevel.csv'
        MultiLevelModel.to_csv(out_path)







    


    