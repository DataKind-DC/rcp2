
from pathlib import Path
from src import utils
from src.data import DataLoaders

import numpy as np
import pandas as pd

from xgboost import XGBClassifier 

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


import pickle

import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix,mean_squared_error,mean_absolute_error,roc_auc_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
    


class ModelBaseClass:
    def __init__(self):
        pass
    def save(self):
        pass
    def load(self):
        pass


class FireRiskModels():
    def  __init__(self,Modeltype ='propensity',ACS_year = '2016'):

        self.SEED = 0 
        self.type = Modeltype
        

    def train(self,NFIRS,ACS,ACS_variables =None, test_year='Current',n_years = 5):  
    # Create framework to predict whether a given census block has a fire risk score in the 90th percentile 
            # based on the specific number of previous years' data
        
        if not ACS_variables:
            ACS_variables = ACS.data.columns

        self.ACS_variables_used = ACS_variables
        ACS_data = ACS.data[ACS_variables]
        fires = NFIRS.fires
        top10 = NFIRS.top10
        years = top10.columns
        Model_Input = None
        Model_Predictions = None
        Model_Prediction_Probs = None
        Model_Actual = None
        
        # get year index 
        if test_year == 'Current': 
            test_year = fires.columns[-1]
        
        if test_year in fires.columns:
            test_year_idx = fires.columns.get_loc(test_year)
        else:
            raise ValueError(f"{test_year} is not in NFIRS Data." 
                             f" The most recent year in NFIRS is  {fires.columns[-1]}")

     
                
        # each model will train on `n_years` of data to predict the locations subsequent year with highest fire risk 
        # model will train predicting 1 year and then test model accuracy by predicting the next year 
        
        # for example: 
        
        #  Train 
        # predict 2016 using 2015-2012 data 
        #
        #
        #  Test
        # predict 2017 using 2016-2015 data  
       
        X_train, y_train,_ = self.munge_dataset(top10,fires,ACS_data,n_years,test_year_idx-1)
        self.X_train = X_train
        
        X_test, y_test,Input = self.munge_dataset_test(top10,fires,ACS_data,n_years,test_year_idx)
        print(len(X_test))
        print(len(y_test))
        
        # Note: `Input` is used for manual data validation to ensure munging performed correctly 
        

        # Create 80/20 training/testing set split
        #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .2 )

        # Perform resampling if data classes are unbalanced
        X_train, y_train = self.resample_df(X_train,y_train,upsample = False)



        # Standardize features by removing the mean and scaling to unit variance

        # scaler = preprocessing.StandardScaler().fit(X_train)
        # X_train= scaler.transform(X_train)
        # X_test = scaler.transform(X_test)


        # Fit model to training set

        print('Predicting {}:'.format(str(test_year)) )
        
        model = XGBClassifier( n_estimators=60,
                    max_depth=10,
                    random_state=0,
                    max_features = None,
                    n_jobs = -1, 
                    seed = self.SEED )
        
        model = model.fit(X_train,y_train)
      

        # Calculate training set performance

        #train_prediction_probs = model.predict_proba(X_train)
        #train_predictions = model.predict(X_train)
        #print (confusion_matrix(y_train, train_predictions))
        #print (roc_auc_score(y_train, train_prediction_probs[:,1]))


        # Calculate test set performance

        self.test_prediction_probs = model.predict_proba(X_test)
        self.test_predictions = model.predict(X_test)
        #Model_Predictions = pd.Series(test_predictions)
        #Model_Prediction_Probs = pd.Series(test_prediction_probs[:,[1]].flatten())
        #print (confusion_matrix(y_test, self.test_predictions))
        #print (roc_auc_score(y_test, self.test_prediction_probs[:,1]))
        #print (classification_report(y_test,self.test_predictions))
        #print (log_loss(y_test,self.test_predictions))


        #Calculate feature importance for each model
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(len(X_test.columns)):
            print("%d. %s (%f)" % (f + 1, X_test.columns[indices[f]], importances[indices[f]]))
        
        
        self.model = model
        self.Input = Input
        
    

    def predict(self,NFIRS,ACS=[],predict_year='Current'):
        pass
    
    @staticmethod
    def munge_dataset(top10,fires,ACS,n_years,test_year_idx):    
        years = top10.columns
        test_loc = test_year_idx
        
        # convert format for consistent output
        X =  fires.iloc[:,test_loc-n_years:test_loc].copy()
        
        #X.columns = ['year-{}'.format(n_years-1 - year) for year in range(n_years-1)]

        #sm = np.nansum(X, axis = 1 )
        #mu = np.nanmean(X, axis = 1)
        mx = np.nanmax(X, axis =1)
        md = np.nanmedian(X,axis =1 )
        X['Median'] = md  
        #X['Sum']  = sm
        #X['Mean'] = mu
        X['Max']  = mx
        
        
        
    
        
        y  = top10.iloc[:,test_loc]
        
    



        # merge in ACS Data into X unless NFIRS-Only model
        out_fires = []
        if not ACS.empty:
            
            # save copy for manual validation
            out_fires = X.copy().merge(ACS, how ='left',left_index = True, right_index = True)
            
            X=X[['Max','Median']] # drop all other NFIRS columns that have low feature importance scores
            X = X.merge(ACS, how ='left',left_index = True, right_index = True)
            
            
            
            

        
        
        return X,y,out_fires 

    @staticmethod
    def munge_dataset_test(top10,fires,ACS,n_years,test_year_idx):    
        years = top10.columns
        test_loc = test_year_idx
        
        # convert format for consistent output
        X =  fires.iloc[:,test_loc-n_years:test_loc].copy()
        
        #X.columns = ['year-{}'.format(n_years-1 - year) for year in range(n_years-1)]

        #sm = np.nansum(X, axis = 1 )
        #mu = np.nanmean(X, axis = 1)
        mx = np.nanmax(X, axis =1)
        md = np.nanmedian(X,axis =1 )
        X['Median'] = md  
        #X['Sum']  = sm
        #X['Mean'] = mu
        X['Max']  = mx
        
        
        
    
        
        y  = top10.iloc[:,test_loc]
        
    



        # merge in ACS Data into X unless NFIRS-Only model
        out_fires = []
        if not ACS.empty:
            
            # save copy for manual validation
            out_fires = X.copy().merge(ACS, how ='right',left_index = True, right_index = True)
            
            X=X[['Max','Median']] # drop all other NFIRS columns that have low feature importance scores
            X = X.merge(ACS, how ='right',left_index = True, right_index = True)
            
            
            
            

        
        
        return X,y,out_fires     
    
    
    @staticmethod
    def resample_df(X,y,upsample=True,seed = 0):
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
    def  __init__(self):
                

        self.models = {}
        
    def trainModels(self,ARC,ACS,SVI, ACS_variables,svi_use, data_path):

        if not ACS_variables:
                ACS_variables = ACS.data.columns

        self.ACS_variables_used = ACS_variables
        #ACS = ACS.data[ACS_variables]



        self.arc = ARC.data
        self.acs = ACS.data[ACS_variables]
        self.acs_pop = ACS.tot_pop
        self.svi = SVI.data
        self.svi_use = svi_use
        
        self.trainStatisticalModel()
        return  self.trainDLModel(data_path)


    def trainStatisticalModel(self):



        # single level models 
        for geo in ['State','County','Tract','Block_Group'] :
            df = self.createSingleLevelSmokeAlarmModel(geo) 
            name = 'SmokeAlarmModel' + geo + '.csv'
            df.index.name = 'geoid'
            self.models[geo] = df
            df.index =  '#_' + df.index 
            out_path =  utils.DATA['model-outputs'] /'Smoke_Alarm_Single_Level' / name
            df.to_csv(out_path)
            
        
        self.createMultiLevelSmokeAlarmModel()
        
    





    def trainDLModel(self, data_path):
        
        sm = self.models['MultiLevel'].copy()
        if self.svi_use:
            df = self.svi.copy()
        else:
            df = self.acs.copy()        
        
        
        sm = sm.reset_index()
        sm['geoid'] = sm['geoid'].str[2:]
        sm['tract'] = sm['geoid'].str[:-1]
        sm.set_index('geoid', inplace =  True)
        sm_all = sm.copy()
        sm = sm[ sm['geography'].isin(['tract','block_group']) ].copy()
        
        rd = self.create_rurality_data(sm,data_path, True)
        rd_all = self.create_rurality_data(sm_all, data_path)
        
        if self.svi_use:
            rd = rd['Population Density (per square mile), 2010'].to_frame()
            rd_all = rd_all['Population Density (per square mile), 2010'].to_frame()
            

        acs_pop = self.acs_pop[self.acs_pop['tot_population'] >= 50 ] 
        rd = rd.filter(acs_pop.index, axis= 0)
        
        mdl,X_test,y_test = self.trainXGB(X = rd, df = df, y = sm, predict = 'Presence', modeltype= 'XGBoost')
        
        predictions = mdl.predict(rd_all.merge(df,how = 'left', left_index = True, right_index = True) )

        sm_all['Predictions'] =np.clip(predictions,0,100)  
        
        sm_all.loc[:,['num_surveys','geography',
              'detectors_found_prc',
              'detectors_working_prc',
              'Predictions'] ]
        sm_all = sm_all.merge(rd_all['Population Density (per square mile), 2010'],how = 'left',left_index = True,right_index = True)
         
        return sm_all
    
    def trainXGB(self, X, df, y, predict, modeltype):
        
       assert(predict in ['Presence', 'Working'])  
        
       model = xgb.XGBRegressor(objective = 'reg:squarederror',random_state = 0)
            
       if  predict == 'Presence':
           y = y['detectors_found_prc']
       elif predict =='Working':
           y = y['detectors_working_prc']


       # merge in ACS Data into X unless NFIRS-Only model
       if not df.empty:
           X = X.merge(df, how ='left',left_index = True, right_index = True)
           y = y.filter(X.index)
   
       # Create 80/20 training/testing set split
       X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .2 )
       model = model.fit(X_train,y_train)
        # Calculate training set performance
       train_predictions = model.predict(X_train)
       print('-----Training_Performance------')
       print(mean_squared_error(y_train, train_predictions))
       print ('Test RMSE: {}'.format(mean_squared_error(y_train, train_predictions, squared = False)) )
       print ('Test MAE: {}'.format(mean_absolute_error(y_train, train_predictions)) )
       sns.scatterplot(y_train,train_predictions) 
       plt.show()
    
       # Calculate test set performance
       test_predictions = model.predict(X_test)
       print ('-----Test Performance ----- ')
       print ('Test RMSE: {}'.format(mean_squared_error(y_test, test_predictions, squared = False)) )
       print ('Test MAE: {}'.format(mean_absolute_error(y_test, test_predictions)) )
       sns.scatterplot(y_test,test_predictions) 
       plt.show()
       print ('Test Correlation: {}'.format(pearsonr(y_test, test_predictions)) )
       print ('Test R-squared: {}'.format(r2_score(y_test, test_predictions)) )

       #Calculate feature importance for each model
       if modeltype == 'XGBoost':
           importances = model.feature_importances_
           indices = np.argsort(importances)[::-1]
           print("\n Feature ranking:")
           for f in range(len(X_test.columns)):
               print("%d. %s (%f)" % (f + 1, X_test.columns[indices[f]], importances[indices[f]])) 

       return  model,X_test,y_test


        
        
    def create_rurality_data(self, sm, data_path, subset_county = False): 
        #Rurality Data Munging 
        rd = pd.read_csv( data_path/'Master Project Data'/'Tract Rurality Data.csv', dtype = {'Tract':'object'},encoding = 'latin-1' )
        rd['Population Density (per square mile), 2010'] =  rd['Population Density (per square mile), 2010'].str.replace(',','').astype('float')
        rd['Tract'] = rd['Tract'].str[2:]
        rd = rd.iloc[:,[0,2,4,6,8]]
        block_tract = sm['tract'].to_frame()
        block_tract = block_tract.reset_index()
        rd = block_tract.merge(rd, how = 'left', left_on = 'tract' , right_on ='Tract')
        rd.set_index('geoid',inplace= True)
        rd = rd.iloc[:,2:]
        rd['Select State'] = rd['Select State'].astype('category')

        # add state level model estimates 
        sms = pd.rd = pd.read_csv( data_path /'Model Outputs'/'Smoke_Alarm_Single_Level'/ 'SmokeAlarmModelState.csv')
        sms['geoid'] = sms['geoid'].str[2:]
        sms =  sms.loc[:,['geoid','detectors_found_prc']]
        sms = sms.rename(columns= {'geoid':'state_geoid'}  )

        rd['state_geoid'] = rd.index.str[:2]
        rd = rd.reset_index()
        rd = rd.merge(sms,how = 'left', on = 'state_geoid' )
        rd.drop('state_geoid',axis = 1,inplace = True)
        rd = rd.rename(columns = {'detectors_found_prc':'state_detectors_found_prc'}) 
        rd = rd.set_index('geoid')


        # add county level estimates
        smc = pd.read_csv( data_path /'Model Outputs'/'Smoke_Alarm_Single_Level'/ 'SmokeAlarmModelCounty.csv')
        smc['geoid'] = smc['geoid'].str[2:]
        if subset_county:
            smc.iloc[0::2,:] = np.nan
        smc =  smc.loc[:,['geoid','detectors_found_prc']]
        smc = smc.rename(columns= {'geoid':'county_geoid'}  )

        rd['county_geoid'] = rd.index.str[:5]
        rd = rd.reset_index()
        rd = rd.merge(smc,how = 'left', on = 'county_geoid' )
        rd.drop('county_geoid',axis = 1,inplace = True)
        rd = rd.rename(columns = {'detectors_found_prc':'county_detectors_found_prc'}) 
        rd = rd.set_index('geoid')
  #  rd['RUCA_rurality_index'] = rd['Primary RUCA Code 2010']
  #  rd[rd['RUCA_rurality_index'] > 10 ] = np.NaN
        rd = rd.iloc[:,3:]
        self.rd = rd

        return rd 
        
        

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
        #acs.drop_duplicates(inplace = True) #why drop duplicates?
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
            ('tract', 11 + 2 , self.models['Tract'].copy() ),
            ('county', 5 + 2 , self.models['County'].copy() ),
            ('state', 2 + 2 , self.models['State'].copy() )
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


        #MultiLevelModel = MultiLevelModel.reset_index()
        self.models['MultiLevel'] = MultiLevelModel
        

    
        out_path =  utils.DATA['model-outputs'] / 'SmokeAlarmModelMultiLevel.csv'
        MultiLevelModel.to_csv(out_path)






    


    