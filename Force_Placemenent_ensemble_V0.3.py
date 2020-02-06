# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:00:20 2019

@author: ashok.swarna
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#==============Input Data=====================================================================

os.chdir('C:/Users/ashok.swarna/Desktop')

#force_placement_data=pd.read_excel("2017 Flood_Final_12_1.xlsx",sheet_name='Sheet1')

force_placement_data=pd.read_excel("2018_Final _V3 - Copy_Final.xlsx",sheet_name='Sheet1')

#=============Feature Engineering================================================================

force_placement_data['Customer_N_Address']=force_placement_data['Customer_Name']+'-'+force_placement_data['Property_Address']
force_placement_data['Policy_lapse_time']=force_placement_data['Policy_Expiration_Date']-force_placement_data['Policy_issue_date']

force_placement_data['Premium']=force_placement_data['Premium_Amount']
force_placement_data.Premium.fillna(force_placement_data.Anticipated_Premium_amount,inplace=True)
force_placement_data.Premium.fillna(0)

#==========================Variable Selection for Model (as per business logic)==================

fp_data=force_placement_data[['Customer_N_Address','YOY_No_of_Force_placement',
                              'Loan_Type','Policy_lapse_time','Loan_amount',
                              'Property_Type_Commercial_/_Residential',
                              'Portfolio','Business_Scale','State_Code',
                              'Premium','Coverage_Amount',
                              'Category',
                              'Ratings_of_agency']]

#=========================Creating Force Placement Flag==========================================

fp_data=fp_data[(force_placement_data.Category=="POLICY RENEWAL")
                             | (force_placement_data.Category=="FORCE PLACEMENT")
                             | (force_placement_data.Category=="FP RENEWAL")]

fp_data.Category.value_counts()

fp_data['Force_Placement_flag'] = np.where(fp_data['Category']=="POLICY RENEWAL",0,1)

fp_data.Force_Placement_flag.value_counts()

fp_data.drop_duplicates(keep=False, inplace=True)

fp_data_1=fp_data.drop(['Customer_N_Address','Category'],axis=1)

#=======================Type Casting and Missing Value Treatment================================

fp_data_1['YOY_No_of_Force_placement'].astype(int)

fp_data_1['Loan_amount'].astype(int)

#fp_data_1['Policy_lapse_time'].fillna(0)

fp_data_1['Policy_lapse_time'].replace({pd.NaT: pd.Timedelta('0 days')}, inplace=True)

fp_data_1['Policy_lapse_time'] = fp_data_1['Policy_lapse_time'].dt.days

fp_data_1.to_csv("force_placemenent_data2_5_2_stage_1_op.csv")

#==============Manual====================================================================

#=============Input Cleaned Data=========================================================


fp_data_2=pd.read_csv("force_placemenent_data2_5_2_stage_1_op.csv")


#=====================Futher - Missing Value treatment===================================

fp_data_2['Premium'].fillna(0,inplace=True)
fp_data_2['Business_Scale'].fillna("NA",inplace=True)


fp_data_2_categorical=fp_data_2[['Loan_Type','Property_Type_Commercial_/_Residential',
                                 'Portfolio','Business_Scale','State_Code']]

fp_data_dummies = pd.get_dummies(fp_data_2_categorical,drop_first=True)
    
fp_data_2_numerical=fp_data_2[['YOY_No_of_Force_placement','Policy_lapse_time',
                               'Loan_amount','Premium',
                               'Coverage_Amount',
                               'Force_Placement_flag']]

#====================Creating dummy variables for Categorical Variables================

fp_data_3=pd.concat([fp_data_2_numerical,fp_data_dummies],axis=1)

X1 = fp_data_3.iloc[:,0:5]
X2 = fp_data_3.iloc[:,6:len(fp_data_3.columns)]

X=pd.concat([X1.reset_index(drop=True), X2], axis=1)

Y = fp_data_3['Force_Placement_flag']

#==================Train/Test Split (80/20)============================================

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

#eval_set = [(X_test, Y_test)]

start_time = time.time()


#==============================XGBoost===================================================
'''X_train=X_FC_train
X_test=X_FC_train
Y_train=Y_FC_train
Y_test=Y_FC_test'''

def classifier_xgb(X_train,X_test,Y_train,Y_test,flag):

    #=====This function can return AUC/dump (pickle) the model depending on value of flag
    #if(flag==0) : it returns AUC
    #if(flag==1) : it dumps(pickles) the model===========
    
    estimator_xgb = XGBClassifier(
        objective= 'binary:logistic',
        nthread=4,
        seed=42)
    
    #=============Hyperparamters============================
    
    parameters_xgb = {
        "early_stopping_rounds":[5],
        "eval_metric" : ["mae"],
        "eval_set" : [[X_test, Y_test]],            
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
                     }
    
    #============Hyperparameter tuning using Grid Search using AUC as Scoring metric===
    
    grid_search_xgb = GridSearchCV(
        estimator=estimator_xgb,
        param_grid=parameters_xgb,
        scoring = 'roc_auc',
        n_jobs = 10,
        cv = 10,
        verbose=True)
   
    grid_search_xgb.fit(X_train, Y_train)
 
    #grid_search_xgb.best_estimator_.feature_importances_    
    y_pred_xgb = grid_search_xgb.predict(X_test)    
    predictions_xgb = [round(value) for value in y_pred_xgb]    
    accuracy_xgb = accuracy_score(Y_test, predictions_xgb)    
    auc_xgb=roc_auc_score(Y_test,predictions_xgb)    
    print("Accuracy: %.2f%%" % (accuracy_xgb * 100.0))    
    precision_recall_f1_xgb=classification_report(Y_test, predictions_xgb)
        
    
    if flag==1:
        return auc_xgb
    else:
        return pickle.dump(grid_search_xgb, open('XGB_model.sav', 'wb'))
    
#==========================================Random Forest================================
 
def classifier_random_forest(X_train,X_test,Y_train,Y_test,flag):
    
    estimator_rf = RandomForestClassifier(
        random_state = 1,
        n_estimators = 750,
        max_depth = 15, 
        min_samples_split = 5,  
        min_samples_leaf = 1)
    
    parameters_rf = {
                    'bootstrap': [True],
                    'max_depth': [80, 90, 100, 110],
                    'max_features': [2, 3],
                    'min_samples_leaf': [3, 4, 5],
                    'min_samples_split': [8, 10, 12],
                    'n_estimators': [100, 200, 300, 1000]
                    }
            
    grid_search_rf=GridSearchCV(
                estimator=estimator_rf,
                param_grid=parameters_rf,
                scoring = 'roc_auc',
                n_jobs = 10,
                cv = 10,
                verbose=True)
                            
    grid_search_rf.fit(X_train,Y_train)    
    #grid_search_rf.best_estimator_.feature_importances_ 
    y_pred_rf = grid_search_rf.predict(X_test)    
    predictions_rf = [round(value) for value in y_pred_rf]    
    accuracy_rf = accuracy_score(Y_test, predictions_rf)    
    auc_rf=roc_auc_score(Y_test,predictions_rf)    
    print("Accuracy: %.2f%%" % (accuracy_rf * 100.0))    
    precision_recall_f1_rf=classification_report(Y_test, predictions_rf)
    
    if flag==1:
        return auc_rf
    else:
        return pickle.dump(grid_search_rf, open('RF_model.sav', 'wb'))
    
#=========================================Light GBM=====================================
    
def classifier_light_gbm(X_train,X_test,Y_train,Y_test,flag):
    
    estimator_light_gbm=lgb.LGBMClassifier()
    
    parameters_light_gbm={
                        'learning_rate': [ 0.1],
                        'num_leaves': [31],
                        'boosting_type' : ['gbdt'],
                        'objective' : ['binary']
                        }
    
    grid_search_gbm=GridSearchCV(
                estimator=estimator_light_gbm,
                param_grid=parameters_light_gbm,
                scoring = 'roc_auc',
                n_jobs = 10,
                cv = 10,
                verbose=True)
    
    grid_search_gbm.fit(X_train,Y_train)    
    #grid_search_gbm.best_estimator_.feature_importances_
    y_pred_gbm = grid_search_gbm.predict(X_test)    
    predictions_gbm = [round(value) for value in y_pred_gbm]    
    accuracy_gbm = accuracy_score(Y_test, predictions_gbm)    
    auc_gbm=roc_auc_score(Y_test, predictions_gbm)    
    print("Accuracy: %.2f%%" % (accuracy_gbm * 100.0))    
    precision_recall_f1_gbm=classification_report(Y_test, predictions_gbm)
    
    if flag==1:
        return auc_gbm
    else:
        return pickle.dump(grid_search_gbm, open('Light_GBM_model.sav', 'wb'))
    
#============================================SVM========================================
    
def classifier_SVM(X_train,X_test,Y_train,Y_test,flag):
    
    estimator_SVM=SVC()
    
    parameters_SVM={'C': [0.1, 1, 10, 100, 1000],  
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                          'kernel': ['rbf']             
                   }
    
    grid_search_SVM=GridSearchCV(
                estimator=estimator_SVM,
                param_grid=parameters_SVM,
                scoring = 'roc_auc',
                n_jobs = 10,
                cv = 10,
                verbose=True)
    
    grid_search_SVM.fit(X_train,Y_train)    
    #grid_search_SVM.best_estimator_.feature_importances_
    y_pred_SVM = grid_search_SVM.predict(X_test)    
    predictions_SVM = [round(value) for value in y_pred_SVM]    
    accuracy_SVM = accuracy_score(Y_test, predictions_SVM)    
    auc_SVM=roc_auc_score(Y_test, predictions_SVM)    
    print("Accuracy: %.2f%%" % (accuracy_SVM * 100.0))    
    precision_recall_f1_SVM=classification_report(Y_test, predictions_SVM)
    
    if flag==1:
        return auc_SVM
    else:
        return pickle.dump(grid_search_SVM, open('SVM_model.sav', 'wb'))
    

def logistic_regression(X_train,X_test,Y_train,Y_test,flag):
    
    logreg=LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    print(logreg.coef_)
    print(logreg.intercept_)
    
    y_pred = logreg.predict(X_test)
    
    predictions_logistic= [round(value) for value in y_pred]    
    
    roc_auc_score(Y_test, predictions_logistic)
    accuracy_logistic = accuracy_score(Y_test, predictions_logistic)    
    
    confusion_matrix(Y_test, y_pred)
    
    if flag==0:
        return  logreg.predict_proba(X_test)
    
  
#==================Get the AUC for each model=========================================
    
auc_xgb=classifier_xgb(X_train,X_test,Y_train,Y_test,1)
auc_rf=classifier_random_forest(X_train,X_test,Y_train,Y_test,1)
auc_gbm=classifier_light_gbm(X_train,X_test,Y_train,Y_test,1)
auc_SVM=classifier_SVM(X_train,X_test,Y_train,Y_test,1)

perf=pd.DataFrame({'Algorithm':['XGBoost','Random Forest','Light GBM','SVM'],
                       'AUC':[auc_xgb,auc_rf,auc_gbm,auc_SVM]})
        
perf.sort_values("AUC",axis=0,ascending=False,inplace= True,
                     na_position='last')   

#===================Select the best model with highest AUC=============================

best_algo=perf.loc[0,'Algorithm']


#=================Dump the best model in pcikle format (.sav)==========================

if best_algo=='XGBoost':
    classifier_xgb(X,X_test,Y,Y_test,0)
elif best_algo=='Random Forest':
    classifier_random_forest(X,X_test,Y,Y_test,0)
elif best_algo=='Light GBM':
    classifier_light_gbm(X,X_test,Y,Y_test,0)
else : classifier_SVM(X,X_test,Y,Y_test,0)
    
#performance.to_csv("performance.csv",index=False)

end_time = time.time()
time_execution=end_time-start_time
print("Time taken for execution ",time_execution/60,"min")

#===========================Loading Model========================================

model=pickle.load(open("XGB_model.sav","rb"))

#===========================Score a new data=====================================

pred=pd.DataFrame(model.predict_proba(X_test)).iloc[:,1]

#==========================Extract feature importance============================

model.best_estimator_.feature_importances_

#=========================Flat Cancellation======================================

fp_fc_data=force_placement_data[['Customer_N_Address','YOY_No_of_Force_placement',
                              'Loan_Type','Policy_lapse_time','Loan_amount',
                              'Property_Type_Commercial_/_Residential',
                              'Portfolio','Business_Scale','State_Code',
                              'Premium','Coverage_Amount',
                              'Category',
                              'Ratings_of_agency']]

fp_fc_data_1=fp_fc_data[(fp_fc_data.Category=="FP CANCELLATION")|
                         (fp_fc_data.Category=="FP CANCEL/UPDATE")|
                         (fp_fc_data.Category=="FORCE PLACEMENT")]


fp_fc_data_1_cat=fp_fc_data_1[['Loan_Type','Property_Type_Commercial_/_Residential',
                               'Portfolio','Business_Scale','State_Code']]

fp_fc_data_1_numeric=fp_fc_data_1[['YOY_No_of_Force_placement','Policy_lapse_time',
                                   'Loan_amount','Premium','Coverage_Amount',
                                   'Category']]

fp_fc_data_1_dummies=pd.get_dummies(fp_fc_data_1_cat,drop_first=True)


fp_fc_data_2=pd.concat([fp_fc_data_1_dummies.reset_index(drop=True), fp_fc_data_1_numeric.reset_index(drop=True)], axis=1)


fp_fc_data_2['Flat_Cancellation_Flag'] = np.where(fp_fc_data_2['Category']=="FORCE PLACEMENT",0,1)

del fp_fc_data_2['Category']

fp_fc_data_2.to_csv("flat_cancellation_ip.csv")

#===============================Manual===========================================

#================================================================================

fp_fc_data_3=pd.read_csv("flat_cancellation_ip.csv")

X_FC = fp_fc_data_3.iloc[:,0:(len(fp_fc_data_2.columns)-1)]

Y_FC = fp_fc_data_3['Flat_Cancellation_Flag']

X_FC_train,X_FC_test,Y_FC_train,Y_FC_test=train_test_split(X_FC,Y_FC,test_size=0.2,random_state=1)

auc_xgb=classifier_xgb(X_FC_train,X_FC_test,Y_FC_train,Y_FC_test,1)
auc_rf=classifier_random_forest(X_train,X_test,Y_train,Y_test,1)
auc_gbm=classifier_light_gbm(X_train,X_test,Y_train,Y_test,1)
auc_SVM=classifier_SVM(X_train,X_test,Y_train,Y_test,1)













