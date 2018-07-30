# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:13:31 2018

@author: sonu
"""

from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.externals import joblib


from sklearn.externals import joblib
#from training import get_cm_tags

import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import time
import pdb

def get_cm_tags(act, pred):
    
    cm_df = pd.DataFrame({'ACT':act, 'PRED':pred})
    cm_df['cm_tags'] = np.repeat(" ", len(act))
 
    cm_df['cm_tags'][(cm_df.ACT == 1) & (cm_df.PRED == 1)] = 'TP'
    cm_df['cm_tags'][(cm_df.ACT == 0) & (cm_df.PRED == 1)] = 'FP'

    cm_df['cm_tags'][(cm_df.ACT == 0) & (cm_df.PRED == 0)] = 'TN'
    cm_df['cm_tags'][(cm_df.ACT == 1) & (cm_df.PRED == 0)] = 'FN'
    return cm_df

cm_tags_path = "../output/cm_tags.csv"
cm_tags = pd.read_csv(cm_tags_path)
cm_tags_train = cm_tags[cm_tags['Type'] == 'train']
train_data_pos_ids = cm_tags_train[cm_tags_train.ACT == 1].SK_ID_CURR.unique()

cm_tags_test = cm_tags[cm_tags['Type'] == 'test']

test_ids = cm_tags_test.SK_ID_CURR.unique()

# Load Neg ids ##
neg_set_ids = pd.read_csv("../output/neg_set_ids.csv")
neg_set_ids = neg_set_ids.SK_ID_CURR.values
 
resst_TN = pd.read_csv("../output/restTN.csv")
resst_TN = resst_TN.SK_ID_CURR.values

other_cv = pd.read_csv("../output/other_cv.csv")
#other_cv.columns = ['SK_ID_CURR']

data = pd.read_csv("../output/train_features.csv")
data = data.drop('index',axis=1)


global_cv = data[np.in1d(data.SK_ID_CURR, other_cv.SK_ID_CURR)]


global_cv_y = global_cv.TARGET.values
global_cv_id = global_cv.SK_ID_CURR.values

global_cv = global_cv.drop('TARGET', axis=1)
global_cv= global_cv.drop('SK_ID_CURR', axis=1)
global_cv= global_cv.drop('y_kmeans', axis=1)

Dglobal_cv = xgb.DMatrix(global_cv, label=global_cv_y)


######### Build train data ######
#pdb.set_trace()
sub1 = data[np.in1d( data.SK_ID_CURR.values, train_data_pos_ids)]
sub2 = data[np.in1d( data.SK_ID_CURR.values, neg_set_ids)]

train_data = pd.concat([sub1, sub2])
test_data = data[np.in1d( data.SK_ID_CURR.values, test_ids )]


TN_subset_fromTR1 = data[np.in1d( data.SK_ID_CURR.values, resst_TN )]

############### Training #################

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective' : 'binary:logistic',
    'eval_metric' : 'auc'
}

y = train_data.TARGET.values
y_test = test_data.TARGET.values
y_train1_tn = TN_subset_fromTR1.TARGET.values

train_data = train_data.drop('TARGET', axis=1)
train_data = train_data.drop('y_kmeans', axis=1)

test_data  = test_data.drop('TARGET', axis=1)
test_data = test_data.drop('y_kmeans', axis=1)

TN_subset_fromTR1 = TN_subset_fromTR1.drop('TARGET', axis=1)
TN_subset_fromTR1 = TN_subset_fromTR1.drop('y_kmeans', axis=1)

DTrain = xgb.DMatrix(train_data.drop('SK_ID_CURR', axis=1), y)

DTest = xgb.DMatrix(test_data.drop('SK_ID_CURR', axis=1), label=y_test)

Dtrain1SubTN = xgb.DMatrix(TN_subset_fromTR1.drop('SK_ID_CURR', axis=1), label=y_train1_tn)
#xgb.cv(params, DTrain, num_boost_round = 999, early_stopping_rounds=40, verbose_eval=2 )

#%%

model = xgb.train(
        params,
        DTrain,
        num_boost_round=999,
        evals=[(DTest, "Test")],
        early_stopping_rounds=40, verbose_eval=2)

#model = joblib.load("../output/model_retrain.joblib.dat")
print("Model Name: {} and Best AUC: {:.2f} with {} rounds".format("Model", model.best_score, model.best_iteration+1))

#######

#save model
#%%


pred_test_prob = model.predict(DTest)
pred_test = pred_test_prob > 0.5
pred_test = pred_test.astype(int) 
test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, pred_test).ravel()

print(confusion_matrix(pred_test, y_test))
print roc_auc_score(y_test, pred_test_prob )
print("Precision: " + str(test_tp / (test_fp + test_tp)))
print("Recall: " + str(test_tp / (test_fn + test_tp)))
             
#%%  
#pdb.set_trace()
test_cm_tags = get_cm_tags(y_test, pred_test)
test_cm_tags['Score'] = pred_test_prob
test_cm_tags['Type'] = 'test'
test_cm_tags['SK_ID_CURR'] = test_data.SK_ID_CURR.values
#test_cm_tags['m_name'] = model_npin name
test_cm_tags_fp = test_cm_tags[test_cm_tags.cm_tags == 'FP']

data_testFP_subset = data[np.in1d(data.SK_ID_CURR.values, test_cm_tags_fp.SK_ID_CURR.unique())]

#data_testFP_subset2 =
#data_testFP_subset3 =

###########################
# Train the model again
train_data = pd.concat([sub1, sub2])
train_data = pd.concat([train_data, data_testFP_subset])

y = train_data.TARGET.values
train_data = train_data.drop('TARGET', axis=1)
train_data = train_data.drop('y_kmeans', axis=1)

DTrain = xgb.DMatrix(train_data.drop('SK_ID_CURR', axis=1), y)
xgb.cv(params, DTrain, num_boost_round = 999, early_stopping_rounds=40, verbose_eval=2 )


############################

train1_TN_prob = model.predict(Dtrain1SubTN)
pred_train1_TN = train1_TN_prob > 0.5
pred_train1_TN = pred_train1_TN.astype(int) 
test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_train1_tn, pred_train1_TN).ravel()

print(confusion_matrix(pred_train1_TN, y_train1_tn))

train1_TN_cm_tags = get_cm_tags(y_train1_tn, pred_train1_TN)
train1_TN_cm_tags['Score'] = train1_TN_prob
train1_TN_cm_tags['Type'] = 'train1_TN'
train1_TN_cm_tags['SK_ID_CURR'] = TN_subset_fromTR1.SK_ID_CURR.values
#test_cm_tags['m_name'] = model_npin name
train1_TN_cm_tags_fp = train1_TN_cm_tags[train1_TN_cm_tags.cm_tags == 'FP']

data_train1_TN_cm_tags_fp_subset = data[np.in1d(data.SK_ID_CURR.values, train1_TN_cm_tags_fp.SK_ID_CURR.unique())]

clust_names = [0, 1, 2,6,8]             
rest = np.array([])
for x in clust_names:
    sub_FP = data_train1_TN_cm_tags_fp_subset[data_train1_TN_cm_tags_fp_subset.y_kmeans== x]

    np.random.seed(1234)

    size = 1200
    fp_SK_ID_CURR = np.random.choice(sub_FP.SK_ID_CURR.values, size, replace=False)
    
    #pdb.set_trace()
    rest_ng_ids = sub_FP.SK_ID_CURR.values[np.in1d(sub_FP.SK_ID_CURR.values , fp_SK_ID_CURR, invert=False)]
    rest = np.append(rest, rest_ng_ids )
    
pdb.set_trace()
sub3 = data[np.in1d( data.SK_ID_CURR.values, rest)]

#train_data = pd.concat([train_data, sub3])
   

##########################
gcv_pred = model.predict(Dglobal_cv)
        
global_cv_pred = gcv_pred > 0.5
global_cv_pred = global_cv_pred.astype(int)

print(confusion_matrix(global_cv_pred, global_cv_y))
print roc_auc_score(global_cv_y, gcv_pred )

test_tn, test_fp, test_fn, test_tp = confusion_matrix(global_cv_y, global_cv_pred).ravel()

print("Precision: " + str(test_tp / (test_fp + test_tp)))

print("Recall: " + str(test_tp / (test_fn + test_tp)))
        
print ("Done")
#==============================================================================
# model = xgb.train(
#         params,
#         DTrain,
#         num_boost_round=999,
#         evals=[(DTest, "Test")],
#         early_stopping_rounds=40, verbose_eval=2)
# 
#==============================================================================
    