from __future__ import division
from functools import reduce


from sklearn.model_selection import train_test_split

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold


import from_lib
import pandas as pd
import numpy as np
import cPickle as pickle
import gc
import pdb
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

#file = open('../output/features.pkl','rb')
#df = pickle.load(file)

data_path = "../output/hcdr_f.csv"
df = pd.read_csv(data_path)

data_train.shape

stratified = False; debug= False

cn = df.columns.tolist()
cn[0] = 'SK_ID_CURR'

df.columns = cn

df.set_index('SK_ID_CURR', inplace=True)
df = df[df['TARGET'].notnull()]

np.sum(pd.isnull(df)) [ np.sum(pd.isnull(df)) > 0 ]
df = df[df['TARGET'].notnull()]

bu_SK_ID_CURR = bureau_features1.SK_ID_CURR.values
bureau_features1['bu_SK_ID_CURR'] = bu_SK_ID_CURR

bureau_features1 = bureau_features1.set_index('SK_ID_CURR')

df = df.join(bureau_features1, how='left', rsuffix="_bureau") # , on = 'SK_ID_CURR'
d = df.join(bureau_features1, how='left', rsuffix="_bureau") # , on = 'SK_ID_CURR'

df[ np.array(['bu_SK_ID_CURR', 'bureau_mean_ACTIVE_x', 'bureau_mean_CLOSED_x' ])].head()

df[pd.notnull(df.bu_SK_ID_CURR)][ np.array(['bu_SK_ID_CURR', 'bureau_mean_ACTIVE_x', 'bureau_mean_CLOSED_x','TARGET' ])].head()
# 179182
SK_ID_CURR_bureau, 
'bureau_mean_ACTIVE_x', 'bureau_mean_CLOSED_x' 

# 220602
CC_NAME_CONTRACT_STATUS_Completed_MAX, CC_NAME_CONTRACT_STATUS_Refused_MIN, 


df[pd.isnull(df.bu_SK_ID_CURR)]

bs = [b for b in cn if 'bureau' in b]

bs = ['bu_SK_ID_CURR', 'TARGET'] + bs
df[pd.isnull(df.bu_SK_ID_CURR)][bs].iloc[0:4,0:4]
df[pd.notnull(df.bu_SK_ID_CURR)][bs].iloc[0:4,0:4]


cc = [b for b in cn if 'CC_' in b]
application_train[application_train.SK_ID_CURR == 174405]

CC_NAME_CONTRACT_STATUS_Completed_MAX

df[pd.isnull(df.CC_NAME_CONTRACT_STATUS_Completed_MAX)][cc].shape

df[pd.isnull(df.CC_NAME_CONTRACT_STATUS_Completed_MAX)][['bu_SK_ID_CURR','TARGET'] + cc].iloc[0:5,0:8]
