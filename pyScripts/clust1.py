# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 00:26:02 2018

@author: sonu
"""

#%%
import pandas as pd
import numpy as np
import pdb

#%%

data_path = "../output/hcdr_f.csv"
data_train = pd.read_csv(data_path)

#%%

#data_train = df[df['TARGET'].notnull()]
train_df = data_train[data_train['TARGET'].notnull()]
test_df = data_train[data_train['TARGET'].isnull()]


data_train_pos = train_df[train_df.TARGET == 1]
data_train_neg = train_df[train_df.TARGET == 0]

#%% 

# check no of non NA features

#data_train_pos.isnotnull()

not_null_cols = np.sum(data_train_neg.isnull())
not_null_cols = not_null_cols [not_null_cols == 0].index.values
print(len(not_null_cols))

#%%

y = data_train_neg[not_null_cols].TARGET.values
X = data_train_neg[not_null_cols].drop('TARGET', axis=1)


#%%

#%%

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
#import matplotlib.pyplot as plt


#%%

# k means determine k
#==============================================================================
# distortions = []
# K = range(1,30)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
#     print(k)
#==============================================================================
    
    
#%%    

# Plot the elbow
#==============================================================================
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
#==============================================================================


#%%

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#%%

data_train_neg['y_kmeans'] = y_kmeans
#X['TARGET'] = y

#%%

pdb.set_trace()
X_sub = data_train_neg[data_train_neg.y_kmeans == 4]

X_final = pd.concat([X_sub, data_train_pos])

#%%

import xgboost as xgb

#%%

xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=200,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#%%

xgb_param = xgb1.get_xgb_params()

#%%

ytrain =  X_final.TARGET.values
dtrain = X_final.drop('TARGET', axis=1)

#%%

#xgtrain = 

xgtrain = xgb.DMatrix(dtrain.values, label=ytrain)

#%%
cvresult = xgb.cv(xgb_param, xgtrain, 
                  num_boost_round=xgb1.get_params()['n_estimators'], nfold=3,
            metrics='auc', early_stopping_rounds=50, verbose_eval = 2) #show_progress=True
        
##alg.set_params(n_estimators=cvresult.shape[0])
        
#%%        


