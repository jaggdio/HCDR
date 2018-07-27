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
X = X.drop('SK_ID_CURR', axis=1)


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

pdb.set_trace()

#%%
data_train_neg['y_kmeans'] = y_kmeans
data_train_pos['y_kmeans'] = -1


df = pd.concat([data_train_neg, data_train_pos])
df.to_csv("../output/train_features.csv", index=False)
