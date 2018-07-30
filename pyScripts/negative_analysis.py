# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:05:10 2018

@author: sonu
"""

#%%
import pandas as pd
import numpy as np
import pdb

pd.set_option('display.max_columns', None)

cm_tags_path = "../output/cm_tags.csv"
scores_path = "../output/scores.csv"

#%%

cm_tags = pd.read_csv(cm_tags_path)
scores = pd.read_csv(scores_path)

#%%

cm_tags.groupby(['m_name'], as_index=False).agg({'cm_tags':'count'})

#%%

cm_tags_test = cm_tags[cm_tags['Type'] == 'test']
pd.crosstab(index=cm_tags_test ['m_name'], 
            columns = cm_tags_test['cm_tags'])


#%%

cm_tags_train = cm_tags[cm_tags['Type'] == 'train']

pd.crosstab(index=cm_tags_train ['m_name'], 
            columns = cm_tags_train['cm_tags'])
            
#%%            

# From all TN select 12 % of each

cm_tags_train.cm_tags.value_counts()
            
#%%

models =  cm_tags.m_name.unique()            
cut_point = 0.12

neg_set_ids = np.array([])
m_names = np.array([])

rest = np.array([])
m_names_for_rest = np.array([])

for m in models :
    sub_TN = cm_tags_train[(cm_tags_train.m_name == m) & (cm_tags_train.cm_tags == 'TN')]
    
    np.random.seed(1234)

    size = int(sub_TN.shape[0] * cut_point)
    tn_SK_ID_CURR = np.random.choice(sub_TN.SK_ID_CURR.values, size, replace=False)
    
    #pdb.set_trace()
    rest_ng_ids = sub_TN.SK_ID_CURR.values[np.in1d(sub_TN.SK_ID_CURR.values , tn_SK_ID_CURR, invert=True)]
    
    rest = np.append(rest, rest_ng_ids )
    m_names_for_rest = np.append(m_names_for_rest, np.repeat(m, len(rest_ng_ids )))
    
    neg_set_ids = np.append(neg_set_ids, tn_SK_ID_CURR)
    m_names = np.append(m_names, np.repeat(m, len(tn_SK_ID_CURR)))
    
    
#%%    
pdb.set_trace()    
neg_set_ids = pd.DataFrame({'SK_ID_CURR' : neg_set_ids,
                            'm_names' : m_names})    
neg_set_ids.to_csv("../output/neg_set_ids.csv", index=False)

rest_frame = pd.DataFrame({'SK_ID_CURR' : rest,
                            'm_names' : m_names_for_rest})

rest_frame.to_csv("../output/restTN.csv", index=False)
#%%    
    
len(cm_tags_train[cm_tags_train.ACT == 1].SK_ID_CURR.unique())
    
    