# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:45:30 2018

@author: jayphate
"""

#%%

from functools import reduce
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
#%%

bureau_balance = import_data('../input/bureau_balance.csv')
bureau = import_data('../input/bureau.csv')

#%%

buro_grouped_size = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
 
buro_counts = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)

buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]

buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max


#%%

buro = bureau.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')

#%%

#buro[buro.SK_ID_BUREAU == 5001710][['SK_ID_CURR',u'STATUS_0', u'STATUS_1', u'STATUS_2', u'STATUS_3',u'STATUS_4', u'STATUS_5', u'STATUS_C', u'STATUS_X', u'MONTHS_COUNT',
# u'MONTHS_MIN', u'MONTHS_MAX']].groupby('SK_ID_CURR').max()
 
#%%
#bureau.head()

bureau_years =  bureau[['SK_ID_CURR','SK_ID_BUREAU','CREDIT_ACTIVE','DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_UPDATE']]

bureau_years['DAYS_CREDIT_YRS'] = bureau_years.DAYS_CREDIT / 365
bureau_years['DAYS_CREDIT_ENDDATE_YRS'] = bureau_years.DAYS_CREDIT_ENDDATE / 365
bureau_years['DAYS_ENDDATE_FACT_YRS'] = bureau_years.DAYS_ENDDATE_FACT / 365
bureau_years['CREDIT_DAY_OVERDUE_YRS'] = bureau_years.CREDIT_DAY_OVERDUE/ 365
bureau_years['DAYS_CREDIT_UPDATE_YRS'] = bureau_years.DAYS_CREDIT_UPDATE / 365

#bureau_years.head()
#%%

def add_SK_ID_CURR_to_crossTabRes(crossTabRes):
    index_frame = crossTabRes.index.to_frame(index=False)
    crossTabRes = crossTabRes.reset_index(drop=True)
    df = pd.merge(index_frame, crossTabRes, left_index=True, right_index=True)
    return df


#%%

crd_active_bureau = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_ACTIVE'])
crd_active_bureau = add_SK_ID_CURR_to_crossTabRes(crd_active_bureau )
print(crd_active_bureau.shape)

crd_active_currency = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_CURRENCY'])
crd_active_currency = add_SK_ID_CURR_to_crossTabRes(crd_active_currency )
print(crd_active_currency.shape)

# NaN menas there is no values for credit status
crd_days_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns=bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_CREDIT_YRS'], aggfunc=np.mean, dropna=False)
crd_days_by_active_status.rename(columns=lambda x : 'DAYS_CREDIT_YRS_' + x, inplace=True) 
crd_days_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_days_by_active_status)
print(crd_days_by_active_status.shape)

crd_enddate_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns = bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_CREDIT_ENDDATE_YRS'], aggfunc=np.mean, dropna=False) 
crd_enddate_by_active_status.rename(columns=lambda x: 'DAYS_CREDIT_ENDDATE_YRS_' + x, inplace=True)
crd_enddate_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_enddate_by_active_status)
print(crd_enddate_by_active_status.shape)


#==============================================================================
# sub = bureau_years[np.in1d( bureau_years.SK_ID_CURR, [100105,100106])]
# pd.crosstab(index=sub['SK_ID_CURR'], 
#             columns = sub['CREDIT_ACTIVE'], 
#             values=sub['DAYS_CREDIT_ENDDATE_YRS'], 
#             aggfunc=np.mean, dropna=False)
# 
#==============================================================================



crd_enddate_fact_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns = bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_ENDDATE_FACT_YRS'], aggfunc=np.mean, dropna=False)
crd_enddate_fact_by_active_status.rename(columns = lambda x : 'DAYS_ENDDATE_FACT_YRS_' + x, inplace=True)
crd_enddate_fact_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_enddate_fact_by_active_status)
print(crd_enddate_fact_by_active_status.shape)
# 
crd_day_overdue = bureau_years.groupby(['SK_ID_CURR'], as_index=False).agg({'CREDIT_DAY_OVERDUE_YRS':'mean'})

# NaN means no overdue
crd_max_amt_overdue  = bureau.groupby(['SK_ID_CURR'], as_index=False).agg({'AMT_CREDIT_MAX_OVERDUE':'mean'})

# 
crd_cnt_prolong = bureau.groupby(['SK_ID_CURR'], as_index=False).agg({'CNT_CREDIT_PROLONG':'sum'})

# current credit ammount
crd_amt_sum = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM'], aggfunc=np.mean, dropna=False)
crd_amt_sum.rename(columns= lambda x : 'AMT_CREDIT_SUM_' + x, inplace=True) 
crd_amt_sum['SK_ID_CURR'] = crd_amt_sum.index.values
print(crd_amt_sum.shape)




# current debt on Credit Bureau Credit
crd_amt_debt = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_DEBT'], aggfunc=np.mean, dropna=False)
crd_amt_debt.rename(columns= lambda x : 'AMT_CREDIT_SUM_DEBT_' + x, inplace=True) 
crd_amt_debt = add_SK_ID_CURR_to_crossTabRes(crd_amt_debt)
print(crd_amt_debt.shape)


# current credit limit 
crd_sum_limit = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_LIMIT'], aggfunc=np.sum, dropna=False)
crd_sum_limit.rename(columns= lambda x : 'AMT_CREDIT_SUM_LIMIT_' + x , inplace=True) 
crd_sum_limit = add_SK_ID_CURR_to_crossTabRes(crd_sum_limit)
print(crd_sum_limit.shape)


crd_sum_overdue = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_OVERDUE'], aggfunc=np.sum, dropna=False)
crd_sum_overdue.rename(columns= lambda x : 'AMT_CREDIT_SUM_OVERDUE_' + x , inplace=True) 
crd_sum_overdue = add_SK_ID_CURR_to_crossTabRes(crd_sum_overdue)
print(crd_sum_overdue.shape )
# Credit type

crd_type = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_TYPE'], dropna=False)
crd_type = add_SK_ID_CURR_to_crossTabRes(crd_type)
print(crd_type.shape)

#crd_type = pd.get_dummies(bureau['CREDIT_TYPE'])
#crd_type['SK_ID_CURR'] = bureau['SK_ID_CURR']
# ==> conver to dummy

crd_day_update = bureau_years.groupby(['SK_ID_CURR'], as_index=False).agg({'DAYS_CREDIT_UPDATE_YRS':'mean'})
crd_day_annuity = bureau.groupby(['SK_ID_CURR'], as_index=False).agg({'AMT_ANNUITY':'mean'})
print(crd_day_annuity.shape)
####

# Now combine all frames #

####


#%%

dfs = [crd_active_bureau,
        crd_active_currency,
        crd_days_by_active_status,
        
        crd_enddate_by_active_status,
        crd_enddate_fact_by_active_status,
        
        crd_day_overdue,
        crd_max_amt_overdue,
        crd_cnt_prolong,
        crd_amt_sum,
        crd_amt_debt,
        crd_sum_limit,
        crd_sum_overdue,
        crd_type,
        crd_day_update,
        crd_day_annuity]
        
#%%
[ x.shape[0] for x in dfs]

#%%

bureau_features  = reduce(lambda left,right: pd.merge(left, right, on='SK_ID_CURR'), dfs)
bureau_features.fillna(0, inplace=True)
print(bureau_features.shape)

#%%




#%%
x = application_train_ohe
x = x[np.in1d(x.SK_ID_CURR.values, bureau_features.SK_ID_CURR.values)]
x.shape

#%%

bureau_features = bureau_features.set_index('SK_ID_CURR')

#%%

train = x.join(bureau_features, how='left', 
                                   on='SK_ID_CURR', rsuffix='_bureau')
                                
#%%

train.isnull().sum()

application_train_ohe = train

#%%


b = buro[['SK_ID_CURR', u'STATUS_0', u'STATUS_1', u'STATUS_2', u'STATUS_3',
       u'STATUS_4', u'STATUS_5', u'STATUS_C', u'STATUS_X', u'MONTHS_COUNT',
       u'MONTHS_MIN', u'MONTHS_MAX']]
       
b = b.set_index('SK_ID_CURR')
b.index.values
b.shape

nw_bureau_features = bureau_features.join(b, how='left', on='SK_ID_CURR')
nw_bureau_features.shape 

nw_bureau_features = nw_bureau_features[nw_bureau_features.columns[nw_bureau_features.isnull().mean() < 0.85 ]]
nw_bureau_features.shape

nw_bureau_features = nw_bureau_features.set_index('SK_ID_CURR')

#%%

application_train_ohe

train = application_train_ohe.join(bureau_features, how='left', 
                                   on='SK_ID_CURR', rsuffix='_bureau')

x = application_train_ohe
x = x.reset_index()

# Now take only burea data
x = x[np.in1d(x.SK_ID_CURR.values, bureau_features.SK_ID_CURR.values)]
x.shape

train = x.join(bureau_features, how='left', 
                                   on='SK_ID_CURR', rsuffix='_bureau')


len(application_train_ohe.SK_ID_CURR.values)

len(bureau_features.SK_ID_CURR.values)

np.in1d(application_train_ohe.SK_ID_CURR.values, bureau_features.SK_ID_CURR.values)


#%%

len(application_test.SK_ID_CURR.values)

# intrain, 
np.sum( np.in1d(bureau_features.SK_ID_CURR.values, application_train_ohe.SK_ID_CURR.values) )
# in test, 42320 match
np.sum( np.in1d(bureau_features.SK_ID_CURR.values, application_test.SK_ID_CURR.values) )


#%%

# 456216

#%%

d1 =bureau_features[bureau_features.SK_ID_CURR ==456216][['DAYS_CREDIT_YRS_Active','SK_ID_CURR']]

d2 = train[train.SK_ID_CURR ==456216][['SK_ID_CURR']]
d2 = d2.set_index('SK_ID_CURR')
d2

d1.join(d2, how='left', on='SK_ID_CURR', rsuffix = '_bureau', lsuffix = '_app')


#%%

num_rows = 10000
cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)

# objectcolumns
original_columns = list(cc.columns)
categorical_names = [c for c in cc.columns if cc[c].dtype == 'object']

cc.NAME_CONTRACT_STATUS.unique()

dummes = pd.get_dummies(cc, columns = categorical_names , dummy_na=True)
dummes.columns

[c for c in dummes.columns if c not in original_columns]

#cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

cc= dummes
cc.columns

cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

cc.groupby('SK_ID_CURR').size().head()

    # General aggregations
cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
# Count credit card lines
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
del cc
gc.collect()
    

    








