# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:31:17 2018

@author: jayphate
"""

#%%

import pandas as pd
import numpy as np

#%%

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
    
#%%

application_train = import_data('../input/application_train.csv')
application_test = import_data('../input/application_test.csv')


#%%
bureau = import_data('../input/bureau.csv')
bureau_balance = import_data('../input/bureau_balance.csv')

credit_card_balance = import_data('../input/credit_card_balance.csv')
#installments_payments = import_data('../input/installments_payments.csv')

#POS_CASH_balance = import_data('../input/POS_CASH_balance.csv')
#previous_application = import_data('../input/previous_application.csv')


#%%

application_train.TARGET.value_counts()
application_train.TARGET.astype(int).plot.hist()
application_train.dtypes.value_counts()


#%%

# Generate features

# active_CB_credits, closed_CB_credits

agg_bureau = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_ACTIVE'])
agg_bureau['SK_ID_CURR'] = agg_bureau.index.values



#%%

#
# how many loan has no credit (remainng_duration)
    # max of pos and 
    # min of neg

# how many laon has ended credit (credit ended when applied)
    # max of pos and 
    # min of neg

#%%
    
bureau_nw = bureau

bureau_nw['yrs_DAYS_CREDIT'] = bureau.DAYS_CREDIT/365
bureau_nw['yrs_DAYS_CREDIT_ENDDATE'] = bureau.DAYS_CREDIT_ENDDATE/365
bureau_nw['yrs_DAYS_ENDDATE_FACT'] = bureau.DAYS_ENDDATE_FACT/365

bureau_nw['credit_total_duration'] = bureau_nw['yrs_DAYS_CREDIT_ENDDATE'] - bureau_nw['yrs_DAYS_CREDIT']
bureau_nw[bureau_nw.SK_ID_CURR == 100003]

bureau_nw[['SK_ID_CURR', 'CREDIT_ACTIVE', ]]

from_bureau = bureau_nw[['SK_ID_CURR', 'CREDIT_ACTIVE', 'credit_total_duration' ]]
#from_bureau .groupby(['SK_ID_CURR','CREDIT_ACTIVE']).agg({'credit_total_duration': 'sum'}).head()
from_bureau  = from_bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE'], as_index=False).agg({'credit_total_duration':'mean'})

# Calculate total duration by 'CREDIT_ACTIVE'

x = pd.crosstab(index=from_bureau['SK_ID_CURR'], columns=from_bureau['CREDIT_ACTIVE'], values=from_bureau['credit_total_duration'],aggfunc=sum)
x.rename(columns=lambda x: "credit_total_duration_"+x, inplace=True)
x['SK_ID_CURR'] = x.index.values
x.head()



from_bureau = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE', 'credit_total_duration' ]]

from_bureau_sub = from_bureau[np.in1d(from_bureau.SK_ID_CURR, [100002, 100003])]

from_bureau = pd.get_dummies(from_bureau, columns=['CREDIT_ACTIVE'])

trial = 
#%%

from_bureau_sub.pivot_table(index=['SK_ID_CURR'], 
                            columns = 'CREDIT_ACTIVE',aggfunc=len)
                            
#%%
                            
pd.crosstab(index=from_bureau_sub['SK_ID_CURR'], columns = from_bureau_sub['CREDIT_ACTIVE'])




