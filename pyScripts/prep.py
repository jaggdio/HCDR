# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:12:20 2018

@author: jayphate
"""
#%%

from functools import reduce
import from_lib
import pandas as pd
import numpy as np
import gc
pd.set_option('display.max_columns', None)

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
    
def burea_balance_feature(bureau_balance):
    buro_grouped_size = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
    buro_grouped_max = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
    buro_grouped_min = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
 
    buro_counts = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)

    buro_counts_unstacked = buro_counts.unstack('STATUS')
    buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]

    buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
    buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
    buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max
    
    return buro_counts_unstacked
    
def bureau_features(bureau, bb_aggrigations):
    
    def add_SK_ID_CURR_to_crossTabRes(crossTabRes):
        index_frame = crossTabRes.index.to_frame(index=False)
        crossTabRes = crossTabRes.reset_index(drop=True)
        df = pd.merge(index_frame, crossTabRes, left_index=True, right_index=True)
        return df
    
    bb_summarize = bureau.groupby('SK_ID_CURR', as_index=False).agg(bb_aggrigations)    
    bb_summarize.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bb_summarize.columns.tolist()])
    bb_summarize = bb_summarize.rename(columns={'BURO_SK_ID_CURR_':'SK_ID_CURR'})
    
    bureau_years =  bureau[['SK_ID_CURR','SK_ID_BUREAU','CREDIT_ACTIVE','DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_UPDATE']]
    bureau_years['DAYS_CREDIT_YRS'] = bureau_years.DAYS_CREDIT / 365
    bureau_years['DAYS_CREDIT_ENDDATE_YRS'] = bureau_years.DAYS_CREDIT_ENDDATE / 365
    bureau_years['DAYS_ENDDATE_FACT_YRS'] = bureau_years.DAYS_ENDDATE_FACT / 365
    bureau_years['CREDIT_DAY_OVERDUE_YRS'] = bureau_years.CREDIT_DAY_OVERDUE/ 365
    bureau_years['DAYS_CREDIT_UPDATE_YRS'] = bureau_years.DAYS_CREDIT_UPDATE / 365
    
    crd_active_bureau = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_ACTIVE'])
    crd_active_bureau = add_SK_ID_CURR_to_crossTabRes(crd_active_bureau )

    crd_active_currency = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_CURRENCY'])
    crd_active_currency = add_SK_ID_CURR_to_crossTabRes(crd_active_currency )

    # NaN menas there is no values for credit status
    crd_days_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns=bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_CREDIT_YRS'], aggfunc=np.mean, dropna=False)
    crd_days_by_active_status.rename(columns=lambda x : 'DAYS_CREDIT_YRS_' + x, inplace=True) 
    crd_days_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_days_by_active_status)
    

    crd_enddate_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns = bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_CREDIT_ENDDATE_YRS'], aggfunc=np.mean, dropna=False) 
    crd_enddate_by_active_status.rename(columns=lambda x: 'DAYS_CREDIT_ENDDATE_YRS_' + x, inplace=True)
    crd_enddate_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_enddate_by_active_status)
    
    crd_enddate_fact_by_active_status = pd.crosstab(index=bureau_years['SK_ID_CURR'], columns = bureau_years['CREDIT_ACTIVE'], values=bureau_years['DAYS_ENDDATE_FACT_YRS'], aggfunc=np.mean, dropna=False)
    crd_enddate_fact_by_active_status.rename(columns = lambda x : 'DAYS_ENDDATE_FACT_YRS_' + x, inplace=True)
    crd_enddate_fact_by_active_status = add_SK_ID_CURR_to_crossTabRes(crd_enddate_fact_by_active_status)
    
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
    
    # current debt on Credit Bureau Credit
    crd_amt_debt = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_DEBT'], aggfunc=np.mean, dropna=False)
    crd_amt_debt.rename(columns= lambda x : 'AMT_CREDIT_SUM_DEBT_' + x, inplace=True) 
    crd_amt_debt = add_SK_ID_CURR_to_crossTabRes(crd_amt_debt)

    # current credit limit  
    crd_sum_limit = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_LIMIT'], aggfunc=np.sum, dropna=False)
    crd_sum_limit.rename(columns= lambda x : 'AMT_CREDIT_SUM_LIMIT_' + x , inplace=True) 
    crd_sum_limit = add_SK_ID_CURR_to_crossTabRes(crd_sum_limit)

    crd_sum_overdue = pd.crosstab(index=bureau['SK_ID_CURR'], columns=bureau['CREDIT_ACTIVE'], values=bureau['AMT_CREDIT_SUM_OVERDUE'], aggfunc=np.sum, dropna=False)
    crd_sum_overdue.rename(columns= lambda x : 'AMT_CREDIT_SUM_OVERDUE_' + x , inplace=True) 
    crd_sum_overdue = add_SK_ID_CURR_to_crossTabRes(crd_sum_overdue)

    crd_type = pd.crosstab(index=bureau['SK_ID_CURR'], columns = bureau['CREDIT_TYPE'], dropna=False)
    crd_type = add_SK_ID_CURR_to_crossTabRes(crd_type)

    crd_day_update = bureau_years.groupby(['SK_ID_CURR'], as_index=False).agg({'DAYS_CREDIT_UPDATE_YRS':'mean'})
    crd_day_annuity = bureau.groupby(['SK_ID_CURR'], as_index=False).agg({'AMT_ANNUITY':'mean'})
    
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
        crd_day_annuity,
        bb_summarize]
    bureau_features  = reduce(lambda left,right: pd.merge(left, right, on='SK_ID_CURR'), dfs)
    #bureau_features.fillna(0, inplace=True)
    bureau_features = bureau_features.set_index('SK_ID_CURR')
     
    return bureau_features


#%%

if __name__ == '__main__':
    
    num_rows = None
    
    
    bureau_balance = import_data('../input/bureau_balance.csv')
    bureau = import_data('../input/bureau.csv')
        
    bureau_balance_features1 = burea_balance_feature(bureau_balance)
    bureau = bureau.join(bureau_balance_features1, how='left', on='SK_ID_BUREAU')
    
    aggregations = {}
    for x in list(bureau_balance_features1.columns.values): aggregations[x] =  ['min', 'max', 'mean', 'var']
    
    bureau_features1 =  bureau_features(bureau, aggregations )
    bureau_features1 = bureau_features1.set_index('SK_ID_CURR')
        
    del bureau, bureau_balance
    gc.collect()
    

    df = from_lib.application_train_test(num_rows)
    prev = from_lib.previous_applications(num_rows)
    pos = from_lib.pos_cash(num_rows)
    ins = from_lib.installments_payments(num_rows)
    cc = from_lib.credit_card_balance(num_rows)
    
    df = df.join(bureau_features1, how='left', rsuffix="_bureau", on = 'SK_ID_CURR')
    del bureau_features1; gc.collect()
    
    df = df.join(prev, how='left', on = "SK_ID_CURR")
    del prev;gc.collect()
    
    df = df.join(pos, how='left', on = 'SK_ID_CURR')
    del pos; gc.collect()
    
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins; gc.collect()
    
    df = df.join(cc, how='left', on="SK_ID_CURR")
    del cc; gc.collect()
    
        
#%%    
    




    
    
