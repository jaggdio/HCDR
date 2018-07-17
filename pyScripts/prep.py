# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:12:20 2018

@author: jayphate
"""
#%%

from functools import reduce

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


#%%

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframodel = BayesSearchCV(
    estimator=xgb.XGBClassifier(
        n_jobs=4,
        objective='binary:logistic',
        eval_metric='auc',
        silent=1
    ),
    search_spaces=search_spaces,
    scoring='roc_auc',
    cv=StratifiedKFold(
        n_splits=3,
        shuffle=False,
        random_state=42
    ),
    n_jobs=3,
    n_iter=ITERATIONS,
    verbose=1,
    refit=True,
    random_state=42
)me and modify the data type
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
    #bureau_features = bureau_features.set_index('SK_ID_CURR')
     
    return bureau_features


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(opt.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(opt.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(opt.best_score_, 4),
        opt.best_params_
    ))
    
    # Save all model results
    # clf_name = opt.estimator.__class__.__name__
    # all_models.to_csv(clf_name+"_cv_results.csv")


def BayesSearchCV_optimisation(data):

    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    }    

    estimator = LGBMClassifier(
        objective='binary',
        metric='auc'
    )

    opt = BayesSearchCV(
    estimator,
    search_spaces,
    n_iter=100,
    random_state=1234,
    verbose=0
    #scoring = 'accuracy'
    )

    opt.fit(X_train, y_train, callback=status_print)


def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
#==============================================================================
#             learning_rate=0.02,
#             num_leaves=34,
#             colsample_bytree=0.9497036,
#             subsample=0.8715623,
#             subsample_freq=1,
#             max_depth=8,
#             reg_alpha=0.041545473,
#             reg_lambda=0.0735294,
#             min_split_gain=0.0222415,
#             min_child_weight=39.3259775,
#             random_state=0,
#             silent=-1,
#             verbose=-1,
#             'colsample_bylevel': 0.417322896138908,
#==============================================================================
             colsample_bytree= 0.9051691017946866,
             gamma= 4.130828170307698e-08,
             learning_rate= 0.4685157092356401,
             max_delta_step= 14,
             max_depth= 48,
             min_child_weight= 2,
             #n_estimators= 81,
             reg_alpha= 0.1166276409797035,
             reg_lambda= 3.022483203428652e-07,
             scale_pos_weight= 0.004513468724073575,
             subsample= 0.06415751635420602 )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))



#%%

if __name__ == '__main__':
    
    num_rows = None

    bureau_balance = import_data('../input/bureau_balance.csv')
    bureau = import_data('../input/bureau.csv')
    
    print('## bureau_balance_features ##')    
    bureau_balance_features1 = burea_balance_feature(bureau_balance)
    bureau = bureau.join(bureau_balance_features1, how='left', on='SK_ID_BUREAU')
    
    aggregations = {}
    for x in list(bureau_balance_features1.columns.values): aggregations[x] =  ['min', 'max', 'mean', 'var']
    
    bureau_features1 =  bureau_features(bureau, aggregations )
    bureau_features1 = bureau_features1.set_index('SK_ID_CURR')
        
    del bureau, bureau_balance
    gc.collect()
    
    print('## application_train_test ##')    
    df = from_lib.application_train_test(num_rows)

    print('## previous_applications ##')    
    prev = from_lib.previous_applications(num_rows)

    print('## pos_cash ##')    
    pos = from_lib.pos_cash(num_rows)

    print('## installments_payments ##')    
    ins = from_lib.installments_payments(num_rows)

    print('## credit_card_balance ##')    
    cc = from_lib.credit_card_balance(num_rows)
    
    print('## Joining All ##')    
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
    
   # with open('../output/features.pkl', 'wb') as fp:
   #   pickle.dump(df, fp)

    #kfold_lightgbm(df, num_folds= 5, stratified= False, debug= False)
    
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    
    search_spaces1 = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    }

    search_spaces2 = {
    'colsample_bylevel': (0.1, 1.0, 'uniform'),
    'colsample_bytree': (0.1, 1.0, 'uniform'),
    'gamma': (1e-09, 0.5, 'log-uniform'),
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'max_delta_step': (0, 20),
    'max_depth': (0, 50),
    'min_child_weight': (0, 5),
    'n_estimators': (50, 500),
    'reg_alpha': (1e-09, 1.0, 'log-uniform'),
    'reg_lambda': (1e-09, 20, 'log-uniform'),
    'scale_pos_weight': (0.01, 20, 'log-uniform'),
    'subsample': (0.01, 1.0, 'uniform')} 
    
    search_spaces={
    'max_depth': [2], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
    'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],
    'n_estimators': [1000], #[1000,2000,3000]
    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]
    }
    

#==============================================================================
#     opt = BayesSearchCV(
#     estimator,
#     search_spaces,
#     n_iter=100,
#     random_state=1234,
#     verbose=0,
#     
#     cv = KFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=42)    
#     #scoring = 'accuracy'
#     )
#==============================================================================
#==============================================================================
# 
#     model = BayesSearchCV(
#     estimator=xgb.XGBClassifier(
#        ### n_jobs=4,
#         objective='binary:logistic',
#         eval_metric='auc',
#         silent=0
#     ),
#     search_spaces=search_spaces,
#     scoring='roc_auc',
#     cv=StratifiedKFold(
#         n_splits=3,
#         shuffle=False,
#         random_state=42
#     ),
#     n_jobs=1,
#     n_iter=10,
#     verbose=1,
#     refit=True,
#     random_state=42
#     )
#==============================================================================

#==============================================================================
#     model = BayesSearchCV(
#     estimator=xgb.XGBClassifier(
#         n_jobs=2,
#         objective='binary:logistic',
#         eval_metric='auc',
#         silent=1
#     ),
#     search_spaces=search_spaces,
#     scoring='roc_auc',
#     cv=StratifiedKFold(
#         n_splits=3,
#         shuffle=False,
#         random_state=42
#     ),
#     n_jobs=3,
#     n_iter=10,
#     verbose=1,
#     refit=True,
#     random_state=42
#     )
#     y_train = train_df.TARGET.values
#     X_train = train_df.drop('TARGET', axis=1)
#     
#     estimator=xgb.XGBClassifier(
#         n_jobs=2,
#         objective='binary:logistic',
#  #       eval_metric='auc',
#         silent=1,
#         verbose = 2
#     )
#==============================================================================
    
    y_train = train_df.TARGET.values
    X_train = train_df.drop('TARGET', axis=1)
    
    estimator=xgb.XGBClassifier(
        n_jobs=2,
        objective='binary:logistic',
 #       eval_metric='auc',
        silent=1)
    
    gs = GridSearchCV(
    estimator=estimator, param_grid=search_spaces, 
    scoring='roc_auc',
    cv=3,
    refit=True,
    verbose=10)
    
    print(' ## Training ##')
    
    #model.fit(X_train, y_train, callback=status_print)
    
    gs.fit(X_train, y_train)
    best_est = gs.best_estimator_
    
    pdb.set_trace()
    print(best_est)
    
        
#%%    
    
