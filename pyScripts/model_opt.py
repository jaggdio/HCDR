
from functools import reduce

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from lightgbm import LGBMClassifier

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


if __name__ == '__main__':

	with open('../output/features.pkl', 'rb') as fp:
       df = pickle.load(fp)

    print('Reading Done')
       
	train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    
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
   
    y_train = train_df.TARGET.values
    X_train = train_df.drop('TARGET', axis=1)

    print(' ## Training ##')
    opt.fit(X_train, y_train, callback=status_print)

