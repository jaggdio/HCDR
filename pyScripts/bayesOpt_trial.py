# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:46:33 2018

@author: jayphate
"""
#%%

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_iris 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

#%%

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
    clf_name = opt.estimator.__class__.__name__
    #all_models.to_csv(clf_name+"_cv_results.csv")
    
#%%
X, y = load_iris(True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

#%%

paras = { 'C': Real(1e-6, 1e+6, prior='log-uniform'), 
         'gamma': Real(1e-6, 1e+1, prior='log-uniform'), 
         'degree': Integer(1,8), 
         'kernel': Categorical(['linear', 'poly', 'rbf']), }
                    
#%%

opt = BayesSearchCV( SVC(), paras , n_iter=10,  scoring = 'accuracy', verbose=0, return_train_score=True )
opt.fit(X_train, y_train, callback=status_print)

#%%

from skopt import BayesSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#%%

# prep some sample data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1234)

#%%

# we're using a logistic regression model
clf = LogisticRegression(random_state=1234, verbose=0)

# this is our parameter grid
param_grid = {
    'solver': ['liblinear', 'saga'],  
    'penalty': ['l1','l2'],
    'tol': (1e-5, 1e-3, 'log-uniform'),
    'C': (1e-5, 100, 'log-uniform'),
    'fit_intercept': [True, False]
}

# set up our optimiser to find the best params in 30 searches
opt = BayesSearchCV(
    clf,
    param_grid,
    n_iter=30,
    random_state=1234,
    verbose=1,
    scoring = 'accuracy'
)

#%%

opt.fit(X_train, y_train, callback=status_print)

#%%

estimator = LGBMClassifier(
        #n_jobs = 1,
        #objective = 'binary:logistic',
        #eval_metric = 'auc',
        #silent=1,
        #tree_method='approx'
        objective='binary',
        metric='auc'
    )
#%%
    
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
    
#%%
    
opt = BayesSearchCV(
    estimator,
    search_spaces,
    n_iter=15,
    random_state=1234,
    verbose=0
    #scoring = 'accuracy'
)

#%%

opt.fit(X_train, y_train, callback=status_print)
    