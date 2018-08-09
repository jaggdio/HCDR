# load data


from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.externals import joblib

import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import time
import pdb

def get_cm_tags(act, pred):
    
    cm_df = pd.DataFrame({'ACT':act, 'PRED':pred})
    cm_df['cm_tags'] = np.repeat(" ", len(act))
 
    cm_df['cm_tags'][(cm_df.ACT == 1) & (cm_df.PRED == 1)] = 'TP'
    cm_df['cm_tags'][(cm_df.ACT == 0) & (cm_df.PRED == 1)] = 'FP'

    cm_df['cm_tags'][(cm_df.ACT == 0) & (cm_df.PRED == 0)] = 'TN'
    cm_df['cm_tags'][(cm_df.ACT == 1) & (cm_df.PRED == 0)] = 'FN'
    return cm_df


other_cv = pd.read_csv("../output/other_cv.csv")
#other_cv.columns = ['SK_ID_CURR']

data = pd.read_csv("../output/train_features.csv")
data = data.drop('index',axis=1)


global_cv = data[np.in1d(data.SK_ID_CURR, other_cv.SK_ID_CURR)]
data = data[np.in1d(data.SK_ID_CURR, other_cv.SK_ID_CURR, invert=True)]

data_neg = data[data.TARGET == 0]
data_pos = data[data.TARGET == 1]

#pdb.set_trace()
# 1, 8, 5, 6, 9
# 3, 7, 5, 4, 9
data_neg.y_kmeans[(data_neg.y_kmeans == 3) | (data_neg.y_kmeans ==7) | (data_neg.y_kmeans ==5) | (data_neg.y_kmeans ==4) | (data_neg.y_kmeans ==9)] = 10

params = {
    # Parameters that we are going to tune.
    'max_depth':7,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # Other parameters
    'objective' : 'binary:logistic',
    'eval_metric' : 'auc'
}

cm_tags = pd.DataFrame()
scores = pd.DataFrame()

start_time1 = time.time()

for i in data_neg.y_kmeans.unique():
    data_neg1 = data_neg[data_neg.y_kmeans == i]

    train_data = pd.concat([data_pos, data_neg1])
    y = train_data.TARGET.values

    # drop cols
    train_data = train_data.drop('TARGET', axis=1)
    train_data = train_data.drop('y_kmeans', axis=1)
    #train_data = train_data.drop('SK_ID_CURR', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.33, random_state=42)

    DTrain = xgb.DMatrix(X_train.drop('SK_ID_CURR', axis=1), y_train)
    DTest = xgb.DMatrix(X_test.drop('SK_ID_CURR', axis=1), label=y_test)	

    #xgb.cv(params, DTrain, num_boost_round = 999, early_stopping_rounds=30, verbose_eval=2 )

    model = xgb.train(
        params,
        DTrain,
        num_boost_round=999,
        evals=[(DTest, "Test")],
        early_stopping_rounds=40, verbose_eval=2)

    model_name = "m" + str(i)
    print("################ " + model_name +  " ################")
    print("Model Name: {} and Best AUC: {:.2f} with {} rounds".format(model_name, model.best_score, model.best_iteration+1))
                
    #############
    pred_train_prob = model.predict(DTrain)
    pred_train = pred_train_prob > 0.5
    pred_train = pred_train.astype(int) 
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, pred_train).ravel()
    #print classification_report(y_train, pred_train)

    train_cm_tags = get_cm_tags(y_train, pred_train)
    train_cm_tags['Score'] = pred_train_prob
    train_cm_tags['Type'] = 'train'
    train_cm_tags['SK_ID_CURR'] = X_train.SK_ID_CURR.values
    train_cm_tags['m_name'] = model_name

    ##############
    pred_test_prob = model.predict(DTest)
    pred_test = pred_test_prob > 0.5
    pred_test = pred_test.astype(int) 
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, pred_test).ravel()

    test_cm_tags = get_cm_tags(y_test, pred_test)
    test_cm_tags['Score'] = pred_test_prob
    test_cm_tags['Type'] = 'test'
    test_cm_tags['SK_ID_CURR'] = X_test.SK_ID_CURR.values
    test_cm_tags['m_name'] = model_name

    cm_tags = cm_tags.append(pd.concat([train_cm_tags, test_cm_tags]))

    ####### Now save scores #### Pre, Rec, auc, type

    score_frame = pd.DataFrame({
		'Tyepe' : ['train', 'test'],
		'auc' : [roc_auc_score(y_train, pred_train), roc_auc_score(y_test, pred_test_prob)],
		'Precision' : [train_tp / (train_fp + train_tp), test_tp / (test_fp + test_tp)],
		'Recall' : [train_tp / (train_fn + train_tp), test_tp / (test_fn + test_tp)]
		})
    score_frame['m_name'] = model_name

    scores = scores.append(score_frame)

    ## save model ##
    model_path = "../output/" + model_name+".joblib.dat"
    joblib.dump(model, model_path)
    

    print (" ## Train Report ##\n")
    print classification_report(y_train, pred_train)

    print (" ## Test Report ##\n")
    print classification_report(y_test, pred_test)

    del y_train; del pred_train; del y_test; del pred_test; del model; gc.collect()

cm_tags.to_csv("../output/cm_tags.csv", index=False)
scores.to_csv("../output/scores.csv", index=False)

print("Total time taken to complete the execution: %s" % (time.time() - start_time1))
