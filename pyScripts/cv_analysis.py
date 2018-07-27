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

other_cv = pd.read_csv("../output/other_cv.csv")
#other_cv.columns = ['SK_ID_CURR']

data = pd.read_csv("../output/train_features.csv")
data = data.drop('index',axis=1)


global_cv = data[np.in1d(data.SK_ID_CURR, other_cv.SK_ID_CURR)]
#data = data[np.in1d(data.SK_ID_CURR, other_cv.SK_ID_CURR, invert=True)]


global_cv_y = global_cv.TARGET.values
global_cv_id = global_cv.SK_ID_CURR.values

global_cv = global_cv.drop('TARGET', axis=1)
global_cv= global_cv.drop('SK_ID_CURR', axis=1)
global_cv= global_cv.drop('y_kmeans', axis=1)

Dglobal_cv = xgb.DMatrix(global_cv, label=global_cv_y)

m0 = joblib.load("../output/m0.joblib.dat")
m10 = joblib.load("../output/m10.joblib.dat")
m2 = joblib.load("../output/m2.joblib.dat")
m3 = joblib.load("../output/m3.joblib.dat")
m4 = joblib.load("../output/m4.joblib.dat")
m7 = joblib.load("../output/m7.joblib.dat")

m0_pred = m0.predict(Dglobal_cv)
m10_pred = m10.predict(Dglobal_cv)
m2_pred = m2.predict(Dglobal_cv)
m3_pred = m3.predict(Dglobal_cv)
m4_pred = m4.predict(Dglobal_cv)
m7_pred = m7.predict(Dglobal_cv)

final_prob = (m0_pred + m10_pred + m2_pred + m3_pred + m4_pred + m7_pred) / 6

#final_prob = np.sqrt((np.square(m0_pred) + np.square(m10_pred) + np.square(m2_pred) + np.square(m3_pred) + np.square(m4_pred) + np.square(m7_pred) ) / 6)

global_cv_pred = final_prob > 0.5
global_cv_pred = global_cv_pred.astype(int)

confusion_matrix(global_cv_pred, global_cv_y)
print roc_auc_score(global_cv_y, final_prob)



