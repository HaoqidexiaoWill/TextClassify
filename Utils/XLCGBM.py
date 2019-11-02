import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
gc.enable()
def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=100000,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               # nthread = 15,
                               metric='multi_logloss',
                               objective='multiclass',
                               n_jobs=15)

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)

    # Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter + 1)
    model.booster_.save_model(save_to)

    return cv_val


def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=100000,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='multi：softprob',
                              n_jobs=20)

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)

    # Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter + 1)
    pickle.dump(model, open(save_to, "wb"))

    return cv_val


def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    model = cb.CatBoostClassifier(iterations=100000,
                                  max_depth=2,
                                  learning_rate=0.02,
                                  colsample_bylevel=0.03,
                                  objective="MultiClass")

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0, early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)

    # Save Catboost Model
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter + 1)
    model.save_model(save_to, format="coreml",
                     export_parameters={'prediction_type': 'probability'})

    return cv_val