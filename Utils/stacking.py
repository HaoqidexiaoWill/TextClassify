import sys
import os
import random
import pickle
import pandas as pd
import  numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from XLCGBM import fit_lgb,fit_xgb,fit_cb
import gc
import copy
gc.enable()
# TODO 这届参数不行- -
'''
param = {
    'num_leaves': 10,
    'max_bin': 119,
    'min_data_in_leaf': 11,
    'learning_rate': 0.02,
    'min_sum_hessian_in_leaf': 0.00245,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,
    'feature_fraction': 0.05,
    'lambda_l1': 4.972,
    'lambda_l2': 2.276,
    'min_gain_to_split': 0.65,
    'max_depth': 14,
    'save_binary': True,
    'seed': 1337,
    'feature_fraction_seed': 1337,
    'bagging_seed': 1337,
    'drop_seed': 1337,
    'data_random_seed': 1337,
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'verbose': 1,
    'metric': 'multi_logloss',
    'is_unbalance': True,
    'boost_from_average': False,
    # 'device': 'gpu',
    'num_class': 3,
}
'''
param = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': 3,
          'max_depth': -1,
          'learning_rate': 0.002,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          "is_unbalance": True,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu'
          }
dir_path = os.getcwd()
print((dir_path))

# SJT
def get_SJTdata():
    submodel_path = os.path.join(dir_path,'BDCI模型融合数据/SJT_lgbm/')
    submodel_pathList = os.listdir(submodel_path)
    print('子模型数量',len(submodel_pathList))
    print('所有子模型的路径',submodel_pathList)
    model_name = submodel_pathList[0].split('_')[0]
    k_fold = submodel_pathList[0].split('_')[1]
    print('model_name',model_name,'k_fold',k_fold)

    train_dataDict = {}
    for each_file in submodel_pathList:
        model_name = each_file.split('_')[0]
        k_fold = each_file.split('_')[1]
        submodel_data = pd.read_csv(os.path.join(submodel_path,each_file,'dev.csv'),sep = ',')
        submodel_data.rename(columns={
        'label_0':str(model_name+'_label_0'),
        'label_1': str(model_name+'_label_1'),
        'label_2': str(model_name+'_label_2'),
    },inplace=True)
        submodel_data.drop(['label'], axis=1, inplace=True)
        if model_name not in train_dataDict:
            train_dataDict[model_name] = [submodel_data]
        else:
            train_dataDict[model_name].append(submodel_data)

    train_data = pd.read_csv('/home/lsy2018/TextClassification/DATA/DATA_BDCI/Train_DataSet_Label.csv')
    for each_model in train_dataDict:
        each_model_cat = pd.concat(train_dataDict[each_model])
        train_data = each_model_cat.merge(train_data,on = 'id',how = 'left')
    train_data.dropna(how='any', inplace=True)

    test_dataDict = {}
    test_data = pd.read_csv('/home/lsy2018/TextClassification/DATA/DATA_BDCI/Test_DataSet.csv')
    test_data.drop(['title','content'], axis=1, inplace=True)
    for each_file in submodel_pathList:
        model_name = each_file.split('_')[0]
        k_fold = each_file.split('_')[1]
        submodel_data = pd.read_csv(os.path.join(submodel_path,each_file,'test.csv'),sep = ',')
        submodel_data.rename(columns={
        'label_0':str(model_name+'_label_0'),
        'label_1': str(model_name+'_label_1'),
        'label_2': str(model_name+'_label_2'),
    },inplace=True)
        submodel_data.drop(['label'], axis=1, inplace=True)
        # test_data = test_data.merge(submodel_data,on = 'id',how = 'left')

        if k_fold not in test_dataDict:
            test_dataDict[k_fold] = [submodel_data]
        else:
            test_dataDict[k_fold].append(submodel_data)

    test_dataList = []
    for each_fold in test_dataDict:
        test_data_tmp = copy.deepcopy(test_data)
        for each_testdata in test_dataDict[each_fold]:
            test_data_tmp = test_data_tmp.merge(each_testdata,on = 'id',how = 'left')
            # test_data_tmp.dropna(how='any', inplace=True)
        print('*' * 20, 'TestINFO', '*' * 20)
        print('数据格式以及缺失情况统计')
        print(test_data_tmp.shape)
        print(test_data_tmp.isnull().sum())
        test_data_tmp.drop(['id'],axis = 1,inplace=True)
        test_data_tmp.fillna(0, inplace=True)
        print(test_data_tmp.isnull().sum())
        test_dataList.append(test_data_tmp)
        del test_data_tmp

    print('*'*20,'TrainINFO','*'*20)
    print(train_data.describe())
    print(train_data.head())
    print('train的数据格式',train_data.shape)

    return train_data,test_dataList

def trainLGB():
    print('Load Train Data.')
    train_data, test_dataList = get_SJTdata()
    train_features = train_data.drop(['id','label'],axis = 1).astype(int)
    train_target = train_data['label'].astype(int)
    n_splits =5 # Number of K-fold Splits
    oof = np.zeros((train_data.shape[0],3))
    predictions = np.zeros((test_dataList[0].shape[0],3))
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features, train_target))
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {i + 1}')
        x_train = np.array(train_features)
        y_train = np.array(train_target)
        trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
        val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
        num_round = 15000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=10000,
                        early_stopping_rounds=1000)
        oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_dataList[i].astype(int), num_iteration=clf.best_iteration) / 5


    print("CV score: {:<8.5f}".format(f1_score(train_target, np.argmax(oof,axis=1),labels=[0, 1, 2], average='macro')))
    test_data = pd.read_csv('/home/lsy2018/TextClassification/DATA/DATA_BDCI/Test_DataSet.csv')
    id = test_data['id']
    submission = pd.DataFrame({"id": id, "label": np.argmax(predictions,axis=1)})
    submission.to_csv('submission_stacking.csv', index=False, header=True)


    return submission

if __name__ == "__main__":

    # 只用LGB
    submission = trainLGB()
    print(submission.describe())
