import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score
# 读取训练集数据

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
def load_data(data_dir = '/home/lsy2018/TextClassification/Utils/BDCI模型融合数据/SJT_lgbm/'):
    train = []
    for fold_id in range(5):
        # print('fold_id:',fold_id)
        for num, model_id in enumerate(range(1, 4)):
            # print('model_id:',model_id)
            all_fold_id = list(range(5))
            all_fold_id.remove(fold_id)
            folder = "%d_%d" % (model_id, fold_id)
            if (num == 0):
                train_kf = pd.read_csv(data_dir + folder + "/dev.csv")
                train_kf.columns = ['id', 'm%d_label0' % model_id, 'm%d_label1' % model_id, 'm%d_label2' % model_id,
                                    'm%d_label' % model_id]
            else:
                tmp = pd.read_csv(data_dir + folder + "/dev.csv")
                tmp.columns = ['id', 'm%d_label0' % model_id, 'm%d_label1' % model_id, 'm%d_label2' % model_id,
                               'm%d_label' % model_id]
                train_kf = train_kf.merge(tmp, how='left', on='id')
        # print(train_kf)
        train.append(train_kf)
    train = pd.concat(train, axis=0, sort=True)  # 为啥之前concat index就是自动增
    train = train.reset_index(drop=True)
    print('TRAIN数据格式以及缺失情况统计')
    print(train.head())
    print(train.describe())
    print(train.shape)
    print(train.isnull().sum())

    test_list = []
    for fold_id in range(5):
        # print('fold_id:',fold_id)
        for num, model_id in enumerate(range(1, 4)):
            # print('model_id:',model_id)
            all_fold_id = list(range(5))
            all_fold_id.remove(fold_id)
            folder = "%d_%d" % (model_id, fold_id)
            if (num == 0):
                test_kf = pd.read_csv(data_dir+folder + "/test.csv")
                test_kf.columns = ['id', 'm%d_label0' % model_id, 'm%d_label1' % model_id, 'm%d_label2' % model_id,
                                   'm%d_label' % model_id]
            else:
                tmp = pd.read_csv(data_dir+folder + "/test.csv")
                tmp.columns = ['id', 'm%d_label0' % model_id, 'm%d_label1' % model_id, 'm%d_label2' % model_id,
                               'm%d_label' % model_id]
                test_kf = test_kf.merge(tmp, how='left', on='id')
        #     print(test_kf)
        #     input('c')
        test_kf = test_kf.drop(['m1_label', 'm2_label', 'm3_label'], axis=1)
        test_list.append(test_kf)

    X_test_list = []
    for test in test_list:
        X_test = test.drop(['id'], axis=1).values
        X_test_list.append(X_test)
        print(X_test.shape)
    for num, X_test in enumerate(X_test_list):
        if (num == 0):
            sum = X_test
        else:
            sum += X_test
    sum /= 5
    sum, sum.shape
    X_test = sum
    print('TEST  INFO')
    print(X_test.shape)

    return train,X_test
train_data,x_test = load_data()



def trainLGB(train_data,x_test):
    print('Load Train Data.')
    y_train = train_data['m1_label'].values  # 只选择训练集 用:train.shape分割
    X_train = train_data.drop(['id', 'm1_label', 'm2_label', 'm3_label'], axis=1).values
    print(X_train.shape, y_train.shape)
    n_splits =5 # Number of K-fold Splits
    oof = np.zeros((train_data.shape[0],3))
    predictions = np.zeros((x_test.shape[0],3))
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(X_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {i + 1}')
        x_train = np.array(X_train)
        y_train = np.array(y_train)
        trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
        val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
        num_round = 15000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=10000,
                        early_stopping_rounds=1000)
        oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
        predictions += clf.predict(x_test.astype(int), num_iteration=clf.best_iteration) / 5


    print("CV score: {:<8.5f}".format(f1_score(y_train, np.argmax(oof,axis=1),labels=[0, 1, 2], average='macro')))
    test_data = pd.read_csv('/home/lsy2018/TextClassification/DATA/DATA_BDCI/Test_DataSet.csv')
    id = test_data['id']
    submission = pd.DataFrame({"id": id, "label": np.argmax(predictions,axis=1)})
    submission.to_csv('submission_stacking.csv', index=False, header=True)


    return submission

trainLGB(train_data,x_test)