import json
import numpy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from collections import defaultdict
import operator


def obtain_TP_TN_FN_FP(pred, act, TP, TN, FN, FP, elem_wise=False):
    if isinstance(pred, torch.Tensor):
        if elem_wise:
            TP += ((pred.data == 1) & (act.data == 1)).sum(0)
            TN += ((pred.data == 0) & (act.data == 0)).sum(0)
            FN += ((pred.data == 0) & (act.data == 1)).sum(0)
            FP += ((pred.data == 1) & (act.data == 0)).sum(0)
        else:
            TP += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
            TN += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
            FN += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
            FP += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
        return TP, TN, FN, FP
    else:
        TP += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
        TN += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
        FN += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
        FP += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
        return TP, TN, FN, FP

def compute_MRR(scores,labels, ID1,ID2):
    # print(len(scores))
    # print(len(labels))
    # print(len(ID1),len(ID2))
    assert len(scores) == len(labels) == len(ID1) == len(ID2)

    # scores 第一列的概率值
    result = pd.DataFrame({'scores': scores[:, 1],'logit':np.argmax(scores, axis=1), 'labels': labels,'ID1':ID1,'ID2':ID2})
    result['rank']= result['scores'].groupby(result['ID1']).rank(ascending = False)
    result['rec_rank'] = result['rank'].rdiv(1)
    mrr = result[result['labels'] == 1]['rec_rank'].sum()/(result[result['labels'] == 1].shape[0])
    return mrr

def compute_MRR_CQA(scores,labels, questions):
    # print(len(scores))
    # print(len(labels))
    # print(len(ID1),len(ID2))
    assert len(scores) == len(labels) == len(questions)

    # scores 第一列的概率值
    result = pd.DataFrame({'scores': scores[:, 1],'logit':np.argmax(scores, axis=1), 'labels': labels,'questions':questions})
    result['rank']= result['scores'].groupby(result['questions']).rank(ascending = False)
    result['rec_rank'] = result['rank'].rdiv(1)
    mrr = result[result['labels'] == 1]['rec_rank'].sum()/(result[result['labels'] == 1].shape[0])
    return mrr

def compute_5R20(scores,labels,questions):
    assert len(scores) == len(labels) == len(questions)
    # 第一列的概率值
    result = pd.DataFrame({'scores': scores[:, 1],'logit':np.argmax(scores, axis=1), 'labels': labels,'questions':questions})
    result['rank']= result['scores'].groupby(result['questions']).rank(ascending = False)


    eval_5R20 = result[(result['labels']==1) & (result['rank']<=5)]['labels'].sum()/(result[result['labels'] == 1].shape[0])

    return eval_5R20


def accuracyBDCI(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs, labels=[0, 1, 2], average='macro')

def accuracyCQA(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs, labels=[0, 1], average='macro')


def is_valid_query(each_answer):
    # 计算指标的时候对答案标签的合法性进行判断避免除0
    num_pos = 0
    num_neg = 0
    for label, score in each_answer:
        if label > 0:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos > 0 and num_neg > 0:
        return True
    else:
        return False

def compute_DOUBAN(ID,scores,labels):
    MRR,num_query = 0,0
    results = defaultdict(list)
    predict = pd.DataFrame({'scores': scores[:, 1],'labels': labels,'ID':ID})
    for index, row in predict.iterrows():
        results[row[2]].append((row[1],row[0]))

    for key,value in results.items():
        if not is_valid_query(value) : continue
        num_query +=1
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        for index_, final_result in enumerate(sorted_result):
            label,scores = final_result
            if label>0:
                MRR += 1.0/(index_+1)
                break

    predict['rank']= predict['scores'].groupby(predict['ID']).rank(ascending = False)
    predict['rec_rank'] = predict['rank'].rdiv(1)
    mrr = predict[predict['labels'] == 1]['rec_rank'].sum()/(predict[predict['labels'] == 1].shape[0])

    MAP = 0
    for key ,value in results.items():
        if not is_valid_query(value): continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        num_relevant_resp = 0
        AVP = 0 # 每个文档的平均准确率
        for index_,final_result in enumerate(sorted_result):
            each_label,each_score = final_result
            if each_label > 0:
                num_relevant_resp += 1
                precision  = num_relevant_resp/(index_+1)
                AVP += precision
        AVP = AVP/num_relevant_resp
        MAP += AVP

    Precision_1 = 0
    for key, value in results.items():
        if not is_valid_query(value): continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        # 预测的label取最后概率向量里面最大的那一个作为预测结果
        label, score = sorted_result[0]
        if label > 0:
            Precision_1 += 1

    return MRR/num_query,mrr,MAP/num_query,Precision_1/num_query


