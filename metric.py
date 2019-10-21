import json
import numpy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score


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
