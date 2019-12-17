import json
import numpy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score,classification_report
from collections import defaultdict
import operator
from sklearn.metrics import recall_score
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
domain_slot_token = [
            'None',
            'restaurant',
            'attraction',
            'hotel',
            'train',
            'taxi',
            'hospital',
            'bus',
            'leaveat',
            'food',
            'area',
            'type',
            'arriveby',
            'internet',
            'destination',
            'name',
            'book time',
            'book people',
            'day',
            'book day',
            'book stay',
            'pricerange',
            'departure',
            'parking',
            'stars',
            'department']
def accuracyF1(scores, labels,mode = 'domain',report = False):
    if report == False:
        if mode == 'domain':
            predicts = np.argmax(scores, axis=1)
            F1_domain = f1_score(labels, predicts, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
            return F1_domain
        elif mode == 'dependcy':
            predicts = np.argmax(scores, axis=1)
            F1_dependcy = f1_score(labels, predicts, labels=[0, 1, 2, 3], average='macro')
            return F1_dependcy
        else:
            return None
    else:
        if mode == 'domain':

            target_names = ['none','restaurant', 'attraction', 'hotel', 'train', 'taxi', 'hospital', 'bus']
            predicts = np.argmax(scores, axis=1)
            F1_domain = f1_score(labels, predicts, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')

            print(classification_report(
                labels,
                predicts,
                labels=[0,1,2,3,4,5,6,7],
                target_names=target_names, digits = 3))

            return F1_domain
        elif mode == 'dependcy':

            target_names = ['none','domain', 'slot', 'value']
            predicts = np.argmax(scores, axis=1)
            F1_dependcy = f1_score(labels, predicts, labels=[0, 1, 2, 3], average='macro')
            print(classification_report(
                labels,
                predicts,
                labels=[0,1,2,3],
                target_names=target_names, digits = 3))
            return F1_dependcy
        else:
            return None

def compute_jointGoal_domainslot_(
        dialogueID,
        utterance_text,
        scores_domainslot,
        gold_labels_domainslot,
        scores_domain,
        gold_labels_domain,
        scores_dependcy,
        gold_labels_dependcy):
    print(len(dialogueID))
    print(len(utterance_text))
    print(scores_domain.shape)
    print(gold_labels_domain.shape[0])
    print(scores_dependcy.shape)
    print(gold_labels_dependcy.shape[0])
    scores_domain = scores_domain.reshape(len(dialogueID),-1,scores_domain.shape[-1])
    scores_dependcy = scores_dependcy.reshape(len(dialogueID),-1,scores_dependcy.shape[-1])
    gold_labels_domain = gold_labels_domain.reshape(len(dialogueID),-1)
    gold_labels_dependcy = gold_labels_dependcy.reshape(len(dialogueID),-1)
    # print(scores_domain.shape)
    # print(scores_dependcy.shape)
    # print(scores_domainslot.shape)
    # print(gold_labels_domainslot.shape)

    TP, TN, FN, FP = 0, 0, 0, 0
    TP = ((scores_domainslot > 0).astype('long') & (gold_labels_domainslot > 0).astype('long')).sum()
    TN = ((scores_domainslot == 0).astype('long') & (gold_labels_domainslot == 0).astype('long')).sum()
    FN = ((scores_domainslot == 0).astype('long') & (gold_labels_domainslot > 0).astype('long')).sum()
    FP = ((scores_domainslot > 0).astype('long') & (gold_labels_domainslot == 0).astype('long')).sum()

    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    F1 = 2 * precision * recall / (precision + recall + 0.001)
    assert len(dialogueID)==len(utterance_text) == scores_domain.shape[0] == scores_dependcy.shape[0] == gold_labels_dependcy.shape[0] == gold_labels_domain.shape[0]
    assert len(dialogueID) == scores_domainslot.shape[0] == gold_labels_domainslot.shape[0]
    predicts_domain = np.argmax(scores_domain, axis=-1)
    predicts_dependcy = np.argmax(scores_dependcy, axis=-1)
    count = 0
    for each_dialogID,each_utterance_text,each_predict_domainslot,each_predicts_domain,each_predicts_dependcy in zip(
            dialogueID,
            utterance_text,
            scores_domainslot,
            predicts_domain,
            predicts_dependcy):
        count += 1
        if count == 4 :break
        each_utterance_text = each_utterance_text.split(' ')
        each_predicts_domain = each_predicts_domain[1:len(each_utterance_text)+1]
        each_predicts_dependcy = each_predicts_dependcy[1:len(each_utterance_text)+1]
        print(each_dialogID)
        print(each_utterance_text)
        print(each_predict_domainslot)
        print(each_predicts_domain)
        print(each_predicts_dependcy)
        each_result = {}
        domain = []
        slot = []
        value = []
        for index,eachword in enumerate(each_utterance_text):
            if each_predicts_dependcy[index] == 1:
                domain.append(each_utterance_text[index])
            elif each_predicts_dependcy[index] == 2:
                slot.append(each_utterance_text[index])

            elif each_predicts_dependcy[index] == 3:
                value.append(each_utterance_text[index])
        print(domain)
        print(slot)
        print(value)
        # exit()
    # exit()
    return F1

def compute_jointGoal_domainslot__(
        dialogueID,
        utterance_text,
        scores_tokens_start,
        scores_tokens_end,
        scores_sentence_domainslot,
        scores_tokens_domainslot,
        gold_labels_tokens_start,
        gold_labels_tokens_end,
        gold_label_sentence_domainslot,
        gold_label_tokens_domainslot):
    # print(scores_tokens_domainslot)
    # # print(gold_label_tokens_domainslot)
    # print(scores_tokens_domainslot.shape)
    # print(gold_label_tokens_domainslot.shape)
    # exit()
    for each in utterance_text:
        print(len(each),each)
    test_num,utterance_len,domainslot_num = gold_label_tokens_domainslot.shape
    print(test_num,utterance_len,domainslot_num)
    print(scores_tokens_start.shape)
    print(scores_tokens_end.shape)
    # exit()
    result = defaultdict[list]
    for each_test in range(0,test_num):

        for each_word in range(0, utterance_len):
            if scores_tokens_start[each_test][each_word] == 1:
                for j in range(each_word+1,utterance_len):
                    if scores_tokens_end[each_test][j] == 1:
                        value = utterance_text[each_test][each_word:j]
                        for each_domainslot in range(len(scores_sentence_domainslot[0])):
                            if scores_sentence_domainslot[test_num][each_domainslot] == 1:
                                if each_domainslot<8:
                                    domain = domain_slot_token[each_domainslot]
                                else:
                                    slot = domain_slot_token[each_domainslot]






    return 0.1


def compute_jointGoal_domainslot_1_(
        dialogueID,
        utterance_text,
        scores_tokens_start,
        scores_tokens_end,
        scores_sentence_domainslot,
        scores_tokens_domainslot,
        gold_labels_tokens_start,
        gold_labels_tokens_end,
        gold_label_sentence_domainslot,
        gold_label_token_domainslot):

    # print(scores_tokens_start.shape)
    # print(scores_tokens_end.shape)
    # exit()
    # print(gold_labels_tokens_start.shape)
    predicts_tokenstart = np.argmax(scores_tokens_start, axis=1)
    F1_tokenstart = f1_score(gold_labels_tokens_start, predicts_tokenstart, labels=[0, 1], average='macro')
    predicts_tokensend = np.argmax(scores_tokens_end, axis=1)
    F1_tokenend = f1_score(gold_labels_tokens_end, predicts_tokensend, labels=[0, 1], average='macro')
    scores_sentence_domainslot = np.argmax(scores_sentence_domainslot, axis=1)

    predicts_tokenstart = predicts_tokenstart.reshape(len(dialogueID),-1)
    gold_labels_tokens_start = gold_labels_tokens_start.reshape(len(dialogueID),-1)
    predicts_tokensend = predicts_tokensend.reshape(len(dialogueID),-1)
    gold_labels_tokens_end = gold_labels_tokens_end.reshape(len(dialogueID),-1)
    scores_sentence_domainslot = scores_sentence_domainslot.reshape(len(dialogueID),-1)
    gold_label_sentence_domainslot = gold_label_sentence_domainslot.reshape(len(dialogueID),-1)

    acc_tokenstart,acc_tokenend ,num_tokenstart,num_tokensend = 0,0,0,0
    for index in range(predicts_tokenstart.reshape(len(dialogueID),-1).shape[0]):
        if (predicts_tokenstart[index] == gold_labels_tokens_start[index]).all() and gold_labels_tokens_start[index].sum()>0:
            acc_tokenstart += 1
            num_tokenstart += 1
        else:
            num_tokenstart += 1

        if (predicts_tokensend[index].all() == gold_labels_tokens_end[index]).all() and gold_labels_tokens_end[index].sum()>0:
            acc_tokenend += 1
            num_tokensend += 1
        else:
            num_tokensend += 1


    F1_tokenstart = float(acc_tokenstart/num_tokenstart)
    F1_tokenend = float(acc_tokenstart/num_tokensend)

    # print(scores_sentence_domainslot.shape)
    # print(gold_label_sentence_domainslot.shape)
    acc_sentence_domainslot,num_sentence_domainslot = 0,0
    for index in range(scores_sentence_domainslot.shape[0]):
        if (scores_sentence_domainslot[index] == gold_label_sentence_domainslot[index]).all():
            acc_sentence_domainslot += 1
            num_sentence_domainslot += 1
        else:
            num_sentence_domainslot += 1
    F1_sentence_domainslot = float(acc_sentence_domainslot/num_sentence_domainslot)

    # scores_tokens_domainslot = scores_tokens_domainslot.reshape(len(dialogueID),-1,)
    # gold_label_tokens_domainslot = gold_label_tokens_domainslot.reshape(len(dialogueID),-1)

    scores_tokens_domainslot = np.argmax(scores_tokens_domainslot, axis=1)
    scores_tokens_domainslot = scores_tokens_domainslot.reshape(len(dialogueID),-1,)
    gold_label_token_domainslot = gold_label_token_domainslot.reshape(len(dialogueID),-1,)
    acc_token_domainslot,num_token_domainslot = 0,0
    for index in range(len(dialogueID)):
        if (scores_tokens_domainslot[index] == gold_label_token_domainslot[index]).all():
            acc_token_domainslot += 1
            num_token_domainslot += 1
        else:
            num_token_domainslot += 1
    F1_token_domainslot = float(acc_token_domainslot/num_token_domainslot)








    return F1_tokenstart,F1_tokenend,F1_sentence_domainslot,F1_token_domainslot

def compute_jointGoal_domainslot(
        dialogueID,
        utterance_text,
        scores_value_start,
        scores_value_end,
        scores_domainslot,
        gold_labels_value_start,
        gold_labels_value_end,
        gold_label_domainslot):

    # print(scores_tokens_start.shape)
    # print(scores_tokens_end.shape)
    result = defaultdict(set)
    truth = defaultdict(set)
    # print(dialogueID)
    # exit()

    predict_value_start = np.argmax(scores_value_start, axis=-1)
    predict_value_end = np.argmax(scores_value_end, axis=-1)
    predict_domainslot = np.argmax(scores_domainslot, axis=-1)
    # print(predict_value_start.shape)
    max_seq_len = predict_value_start.shape[1]
    # exit()
    # print(predict_domainslot.shape)
    # exit()

    scores_value_start = scores_value_start.tolist()
    scores_value_end = scores_value_end.tolist()
    scores_domainslot = scores_domainslot.tolist()
    predict_value_start = predict_value_start.tolist()
    predict_value_end = predict_value_end.tolist()
    predict_domainslot = predict_domainslot.tolist()
    gold_labels_value_start = gold_labels_value_start.tolist()
    gold_labels_value_end = gold_labels_value_end.tolist()
    gold_label_domainslot = gold_label_domainslot.tolist()
    for index in range(len(dialogueID)):

        dialog_id = dialogueID[index].split('#')[0]
        domain = dialogueID[index].split('#')[1]
        slot = dialogueID[index].split('#')[2]

        if predict_domainslot[index] == 1:
            value = 'dontcare'
            result[dialog_id].add((domain,slot,value))
        elif predict_domainslot[index] == 0:
            for word_start_index in range(max_seq_len):
                if predict_value_start[index][word_start_index] == 1:
                    for word_end_index in range(word_start_index,max_seq_len):
                        if predict_value_end[index][word_end_index] == 1:
                            utt_tokens = ['[CLS]']+ utterance_text[index].split(' ')
                            value = utt_tokens[word_start_index:word_end_index]
                            result[dialog_id].add((domain, slot, ' '.join(value)))

                # value = 'none'
                # result[dialog_id].add((domain,slot,value))
            dialogue,turn_id = dialog_id.split('_')
            if int(turn_id) > 0:
                turn_id = int(turn_id) -1
                last_dialog_id = '_'.join((dialogue,str(turn_id)))
                # print(dialog_id)
                # print(last_dialog_id)
                #
                # print(truth[dialog_id])
                # print(truth[last_dialog_id])

                result[dialog_id] = result[dialog_id].union(result[last_dialog_id])

        # if gold_label_domainslot[index] == 2:
        #     pass
        if gold_label_domainslot[index] == 1:
            value = 'dontcare'
            truth[dialog_id].add((domain, slot, value))
        elif gold_label_domainslot[index] == 0:
            for word_start_index in range(max_seq_len):
                if gold_labels_value_start[index][word_start_index] == 1:
                    for word_end_index in range(word_start_index, max_seq_len):
                        if gold_labels_value_end[index][word_end_index] == 1:
                            utt_tokens = utterance_text[index].split(' ')
                            value = utt_tokens[word_start_index-1:word_end_index]
                            truth[dialog_id].add((domain, slot, ' '.join(value)))
            dialogue,turn_id = dialog_id.split('_')
            if int(turn_id) > 0:
                turn_id = int(turn_id) -1
                last_dialog_id = '_'.join((dialogue,str(turn_id)))
                truth[dialog_id] = truth[dialog_id].union(truth[last_dialog_id])

    jnt_goal, num = 0, 0
    for each_predict,each_truth in zip(result,truth):
        if result[each_predict] == truth[each_truth]:
            jnt_goal += 1
        num += 1




    jnt_goal_acc = float(jnt_goal/num)

    with open('gold.json', 'w') as f:
        json.dump(truth, f,cls=MyEncoder,indent = 4)
    with open('result.json','w') as f:
        json.dump(result,f,cls =MyEncoder,indent = 4)
    return jnt_goal_acc,0,0



