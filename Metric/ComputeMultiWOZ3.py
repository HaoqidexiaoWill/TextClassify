import json
import numpy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score,classification_report
from collections import defaultdict
import operator
from sklearn.metrics import recall_score
ontology = json.load(open('/home/lsy2018/DST/data/multi-woz/MULTIWOZ2 2/ontology.json'
                          '', 'r'))
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

def compute_jointGoal_domainslot_1_(
        dialogueID,
        truth,
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
    predicts_sentence_domainslot = np.argmax(scores_sentence_domainslot, axis=1)
    #
    predicts_tokenstart = predicts_tokenstart.reshape(len(dialogueID),-1)
    # gold_labels_tokens_start = gold_labels_tokens_start.reshape(len(dialogueID),-1)
    predicts_tokensend = predicts_tokensend.reshape(len(dialogueID),-1)
    gold_labels_tokens_end = gold_labels_tokens_end.reshape(len(dialogueID),-1)
    predicts_sentence_domainslot = predicts_sentence_domainslot.reshape(len(dialogueID),-1)
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

    print(scores_sentence_domainslot.shape)
    print(gold_label_sentence_domainslot.shape)
    acc_sentence_domainslot,num_sentence_domainslot = 0,0
    # for index in range(scores_sentence_domainslot.shape[0]):
    #     if (predicts_sentence_domainslot[index] == gold_label_sentence_domainslot[index]).all():
    #         acc_sentence_domainslot += 1
    #         num_sentence_domainslot += 1
    #     else:
    #         num_sentence_domainslot += 1
    # F1_sentence_domainslot = float(acc_sentence_domainslot/num_sentence_domainslot)

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
    result = defaultdict(set)
    for index in range(len(dialogueID)):

        # print(gold_labels_tokens_start[index][:20])
        # print(predicts_tokenstart[index][:20])
        # print(gold_labels_tokens_end[index][:20])
        # print(predicts_tokensend[index][:20])
        # print(utterance_text[index][:20])
        # exit()
        value_list = []
        slot_list = []
        domain_list = []
        # print(predicts_tokenstart.shape)
        # print(len(utterance_text[0]))
        # exit()
        for word_start_index in range(len(predicts_tokenstart[index])-1):
            # print(word_start_index)
            # print(len(utterance_text[index])-1)
            # print(len(predicts_tokenstart[index]))
            if predicts_tokenstart[index][word_start_index] == 1:
                for word_end_index in range(word_start_index,len(predicts_tokenstart[index])-1):
                    if predicts_tokensend[index][word_end_index] == 1:
                        value = utterance_text[index][word_start_index:word_end_index+1]
                        value_list.append(value)
                        break

        # print(value_list)
        # print(predicts_sentence_domainslot[index])
        # print(gold_label_sentence_domainslot[index])
        # exit()
        for domainslot_index,each_domainslot in enumerate(predicts_sentence_domainslot[index]):
            if each_domainslot == 1:
                if domainslot_index<8:
                    domain_list.append(domain_slot_token[domainslot_index])
                else:
                    slot_list.append(domain_slot_token[domainslot_index])


        print(domain_list)
        print(slot_list)
        print(value_list)
        print(ontology.keys())

        for each_domain in domain_list:
            for each_slot in slot_list:
                for each_value in value_list:
                    
                    tmp_domainslot = '-'.join((each_domain,each_slot))
                    tmp_value = ' '.join(each_value)
                    # if tmp_domainslot in ontology and tmp_value in ontology[tmp_domainslot]:
                    if tmp_domainslot in ontology:
                        result[dialogueID[index]].add('-'.join([tmp_domainslot,tmp_value]))
        print(result)
        # exit()
        dialogid,tmp_turnid = dialogueID[index].split('_')
        if int(tmp_turnid)>0:
            tmp_last_turnid = int(tmp_turnid)-1
            last_id = '_'.join((dialogid,str(tmp_last_turnid)))
            result[dialogueID[index]] = result[dialogueID[index]].union(result[last_id])

    print(len(truth))
    print(len(result))
    print(result)
    jnt = 0
    num = 0
    for each in truth:
        print(truth[each])
        print(result[each])
        # exit()
        if set(truth[each]) == set(result[each]):
            jnt+= 1
            num += 1
        else:
            # print(set(truth[each]))
            # print(set(result[each]))
            # exit()
            num += 1
    print(float(jnt/num))
    exit()
        
        
        


    # for index in range(len(dialogueID)):

    #     # print(gold_labels_tokens_start[index][:20])
    #     # print(predicts_tokenstart[index][:20])
    #     # print(gold_labels_tokens_end[index][:20])
    #     # print(predicts_tokensend[index][:20])
    #     # print(utterance_text[index][:20])
    #     # exit()
    #     value_list = []
    #     slot_list = []
    #     domain_list = []
    #     for word_start_index in range(len(utterance_text[index])):
    #         if gold_labels_tokens_start[index][word_start_index] == 1:
    #             for word_end_index in range(word_start_index,len(utterance_text[index])):
    #                 if gold_labels_tokens_end[index][word_end_index] == 1:
    #                     value = utterance_text[index][word_start_index:word_end_index+1]
    #                     value_list.append(value)
    #                     break

    #     # print(value_list)
    #     # print(predicts_sentence_domainslot[index])
    #     # print(gold_label_sentence_domainslot[index])
    #     # exit()
    #     for domainslot_index,each_domainslot in enumerate(gold_label_sentence_domainslot[index]):
    #         if each_domainslot == 1:
    #             if domainslot_index<8:
    #                 domain_list.append(domain_slot_token[domainslot_index])
    #             else:
    #                 slot_list.append(domain_slot_token[domainslot_index])


    #     print(domain_list)
    #     print(slot_list)
    #     print(value_list)
    #     print(ontology.keys())


    #     for each_domain in domain_list:
    #         for each_slot in slot_list:
    #             for each_value in value_list:
                    
    #                 tmp_domainslot = '-'.join((each_domain,each_slot))
    #                 tmp_value = ' '.join(each_value)
    #                 if tmp_domainslot in ontology and tmp_value in ontology[tmp_domainslot]:
    #                     result[dialogueID[index]].add((tmp_domainslot,tmp_value))











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

    gold_labels_value_start = gold_labels_value_start.reshape(len(dialogueID),-1)
    gold_labels_value_end = gold_labels_value_end.reshape(len(dialogueID),-1)
    # gold_label_domainslot = gold_label_domainslot.reshape(len(dialogueID),-1)

    predict_value_start = predict_value_start.reshape(len(dialogueID),-1)
    predict_value_end = predict_value_end.reshape(len(dialogueID),-1)
    predict_domainslot = predict_domainslot.reshape(len(dialogueID),-1)


    print(gold_labels_value_start.shape)
    print(gold_labels_value_end.shape)
    print(gold_label_domainslot.shape)
    print(predict_value_start.shape)
    print(predict_value_end.shape)
    print(predict_domainslot.shape)

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

        print(gold_labels_value_start[index][:20])
        print(predict_value_start[index][:20])
        print(gold_labels_value_end[index][:20])
        print(predict_value_end[index][:20])
        print(utterance_text[index][:20])
        # exit()
        value_list = []
        slot_list = []
        domain_list = []
        for word_start_index in range(len(utterance_text[index])):
            if predict_value_start[index][word_start_index] == 1:
                for word_end_index in range(word_start_index,len(utterance_text[index])):
                    if predict_value_end[index][word_end_index] == 1:
                        value = utterance_text[index][word_start_index:word_end_index+1]
                        value_list.append(value)
                        break

        print(value_list)
        print(predict_domainslot[index])
        print(gold_label_domainslot[index])
        exit()



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

                result[dialog_id] = result[dialog_id].union(result[last_dialog_id])
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

