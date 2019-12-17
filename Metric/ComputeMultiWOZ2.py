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
    max_seq_len = predict_value_start.shape[1]


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



