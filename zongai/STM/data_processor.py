# -*- coding: utf-8 -*-
import json
import random
import csv
import os


folder_path = os.path.abspath(os.path.join(os.getcwd(), "./dataset/DSTC8"))
train_path = os.path.abspath(os.path.join(folder_path, 'task-2.ubuntu.train.json'))
dev_path = os.path.abspath(os.path.join(folder_path, 'task-2.ubuntu.dev.json'))
# with open(train_path, "r") as train_file:
#     train_data = json.load(train_file)[:2]
with open(dev_path, "r") as dev_file:
    dev_data = json.load(dev_file)[0]

# print(dev_data)

def Nan_handler(data):    
    if not data["options-for-correct-answers"]:
        data["options-for-correct-answers"] = [{
                "candidate-id": '0',
                "speaker": "unknown",
                "utterance": "unknown"
            }]
        data["options-for-next"].insert(0,{
                "candidate-id": '0',
                "utterance": "unknown"
            })
        data["options-for-next"].pop()
    return data

