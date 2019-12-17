import json
from collections import OrderedDict
import os
import sys
sys.path.append('/home/lsy2018/TextClassification/')
from Utils.fix_label import fix_general_label_error
import pandas as pd
class MultiWOZ:
    def __init__(self,data_dir,output_dir,onlogy_path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.onlogy_path = onlogy_path
        self.init_everything()
    def init_everything(self):
        self.file_train = os.path.join(self.data_dir,'train_dials.json')
        self.file_dev = os.path.join(self.data_dir, 'dev_dials.json')
        self.file_test = os.path.join(self.data_dir, 'test_dials.json')
        self.file_onlogy = self.data_dir+self.onlogy_path
        self.ontology = json.load(open(self.onlogy_path, 'r'))


        self.DOMAIN_SLOT_VALUES = dict([(k, v) for k, v in self.ontology.items()])
        self.DOMAIN_SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in self.DOMAIN_SLOT_VALUES.keys()]
        self.SLOT_GATE = {"ptr": 0, "dontcare": 1, "none": 2}

        self.domainslot = {}

        # self.DOMAINS = list(set([x.split('-')[0] for x in self.DOMAIN_SLOTS]))
        self.DOMAINS = {'none':0,'restaurant':1, 'attraction':2, 'hotel':3, 'train':4, 'taxi':5, 'hospital':6, 'bus':7}

    def load_data(self):

        self.read_examples(self.file_train, "train",training=True)
        self.read_examples(self.file_dev, "dev")
        self.read_examples(self.file_test, "test")


    def generate_label(self,word,turn_belief_list,mode = 'domain'):

        if mode == 'domain':
            if len(word) < 3:
                return 0
            for each in turn_belief_list:
                each_belief = each.split('-')
                if word in each_belief[1] or word in each_belief[2]:
                    domainslot = '-'.join(each_belief[:2])
                    if domainslot in self.domainslot:
                        return self.domainslot[domainslot]
                    else:
                        self.domainslot[domainslot] = len(self.domainslot)+1
                        return self.domainslot[domainslot]
            else:
                return 0

        else:
            if len(word) < 3:
                return 0
            for each in turn_belief_list:
                each_belief = each.split('-')
                if word in each_belief[2]:
                    return 3
            for each in turn_belief_list:
                each_belief = each.split('-')
                if word in each_belief[1]:
                    return 2
            for each in turn_belief_list:
                each_belief = each.split('-')
                if word in each_belief[0]:
                    return 1
            else:
                return 0


    def read_examples(self,file_name, dataset,training=False):
        print(("Reading from {}".format(file_name)))
        print(file_name)
        max_hist_len = 0
        # 记录领域的数量
        domain_counter = {}
        with open(file_name,encoding='utf-8') as f:
            dials = json.load(f)
        with open(os.path.join(self.output_dir, '{}.json'.format(dataset)), 'a', encoding='utf-8') as fw:

            for dia_index,dial_dict in enumerate(dials):
                dialog_history = ""
                for domain in dial_dict["domains"]:
                    if domain not in domain_counter.keys():
                        domain_counter[domain] = 0
                    domain_counter[domain] += 1

                # Reading data
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    # turn_domain : hotel
                    turn_domain = turn["domain"]
                    turn_id = turn["turn_idx"]

                    # turn_uttr :  ; am looking for a place to to stay that has cheap price range it should be in a type of hotel
                    turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                    turn_uttr = turn_uttr.strip()


                    dialog_history += (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                    dialog_history = dialog_history.strip()
                    turn_belief_dict = fix_general_label_error(turn["belief_state"], False, self.DOMAIN_SLOTS)


                    # OrderedDict([('hotel-pricerange', 'cheap'), ('hotel-type', 'hotel')])
                    # ['hotel-pricerange-cheap', 'hotel-type-hotel']
                    turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items()])
                    turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                    eachturn_token = turn_uttr.split(' ')
                    label_domain = [int(self.generate_label(x,turn_belief_list,'domain')) for x in eachturn_token]
                    label_dependcy = [int(self.generate_label(x,turn_belief_list,'dependcy')) for x in eachturn_token]
                    eachturn_data = {
                        'DialogTurn_id': str(dial_dict["dialogue_idx"].split('.')[0])+'_'+str(turn_id),
                        'eachturn': turn_uttr,
                        'history': dialog_history,
                        'turn_belief': turn_belief_list,
                        'label_domain': label_domain,
                        'label_dependcy':label_dependcy
                    }
                    print(eachturn_data)
                    json.dump(eachturn_data, fw, ensure_ascii=False)
                    fw.writelines('\n')


if __name__ == "__main__":
    trainer = MultiWOZ(
        # data_dir='/home/lsy2018/DST/data',
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ',
        output_dir = '/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ/data_1132/',
        onlogy_path= '/home/lsy2018/DST/data/multi-woz/MULTIWOZ2 2/ontology.json'
    )
    trainer.load_data()
    for index ,each in enumerate(trainer.domainslot):
        print(index,each,trainer.domainslot[each])



    # a = {
    #     "DialogTurn_id": "SNG0889_4",
    #     "eachturn": "yes , they do . would you like to book a reservation ? ; not right now . in fact , that s all the info i needed . thanks for your help !",
    #     "history": "; i need a hotel on the west side . ;there are 4 place -s in the west area . 2 are listed as guesthouses and 2 are hotel -s . would you like more information ? ; can you tell which ones have 4 stars and would be moderate -ly priced ? ;i am sorry , but there are no hotel -s in the west part of town that have 4 stars and are moderate -ly priced . can i look for something different for you ? ; yes , how about moderate -ly priced place -s to stay with 4 stars in the north ? i also need free parking . ;there are 6 moderate -ly priced 4 star place -s in the north which offer free parking . the acorn guest house is a popular choice . ; that sounds fine . do they have internet access ? ;yes , they do . would you like to book a reservation ? ; not right now . in fact , that s all the info i needed . thanks for your help ! ;",
    #     "turn_belief": ["hotel-area-north", "hotel-parking-yes", "hotel-pricerange-moderate", "hotel-stars-4"],
    #     "label_domain": [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     "label_dependcy": [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}






