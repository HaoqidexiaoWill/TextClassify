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



    def load_data(self):

        self.read_examples(self.file_train, "train",training=True)
        self.read_examples(self.file_dev, "dev")
        self.read_examples(self.file_test, "test")


    def label_value(self,tokens_context,value):
        label_tokens_start = [0]*len(tokens_context)
        label_tokens_end = [0]*len(tokens_context)

        for index ,each_token in enumerate(tokens_context):

            value_start = value.split(' ')[0]
            value_end = value.split(' ')[-1]

            if each_token == value_start:
                label_tokens_start[index] = 1

            if each_token == value_end:
                label_tokens_end[index] = 1


        return label_tokens_start,label_tokens_end





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
                if dia_index == 10:
                    break
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

                    eachdiahist_token = dialog_history.split(' ')
                    eachturn_token = turn_uttr.split(' ')
                    for each in self.DOMAIN_SLOTS:
                        # print(each)
                        # print(turn_belief_dict)
                        domain = each.split('-')[0]
                        slot = each.split('-')[1]
                        if each in turn_belief_dict:
                            # print(turn_belief_dict[each])
                            value = turn_belief_dict[each]
                            if value == 'dontcare':
                                label_domainslot = self.SLOT_GATE['dontcare']
                                label_value_start,label_value_end = self.label_value(eachturn_token,value)
                            else:
                                label_domainslot = self.SLOT_GATE['ptr']
                                label_value_start,label_value_end = self.label_value(eachturn_token,value)
                        else:
                            label_domainslot = self.SLOT_GATE['none']
                            label_value_start = [0]*len(eachturn_token)
                            label_value_end = [0]*len(eachturn_token)


                        eachturn_data = {
                            'DialogTurn_id': str(dial_dict["dialogue_idx"].split('.')[0])+'_'+str(turn_id),
                            'eachturn': turn_uttr,
                            'history': dialog_history,
                            'domain':domain,
                            'slot':slot,
                            'label_value_start':label_value_start,
                            'label_value_end': label_value_end,
                            'label_domainslot':label_domainslot

                        }
                        print(eachturn_data)
                        json.dump(eachturn_data, fw, ensure_ascii=False)
                        fw.writelines('\n')


if __name__ == "__main__":
    trainer = MultiWOZ(
        # data_dir='/home/lsy2018/DST/data',
        data_dir='/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ',
        output_dir = '/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ/data_1203_/',
        onlogy_path= '/home/lsy2018/DST/data/multi-woz/MULTIWOZ2 2/ontology.json'
    )
    trainer.load_data()
    # slot = []
    # for index ,each in enumerate(trainer.domainslot):
    #     print(index,each,trainer.domainslot[each])
    #     slot.append(each.split('-')[1])
    # slot = list(set(slot))
    # slot_ = {each:i for i,each in enumerate(slot)}
    # print(slot_)









