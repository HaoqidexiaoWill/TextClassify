import json

import torch
from Utils.config import *
from collections import OrderedDict
from collections import defaultdict
import os
import pickle
from Utils.fix_label import *

class MultiWOZDATA:
    def __init__(self,data_dir,output_dir,onlogy_path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.onlogy_path = onlogy_path

        self.init_everything()
    def init_everything(self):
        self.file_train = os.path.join(self.data_dir,'train_dials.json')
        self.file_dev = os.path.join(self.data_dir, 'dev_dials.json')
        self.file_test = os.path.join(self.data_dir, 'test_dials.json')
        self.file_onlogy = self.data_dir+self.onlogy_path
        self.ontology = json.load(open(self.file_onlogy, 'r'))


        self.DOMAIN_SLOT_VALUES = dict([(k, v) for k, v in self.ontology.items()])
        self.DOMAIN_SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in self.DOMAIN_SLOT_VALUES.keys()]
        self.SLOT_GATE = {"ptr": 0, "dontcare": 1, "none": 2}
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)


    def read_examples(self,file_name, dataset,training=False):
        print(("Reading from {}".format(file_name)))
        data = []
        max_hist_len = 0
        # 记录领域的数量
        domain_counter = {}
        with open(file_name) as f:
            dials = json.load(f)

            for dia_index,dial_dict in enumerate(dials):
                if dia_index ==60:
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


                    class_label, generate_value, slot_mask, gating_label = [], [], [], []
                    for slot in self.DOMAIN_SLOTS:
                        if slot in turn_belief_dict.keys():
                            generate_value.append(turn_belief_dict[slot])

                            if turn_belief_dict[slot] == "dontcare":
                                gating_label.append(self.SLOT_GATE["dontcare"])
                            elif turn_belief_dict[slot] == "none":
                                gating_label.append(self.SLOT_GATE["none"])
                            else:
                                gating_label.append(self.SLOT_GATE["ptr"])

                        else:
                            generate_value.append("none")
                            gating_label.append(self.SLOT_GATE["none"])

                    '''
                    generate_value , gating_label
                    ['cheap', 'hotel', 'none', 'none', 'none','none']
                    [0, 0, 2, 2, 2, 2]
                     '''


                    data_detail = {
                        "ID": dial_dict["dialogue_idx"],
                        "domains": dial_dict["domains"],
                        "turn_domain": turn_domain,
                        "turn_id": turn_id,
                        "dialog_history": dialog_history,
                        "turn_belief": turn_belief_list,
                        "gating_label": gating_label,
                        "turn_uttr": turn_uttr,
                        'generate_value': generate_value
                    }
                    if max_hist_len < len(dialog_history.split()):
                        max_hist_len = len(dialog_history.split())
                    else:
                        continue

        return data, max_hist_len


    def load_data(self,training, batch_size):

        train_examples, train_max_hist_len = self.read_examples(self.file_train,'train')
        dev_examples, dev_max_hist_len,= self.read_examples(self.file_dev, "dev")
        test_examples, test_max_hist_len = self.read_examples(self.file_test, "test")



if __name__ == "__main__":
    a = MultiWOZDATA(
        data_dir = '/home/lsy2018/graphDialog/Utils/trade-dst-master/data',
        output_dir = '/home/lsy2018/graphDialog/Utils/trade-dst-master/data',
        onlogy_path = '/multi-woz/MULTIWOZ2 2/ontology.json'
    a.load_data()