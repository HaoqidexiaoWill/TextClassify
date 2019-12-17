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
        self.ontology = json.load(open(self.file_onlogy, 'r'))


        self.DOMAIN_SLOT_VALUES = dict([(k, v) for k, v in self.ontology.items()])
        self.DOMAIN_SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in self.DOMAIN_SLOT_VALUES.keys()]
        self.SLOT_GATE = {"ptr": 0, "dontcare": 1, "none": 2}

        # self.DOMAINS = list(set([x.split('-')[0] for x in self.DOMAIN_SLOTS]))
        self.DOMAINS = {'none':0,'restaurant':1, 'attraction':2, 'hotel':3, 'train':4, 'taxi':5, 'hospital':6, 'bus':7}

    def load_data(self):

        train_examples = self.read_examples(self.file_train, "train",training=True)
        dev_examples= self.read_examples(self.file_dev, "dev")
        test_examples  = self.read_examples(self.file_test, "test")

        print(train_examples.shape)
        train_examples.to_csv(os.path.join(self.output_dir, "train.csv"), index=False, header=False)
        dev_examples.to_csv(os.path.join(self.output_dir, "dev.csv"), index=False, header=False)
        test_examples.to_csv(os.path.join(self.output_dir, "test.csv"), index=False, header=False)


        print("Read %s examples train" % len(train_examples))
        print("Read %s examples dev" % len(dev_examples))
        print("Read %s examples test" % len(test_examples))
        print("[Slots]: Number is {} in total".format(str(len(self.DOMAIN_SLOTS))))
        print(self.DOMAIN_SLOTS)

    def generate_label(self,word,turn_belief_list,mode = 'domain'):
        return_value = 0
        if mode == 'domain':
            if len(word) < 2:
                return 0
            for each in turn_belief_list:
                each_belief = each.split('-')
                for each_beliefspan in each_belief:
                    if word in each_beliefspan:
                        return self.DOMAINS[each_belief[0]]
            else:
                return self.DOMAINS['none']

        else:
            if len(word) < 3:
                return 0
            for each in turn_belief_list:
                each_belief = each.split('-')
                if word == 'hotel':

                    print(word in each_belief[2])
                if word in each_belief[2]:

                    return 1
                elif word in each_belief[1]:
                    return 2
                elif word in each_belief[0]:
                    return 3
            else:
                return 0


    def read_examples(self,file_name, dataset,training=False):
        print(("Reading from {}".format(file_name)))
        data = []
        dialogue_idx_list = []
        turn_id_list = []
        dialogue_domains_list = []
        turn_domain_list = []
        dialog_history_list = []
        turn_belief_List = []
        turn_uttr_list = []
        domain_List = []
        slot_List = []
        value_List = []
        label = []

        print(type(dialogue_idx_list))

        max_hist_len = 0
        # 记录领域的数量
        domain_counter = {}
        with open(file_name) as f:
            dials = json.load(f)

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
                    domainlist = list(set([x.split('-')[0] for x in turn_belief_list]))
                    slotlist = list(set([x.split('-')[1] for x in turn_belief_list]))
                    valuelist = list(set([x.split('-')[2] for x in turn_belief_list]))

                    eachturn_token = turn_uttr.split(' ')
                    label_domain = [int(self.generate_label(x,turn_belief_list,'domain')) for x in eachturn_token]
                    label_dependcy = [int(self.generate_label(x,turn_belief_list,'dependcy')) for x in eachturn_token]
                    print(domainlist)
                    print(slotlist)
                    print(valuelist)

                    print(turn_uttr)
                    print(eachturn_token)
                    print(turn_belief_list)
                    print(label_domain)
                    print(label_dependcy)
                    exit()
                    eachturn_data = {
                        'DialogTurn_id': str(dial_dict["dialogue_idx"].split('.')[0])+'_'+str(turn_id),
                        'eachturn': turn_uttr,
                        'history': dialog_history,
                        'turn_belief': turn_belief_List,
                        'label_domain': label_domain,
                        'label_dependcy':label_dependcy
                    }

                    # dialogue_idx_list.append(dial_dict["dialogue_idx"])
                    # turn_id_list.append(turn_id)
                    # dialogue_domains_list.append(dial_dict["domains"])
                    # turn_domain_list.append(turn["domain"])
                    # dialog_history_list.append(dialog_history)
                    # turn_belief_List.append('#'.join(turn_belief_list))
                    # domain_List.append('#'.join([x.split('-')[0] for x in turn_belief_list]))
                    # slot_List.append('#'.join([x.split('-')[1] for x in turn_belief_list]))
                    # value_List.append('#'.join([x.split('-')[2] for x in turn_belief_list]))
                    # turn_uttr_list.append(turn_uttr)
                    # assert len(dialogue_idx_list) == len(turn_id_list) == len(dialogue_domains_list)
                    # assert len(turn_domain_list) == len(dialog_history_list) == len(turn_belief_List)
                    # assert len(domain_List) == len(slot_List) == len(value_List) == len(turn_uttr_list)


                    # class_label, generate_value, slot_mask, gating_label = [], [], [], []
                    # for slot in self.DOMAIN_SLOTS:
                    #     if slot in turn_belief_dict.keys():
                    #         generate_value.append(turn_belief_dict[slot])
                    #
                    #         if turn_belief_dict[slot] == "dontcare":
                    #             gating_label.append(self.SLOT_GATE["dontcare"])
                    #         elif turn_belief_dict[slot] == "none":
                    #             gating_label.append(self.SLOT_GATE["none"])
                    #         else:
                    #             gating_label.append(self.SLOT_GATE["ptr"])
                    #
                    #     else:
                    #         generate_value.append("none")
                    #         gating_label.append(self.SLOT_GATE["none"])

                    '''
                    generate_value , gating_label
                    ['cheap', 'hotel', 'none', 'none', 'none','none']
                    [0, 0, 2, 2, 2, 2]
                     '''


                    # data_detail = {
                    #     "ID": dial_dict["dialogue_idx"],
                    #     "domains": dial_dict["domains"],
                    #     "turn_domain": turn_domain,
                    #     "turn_id": turn_id,
                    #     "dialog_history": dialog_history,
                    #     "turn_belief": turn_belief_list,
                    #     "gating_label": gating_label,
                    #     "turn_uttr": turn_uttr,
                    #     'generate_value': generate_value
                    # }
                    # if max_hist_len < len(dialog_history.split()):
                    #     max_hist_len = len(dialog_history.split())
                    #
                    # data.append(data_detail)
        data_dict = {
            'dialog_id':dialogue_idx_list,
            'turn_id':turn_id_list,
            'dialogue_domain':dialogue_domains_list,
            'turn_domain':turn_domain_list,
            'uterance':turn_uttr_list,
            'dialogue_history':dialog_history_list,
            'turn_belief':turn_belief_List,
            'domain':domain_List,
            'slot':slot_List,
            'value':value_List,
        }
        data_example = pd.DataFrame(data_dict)
        return data_example


        # {'ID': 'SNG01856.json',
        # 'domains': ['hotel'],
        # 'turn_domain': 'hotel',
        # 'turn_id': 0, 'dialog_history': '; am looking for a plae it should be in a type of hotel ;',
        # 'turn_belief': ['hotel-pricerange-cheap', 'hotel-type-hotel'],
        # 'gating_label': [0, 0, 2, 2, 2],
        # 'turn_uttr': '; am looking for a placa type of hotel',
        # 'generate_value': ['cheap', 'hotel', 'none', 'none','none']}
        # data_info
        # {'ID': ['SNG01856.json', 'SNG01856.json', 'SNG01856.json'],
        #  'domains': [['hotel'], ['hotel'], ['hotel']],
        #  'turn_domain': ['hotel', 'hotel', 'hotel'],
        #  'turn_id': [0, 1, 2],
        #  'dialog_history': [
        #     '; am looking for a place to to stay that has cheap price',
        #     '; am looking for a place to to stay that has cheap price',
        #     '; am looking for a place to to stay that has cheap price '],
        #  'turn_belief': [['hotel-pricerange-cheap', 'hotel-type-hotel'],
        #                  ['hotel-parking-yes', 'hotel-pricerange-cheap', 'hotel-type-hotel'],
        #                  ['hotel-book day-tuesday', 'hotel-book people-6', 'hotel-book stay-3', 'hotel-parking-yes',
        #                   'hotel-pricerange-cheap', 'hotel-type-hotel']], 'gating_label': [
        #     [0, 0, 2, 2, 2,2],
        #     [0, 0, 0, 2, 2],
        #     [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,2]],
        #  'turn_uttr': ['; am looking for a place to to stay that has cheap price range it should be in a type of hotel',
        #                'okay , do you have a specific area you want to stay in ? ; no , i just need to make sure it s cheap . oh , and i need parking',
        #                'i found 1 cheap hotel for you that include -s parking . do you like me to book it ? ; yes , please . 6 people 3 nights starting on tuesday .'],
        #  'generate_value': [
        #      ['cheap', 'hotel', 'none', 'none', 'none', 'none',
        #      ['cheap', 'hotel', 'yes', 'none', 'none', 'none',
        #      ['cheap', 'hotel', 'yes', '3', 'tuesday', '6']]}


        # {'ID': 'SNG01856.json',
        # 'domains': ['hotel'],
        # 'turn_domain': 'hotel',
        # 'turn_id': 0, 'dialog_history': '; am looking for a plae it should be in a type of hotel ;',
        # 'turn_belief': ['hotel-pricerange-cheap', 'hotel-type-hotel'],
        # 'gating_label': [0, 0, 2, 2, 2],
        # 'turn_uttr': '; am looking for a placa type of hotel',
        # 'generate_value': ['cheap', 'hotel', 'none', 'none','none']}


        # '''
        # data_info
        # {'ID': ['SNG01856.json', 'SNG01856.json', 'SNG01856.json'],
        #  'domains': [['hotel'], ['hotel'], ['hotel']],
        #  'turn_domain': ['hotel', 'hotel', 'hotel'],
        #  'turn_id': [0, 1, 2],
        #  'dialog_history': [
        #     '; am looking for a place to to stay that has cheap price',
        #     '; am looking for a place to to stay that has cheap price',
        #     '; am looking for a place to to stay that has cheap price '],
        #  'turn_belief': [['hotel-pricerange-cheap', 'hotel-type-hotel'],
        #                  ['hotel-parking-yes', 'hotel-pricerange-cheap', 'hotel-type-hotel'],
        #                  ['hotel-book day-tuesday', 'hotel-book people-6', 'hotel-book stay-3', 'hotel-parking-yes',
        #                   'hotel-pricerange-cheap', 'hotel-type-hotel']], 'gating_label': [
        #     [0, 0, 2, 2, 2,2],
        #     [0, 0, 0, 2, 2],
        #     [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,2]],
        #  'turn_uttr': ['; am looking for a place to to stay that has cheap price range it should be in a type of hotel',
        #                'okay , do you have a specific area you want to stay in ? ; no , i just need to make sure it s cheap . oh , and i need parking',
        #                'i found 1 cheap hotel for you that include -s parking . do you like me to book it ? ; yes , please . 6 people 3 nights starting on tuesday .'],
        #  'generate_value': [
        #      ['cheap', 'hotel', 'none', 'none', 'none', 'none',
        #      ['cheap', 'hotel', 'yes', 'none', 'none', 'none',
        #      ['cheap', 'hotel', 'yes', '3', 'tuesday', '6']]}
        # '''


if __name__ == "__main__":
    trainer = MultiWOZ(
        data_dir='/home/lsy2018/DST/data',
        output_dir = '/home/lsy2018/TextClassification/DATA/DATA_MultiWOZ/data_1128/',
        onlogy_path= '/multi-woz/MULTIWOZ2 2/ontology.json'
    )
    trainer.load_data()









