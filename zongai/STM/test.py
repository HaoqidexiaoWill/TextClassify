# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
# from oldcode import evaluation as eva
# from oldcode.processor import UbuntuProcessor
# from oldcode.STMDataLoader import build_corpus_dataloader, build_corpus_tokenizer, build_corpus_embedding
import evaluation as eva
from processor_edit2 import UbuntuProcessor
from STMDataLoader_edit import build_corpus_dataloader, build_corpus_tokenizer, build_corpus_embedding
from models import STM
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
# plt.interactive(False)
# plt.figure(figsize=(20,30))
import logging
import gc

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description='Spatio-Temporal Matching Network')
    parser.add_argument('--task', type=str, default='Response Selection',
                        help='task name')
    parser.add_argument('--model', type=str, default='STM',
                        help='model name')
    parser.add_argument('--encoder_type', type=str, default='GRU',
                        help='encoder:[GRU, LSTM, SRU, Transoformer]')
    parser.add_argument('--vocab_size', type=int, default=2147097,
                        help='vocabulary size')
    parser.add_argument('--max_turns_num', type=int, default=9,
                        help='the max turn number in dialogue context')
    parser.add_argument('--max_options_num', type=int, default=100,
                        help='the max turn number in dialogue context')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='the max length of the input sequence')

    parser.add_argument('--emb_dim', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--rnn_layers', type=int, default=3,
                        help='the number of rnn layers for feature extraction')
    parser.add_argument('--mem_dim', type=int, default=200,
                        help='hidden memory size')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='batch size')
    parser.add_argument('--dropoutP', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use CUDA')

    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='The initial checkpoint')
    parser.add_argument('--save_path', type=str, default='./tmp/',
                        help='The initial checkpoint')
    parser.add_argument('--cache_path', type=str, default='./cache/',
                        help='The initial checkpoint')
    parser.add_argument('--pretrain_embedding', type=str, default=os.path.join(os.getcwd(), "dataset/glove_42B_300d_vec_plus_word2vec_100.txt"),
                        help='The pretraining embedding')

    parser.add_argument('--do_train', type=bool, default=True,
                        help='training or not')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='evaluate or not')
    parser.add_argument('--do_test', type=bool, default=False,
                        help='test or not')

    args = parser.parse_args()
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device %s", device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ### Build dataloader
    processor = UbuntuProcessor()
    train_examples = processor.get_train_examples()
    eval_examples = processor.get_dev_examples()
    tokenizer = build_corpus_tokenizer(eval_examples)
    eval_dataset = build_corpus_dataloader(eval_examples, args.max_turns_num, args.max_seq_len, tokenizer)
    test_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)

    ### Build pretrained embedding
    pretrained_embedding, vocab = build_corpus_embedding(args.emb_dim, args.pretrain_embedding, tokenizer)

    ### Build model


    ### testing
    print('start testing')
    model, _ = torch.load(os.path.join(args.save_path, 'model_best.pt'))
    model.eval()
       
    test_loss = 0
    test_score_file = open(os.path.join(args.save_path, 'test_score.txt'), 'w')

    for nb_test_steps, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        contexts_ids, candidate_ids, label_ids = batch
        with torch.no_grad():
            tmp_test_loss, logits = model(contexts_ids, candidate_ids, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        for i in range(args.batch_size):
            for j in range(args.max_options_num):
                if label_ids[i] == j:
                    test_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 1))
                else:
                    test_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 0))

        test_loss += tmp_test_loss.mean().item()

        test_score_file.close()

        test_loss /= (nb_eval_steps + 1)

        # write evaluation result
        test_result = eva.evaluate(os.path.join(args.save_path, 'test_score.txt'))
        test_result_file_path = os.path.join(args.save_path, 'test_result.txt')
        with open(test_result_file_path, 'w') as out_file:
            for p_at in test_result:
                out_file.write(str(p_at) + '\n')



if __name__ == '__main__':
    main()