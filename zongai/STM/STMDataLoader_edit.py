# -*- coding: utf-8 -*-

import numpy as np
import torch
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset
from nltk.tokenize import TreebankWordTokenizer
from sklearn.preprocessing import LabelEncoder

def pad_sequences(sequences, maxlen):
    """
    padding='post', truncating='pre'
    """
    new_seq = []
    for sequence in sequences:
        if len(sequence) <= maxlen:
            sequence.extend([0]*(maxlen-len(sequence)))
        else:
            sequence = sequence[-maxlen:]
        new_seq.append(sequence)
    return new_seq

'''
def tokenizer_oov_handler(vocab_table, text):
    nltk_tokenizer = TreebankWordTokenizer()
    text = nltk_tokenizer.tokenize(text)
    all_vocab = list(vocab_table.keys())
    text = [i if i in all_vocab else '<UNK>' for i in text]
    return text
'''

def build_corpus_tokenizer(training_examples):
    """
    Use training set to build tokenizer
    :param training_examples: 
    :return: 
    """
    assert training_examples[0].guid.startswith('train')

    all_seq = []
    for (ex_index, example) in enumerate(training_examples):
        all_seq += example.text_a + example.text_b
    all_seq = ' '.join(all_seq)
    nltk_tokenizer = TreebankWordTokenizer()
    # tokenizer = LabelEncoder()
    # tokenizer.fit(nltk_tokenizer.tokenize(all_seq) + ['<UNK>'])
    all_vocab = list(set(nltk_tokenizer.tokenize(all_seq)))
    all_vocab.insert(0, '<UNK>')
    vocab_table = {}
    for (index, token) in enumerate(all_vocab):
        vocab_table[token] = index


    # tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>', split=' ', lower=True)
    # tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>', split=' ', lower=True)
    # tokenizer.fit_on_texts(all_seq)

    return vocab_table


def build_corpus_dataloader(examples, max_turn_length, max_seq_length, vocab_table):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))
    # nltk_tokenizer = TreebankWordTokenizer()
    contexts = []
    candidates = []
    labels = []
    for (ex_index, example) in enumerate(examples):

        # tokens_a = pad_sequences(tokenizer.texts_to_sequences(example.text_a), maxlen=max_seq_length, padding='post', truncating='pre')
        # tokens_b = pad_sequences(tokenizer.texts_to_sequences(example.text_b), maxlen=max_seq_length, padding='post', truncating='pre')
        # tokens_a = pad_sequences([list(tokenizer.transform(tokenizer_oov_handler(tokenizer, i))) for i in example.text_a], maxlen=max_seq_length)
        # tokens_b = pad_sequences([list(tokenizer.transform(tokenizer_oov_handler(tokenizer, i))) for i in example.text_b], maxlen=max_seq_length)
        tokens_a = []
        for line in example.text_a:
            indices = []
            for word in line:
                try:
                    indices.append(vocab_table[word])
                except:
                    indices.append(vocab_table['<UNK>'])
            # line = tokenizer_oov_handler(vocab_table, line)
            # indices = [vocab_table[i] for i in line]
            tokens_a.append(indices)
        tokens_a = pad_sequences(tokens_a, maxlen=max_seq_length)
        tokens_b = []
        for line in example.text_b:
            indices = []
            for word in line:
                try:
                    indices.append(vocab_table[word])
                except:
                    indices.append(vocab_table['<UNK>'])
            tokens_b.append(indices)
        tokens_b = pad_sequences(tokens_b, maxlen=max_seq_length)
        # print(tokens_a)
        # exit()

        # Zero-pad up to the sequence length.
        if len(tokens_a) < max_turn_length:
            tokens_a = np.concatenate([tokens_a,np.zeros([max_turn_length - len(tokens_a), max_seq_length])])
        else:
            tokens_a = tokens_a[-max_turn_length:]

        max_candidate_len = 100
        if len(tokens_b) < max_candidate_len:
            if tokens_b:
                tokens_b = np.concatenate([tokens_b,np.zeros([max_candidate_len - len(tokens_b), max_seq_length])])
            else:
                tokens_b = np.zeros([max_candidate_len, max_seq_length])
        else:
            tokens_b = tokens_b[-max_candidate_len:]

        assert len(tokens_a) == max_turn_length
        assert len(tokens_b) == max_candidate_len

        contexts.append(tokens_a)
        candidates.append(tokens_b)
        labels.append(example.label)


    all_contexts = torch.LongTensor(contexts)
    all_candidates = torch.LongTensor(candidates)
    all_labels = torch.LongTensor(labels)

    # for i in all_labels:
    #     if i == -1:
    #         print(i)
    # exit()


    tensor_dataset = TensorDataset(all_contexts, all_candidates, all_labels)

    return tensor_dataset

def build_corpus_embedding( emb_dim, pretrain_dir, vocab_table):

    ### Get vocabulary
    vocab = ['<PAD>'] + list(vocab_table.keys())#[:vocab_size-1]
    vocab_dict = {}
    for i, w in enumerate(vocab):
        vocab_dict[w] = i
    ### Initialize word embedding
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(vocab), emb_dim))

    emb_count = 0
    with open(pretrain_dir) as f:
        for line in f.readlines():
            row = line.split(' ')
            assert len(row[1:]) == emb_dim
            try:
                word_embeds[vocab_dict[row[0]]] = [float(v) for v in row[1:]]
                emb_count += 1
            except:
                pass
    word_embeds[0] = [0.0]*emb_dim
    word_embeds = torch.FloatTensor(word_embeds)
    print('Loaded %i pretrained embeddings.' % emb_count)

    return word_embeds, vocab