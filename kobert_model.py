#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import datetime
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.text import Tokenizer

from konlpy.tag import Komoran
from textrank import KeywordSummarizer
komoran = Komoran()
from model_util import kor_summa

from config import logger as log
import config

import pdb

# Korean stopwords
stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에',
    '와', '한', '하다'
]


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(
                                  token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer,
                                                   max_seq_length=max_len,
                                                   pad=pad,
                                                   pair=pair)

        self.sentences = []
        self.labels = []
        for i, data in enumerate(dataset.values):
            if (i + 1) % 1000 == 0:
                print("tokenizing " + str(i + 1) + " out of " +
                      str(len(dataset)))
            encoded_sent = transform(dataset.values[i])
            self.sentences.append(encoded_sent)
            self.labels.append(np.int32(data[label_idx]))

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


def split_train_val(data, rate=0.1):

    train_data, val_data = np.split(data.sample(frac=1),
                                    [int((1 - rate) * len(data))])

    return train_data, val_data


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (
        max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [
        w for w in words
        if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)
    ]
    return words


def kobert(MODEL_NAME, train_data, test_data, MAX_LEN=32):
    if MODEL_NAME == 'summakobert':
        print("summarizing data that are longer than MAX_LEN " + str(MAX_LEN))
        log.info("summarizing data that are longer than MAX_LEN " +
                 str(MAX_LEN))
        if config.data_name == 'nsmc':
            summarizer = KeywordSummarizer(tokenize=komoran_tokenizer,
                                           min_count=1,
                                           min_cooccurrence=1)
            print("summarizing train data")
            train_data = kor_summa(summarizer, train_data, MAX_LEN)
            print("summarizing test data")
            test_data = kor_summa(summarizer, test_data, MAX_LEN)
    
        elif config.data_name == 'imdb':
            print("not implemented yet...")


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_labels = 2
    num_epochs = 5
    batch_size = 32
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5  # 1e-5 or 2e-5

    X_train = np.array(train_data['document'])
    X_test = np.array(test_data['document'])
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [
        index for index, sentence in enumerate(X_train) if len(sentence) < 1
    ]
    drop_test = [
        index for index, sentence in enumerate(X_test) if len(sentence) < 1
    ]

    #  delete empty samples
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    X_test = np.delete(X_test, drop_test, axis=0)
    y_test = np.delete(y_test, drop_test, axis=0)

    train_data = pd.DataFrame({'document': X_train, 'label': y_train})
    test_data = pd.DataFrame({'document': X_test, 'label': y_test})

    print("loading kobert model")
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer_path = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)

    print("training dataset is splited to train*0.9 + val*0.1")
    train_data, val_data = split_train_val(train_data, rate=0.1)

    #  tokenize...
    log.info("tokenizing...")
    print("tokenizing train data")
    data_train = BERTDataset(train_data, 0, 1, tokenizer, MAX_LEN, True, False)
    print("tokenizing val data")
    data_val = BERTDataset(val_data, 0, 1, tokenizer, MAX_LEN, True, False)
    print("tokenizing test data")
    data_test = BERTDataset(test_data, 0, 1, tokenizer, MAX_LEN, True, False)

    train_dataloader = DataLoader(data_train,
                                  batch_size=batch_size,
                                  num_workers=5)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, num_workers=5)
    test_dataloader = DataLoader(data_test,
                                 batch_size=batch_size,
                                 num_workers=5)

    model = BERTClassifier(bertmodel,
                           dr_rate=0.5).to(device)  # TODO : disable dropout
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.01
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     warmup_steps=warmup_step,
                                     t_total=t_total)

    for e in range(num_epochs):
        train_acc = 0.0
        val_acc = 0.0

        print("epoch {}".format(e))
        log.info("epoch {}".format(e))

        # train
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids,
                       label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if (batch_id + 1) % log_interval == 0:
                print("batch {:>5,} loss {:>5,} train acc {:>5,}".format(
                    batch_id + 1,
                    loss.data.cpu().numpy(), train_acc / (batch_id + 1)))
                log.info("batch {:>5,} loss {:>5,} train acc {:>5,}".format(
                    batch_id + 1,
                    loss.data.cpu().numpy(), train_acc / (batch_id + 1)))
        print("epoch {:>5,} avg train acc {:>5,}".format(
            e + 1, train_acc / (batch_id + 1)))
        log.info("epoch {:>5,} avg train acc {:>5,}".format(
            e + 1, train_acc / (batch_id + 1)))

        # val
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids,
                       label) in enumerate(val_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            val_acc += calc_accuracy(out, label)
        print("epoch {:>5,} val acc {:>5,}".format(e + 1,
                                                   val_acc / (batch_id + 1)))
        log.info("epoch {:>5,} val acc {:>5,}".format(e + 1, val_acc /
                                                      (batch_id + 1)))

    #  test
    test_acc = 0.0
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids,
                   label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("test acc {:>5,}".format(test_acc / (batch_id + 1)))
    log.info("test acc {:>5,}".format(test_acc / (batch_id + 1)))
