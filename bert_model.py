#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import datetime
import random

import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.optimization import WarmupLinearSchedule
from tensorflow.keras.preprocessing.text import Tokenizer

from konlpy.tag import Komoran
from textrank import KeywordSummarizer
komoran = Komoran

from config import logger as log

import pdb

# Korean stopwords
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def split_train_val(data, rate = 0.1):

    train_data, val_data = np.split(data.sample(frac=1), [int((1-rate)*len(data))])
    
    return train_data, val_data


def tokenize(tokenizer, sentences, MAX_LEN):

    input_ids = []
    len_sentences = len(sentences)
    for i, sent in enumerate(sentences):
        if (i + 1) % 1000 == 0:
            print("tokenizing " + str(i + 1) + " out of " + str(len_sentences))
        encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length = MAX_LEN, # 32 / 64 / 128 / none
                pad_to_max_length = True
                )
        encoded_sent = [word for word in encoded_sent if not word in stopwords] # delete stopwords
        input_ids.append(encoded_sent)

    #  MAX_LEN = max([len(sen) for sen in input_ids])

    print("padding...")
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    return input_ids, MAX_LEN, attention_masks


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words


def fine_tune_and_test(train_data, val_data, test_data, num_labels, num_epochs, batch_size):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log.info('BERT embedding model fine tuning : bert-base-uncase')

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    embedding_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False,
            )
    embedding_model.cuda()
    log.info('bert model num labels : ' + str(num_labels))
    
    optimizer = AdamW(embedding_model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    log.info('optimizer : AdamW with lr : 2e-5, eps : 1e-8')

    epochs = num_epochs

    total_steps = len(train_dataloader) * num_epochs

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch
    loss_values = []

    # ================
    # fine tuning
    # ================

    log.info("")
    log.info("fine tuning word dataset with BERT")

    print("")
    print("fine tuning word dataset with BERT")

    # For each epoch...
    for epoch_i in range(0, epochs):
        # ================
        # training
        # ================
        
        print("")
        print("epoch {}".format(epoch_i))
        log.info("")
        log.info("epoch {}".format(epoch_i))

        print("")
        print("training bert model...")
        log.info("")
        log.info("training bert model...")

        t0 = time.time()
        total_loss = 0 # reset total loss

        embedding_model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print("batch {:>5,} of {:>5}. Elapsed: {:}.".format(step, len(train_dataloader), elapsed))
                log.info("batch {:>5,} of {:>5}. Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_attn_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            embedding_model.zero_grad()

            outputs = embedding_model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_attn_mask,
                                    labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 1.0) # prevent the "exploding gradient"

            optimizer.step()

            scheduler.step() # update the learning rate (in transformer architecture)

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        #  log.info("")
        log.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        log.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
        # ================
        # validation
        # ================

        print("")
        print("running bert embedding validation...")
        #  log.info("")
        log.info("running bert embedding validation...")


        t0 = time.time()

        embedding_model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_step, nb_eval_examples = 0, 0

        for batch in val_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_attn_mask, b_labels = batch

            with torch.no_grad():
                outputs = embedding_model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_attn_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_step += 1
        
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_step))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        log.info("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_step))
        log.info("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("fine tuning complete!")
    log.info("")
    log.info("fine tuning complete!")

    print("")
    print("test")
    log.info("")
    log.info("test")

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    #  print("Predicting labels for {:,} test sentences...".format(len()))
    model.eval()

    predictions, true_labels = [], []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
    
    accuracy_set = []

    for i in range(len(true_labels)):

        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        acc = flat_accuracy(predictions, true_labels)
        accuracy_set.append(acc)

    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_lables = np.concatenate(true_labels, axis=0)
    test_accuracy = flat_accuracy(flat_predictions, flat_true_lables)
    print("  Accuracy: {0:.2f}".format(test_accuracy))
    log.info("  Accuracy: {0:.2f}".format(test_accuracy))


def bert(MODEL_NAME, train_data, test_data, MAX_LEN = 32):

    if MODEL_NAME == 'summabert':
        print("summarizing data that are longer than MAX_LEN " + str(MAX_LEN))
        log.info("summarizing data that are longer than MAX_LEN " + str(MAX_LEN))

        summarizer = KeywordSummarizer(tokenize=komoran_tokenizer, min_count=1, min_cooccurrence=1)
        for i, data in train_data.iterrows():
            sent = data['document']
            if (i + 1) % 1000 == 0:
                print("train summarizing " + str(i + 1))
            if len(sent) > MAX_LEN:
                words = sent.split(' ')
                if len(words) < 2:
                    continue
                try:
                    summa_words = summarizer.summarize(words, topk=30)
                except:
                    continue
                summa_sent = ''
                for word in summa_words:
                    summa_sent += word[0].split('/')[0] + ' '
                train_data.loc[i, 'document'] = summa_sent

        for i, data in test_data.iterrows():
            sent = data['document']
            if (i + 1) % 1000 == 0:
                print("test summarizing " + str(i + 1))
            if len(sent) > MAX_LEN:
                words = sent.split(' ')
                if len(words) < 2:
                    continue
                try:
                    summa_words = summarizer.summarize(words, topk=30)
                except:
                    continue
                summa_sent = ''
                for word in summa_words:
                    summa_sent += word[0].split('/')[0] + ' '
                test_data.loc[i, 'document'] = summa_sent
        

    num_labels = 2
    num_epochs = 5
    batch_size = 32
    
    #  count_tokenizer = Tokenizer()
    #  count_tokenizer.fit_on_texts(train_data['document'])
    #
    #  threshold = 3
    #  total_cnt = len(count_tokenizer.word_index)
    #  rare_cnt = 0
    #  total_freq = 0
    #  rare_freq = 0
    #
    #  X_train = np.array(train_data['document'])
    #  X_test = np.array(test_data['document'])
    #  y_train = np.array(train_data['label'])
    #  y_test = np.array(test_data['label'])
    #
    #  for key, value in count_tokenizer.word_counts.items():
    #      total_freq = total_freq + value
    #
    #      if(value < threshold):
    #          rare_cnt = rare_cnt + 1
    #          rare_freq = rare_freq + value
    
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]

    # delete empty samples
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    X_test = np.delete(X_test, drop_test, axis=0)
    y_test = np.delete(y_test, drop_test, axis=0)

    train_data = pd.DataFrame({'document':X_train, 'label':y_train})
    test_data = pd.DataFrame({'document':X_test, 'label':y_test})

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

    print("training dataset is splited to train*0.9 + val*0.1")
    train_data, val_data = split_train_val(train_data, rate = 0.1)

    log.info("tokenizing...")
    print("tokenizing train data...")
    train_inputs, train_MAX_LEN, train_attn_masks = tokenize(tokenizer, train_data['document'], MAX_LEN)
    train_labels = train_data['label'].values

    print("tokenizing val data...")
    val_inputs, val_MAX_LEN, val_attn_masks = tokenize(tokenizer, val_data['document'], MAX_LEN)
    val_labels = val_data['label'].values

    print("tokenizing test data...")
    test_inputs, test_MAX_LEN, test_attn_masks = tokenize(tokenizer, test_data['document'], MAX_LEN)
    test_labels = test_data['label'].values
    
    # convert to Pytorch data types
    train_inputs = torch.tensor(train_inputs)
    train_attn_masks = torch.tensor(train_attn_masks)
    train_labels = torch.tensor(train_labels)
    val_inputs = torch.tensor(val_inputs)
    val_attn_masks = torch.tensor(val_attn_masks)
    val_labels = torch.tensor(val_labels)
    test_inputs = torch.tensor(test_inputs)
    test_attn_masks = torch.tensor(test_attn_masks)
    test_labels = torch.tensor(test_labels)

    # Create the DataLoader
    train_data = TensorDataset(train_inputs, train_attn_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_attn_masks, val_labels)
    test_data = TensorDataset(test_inputs, test_attn_masks, test_labels)
    
    # fine tune and test
    fine_tune_and_test(train_data = train_data, val_data = val_data, test_data = test_data, num_labels = num_labels, num_epochs = num_epochs, batch_size = batch_size)

