"""
KCC summarization model
"""
import pandas as pd
import numpy as np
import argparse
import os
import logging

#  from konlpy.tag import Okt
#  from tensorflow.keras.preprocessing.text import Tokenizer
#  from tensorflow.keras.preprocessing.sequence import pad_sequences
#  from tensorflow.keras.layers import Embedding, Dense, LSTM
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.models import load_model
#  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import config
import data_util
from bert_model import bert
from lstm_model import lstm
from gru_model import gru
from kobert_model import kobert

import pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lstm')
    parser.add_argument('--max_len', type=int, default='32')
    parser.add_argument('--data_name', type=str, default='nsmc')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    DATA_NAME = args.data_name

    # logging
    log = config.logger
    folder_path = config.folder_path
    if os.path.exists(folder_path) == False:
        os.makedirs(folder_path)
    fileHandler = logging.FileHandler(
        os.path.join(
            folder_path, config.current_time + '-' + MODEL_NAME + '-' +
            str(MAX_LEN) + '.txt'))
    fileHandler.setFormatter(config.formatter)
    config.logger.addHandler(fileHandler)

    print("kcc research")
    print("")
    log.info("kcc research")
    log.info("")

    log.info("model_name : " + str(MODEL_NAME))
    log.info("max_len : " + str(MAX_LEN))
    log.info("data_name : " + str(DATA_NAME))
    print("model_name : " + str(MODEL_NAME))
    print("max_len : " + str(MAX_LEN))
    print("data_name : " + str(DATA_NAME))
    config.data_name = DATA_NAME

    # load data
    print("loading data ...")
    log.info("loading data ...")
    train_data, test_data = data_util.load_data(DATA_NAME)
    print("")
    log.info("")

    if (MODEL_NAME == 'lstm' or MODEL_NAME == 'summalstm'):  # lstm is baseline

        lstm(MODEL_NAME, train_data, test_data, MAX_LEN)

    elif (MODEL_NAME == 'gru' or MODEL_NAME == 'summagru'):  # gru is baseline

        gru(MODEL_NAME, train_data, test_data, MAX_LEN)

    elif (MODEL_NAME == 'bert' or MODEL_NAME == 'summabert'):  # bert

        bert(MODEL_NAME, train_data, test_data, MAX_LEN)

    elif (MODEL_NAME == 'kobert' or MODEL_NAME == 'summakobert'):  # kobert

        kobert(MODEL_NAME, train_data, test_data, MAX_LEN)


if __name__ == '__main__':
    main()
