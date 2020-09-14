import numpy as np

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from konlpy.tag import Komoran
from textrank import KeywordSummarizer
komoran = Komoran()

from config import logger as log

import pdb

# Korean stopwords
stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에',
    '와', '한', '하다'
]


def tokenize(sentences, okt):
    X_ = []
    len_sentences = len(sentences)
    for i, sent in enumerate(sentences['document']):
        if (i + 1) % 1000 == 0:
            print("tokenizing " + str(i + 1) + " out of " + str(len_sentences))
        encoded_sent = []
        encoded_sent = okt.morphs(sent, stem=True)  # tokenize
        encoded_sent = [
            word for word in encoded_sent if not word in stopwords
        ]  # delete stopwords
        X_.append(encoded_sent)

    return X_


def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [
        w for w in words
        if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)
    ]
    return words


def gru(MODEL_NAME, train_data, test_data, MAX_LEN=32):

    if MODEL_NAME == 'summagru':
        print("summarizing data that are longer than MAX_LEN " + str(MAX_LEN))
        log.info("summarizing data that are longer than MAX_LEN " +
                 str(MAX_LEN))

        summarizer = KeywordSummarizer(tokenize=komoran_tokenizer,
                                       min_count=1,
                                       min_cooccurrence=1)
        for i, data in train_data.iterrows():
            sent = data['document']
            if (i + 1) % 1000 == 0:
                print("train summarizing " + str(i + 1))
            #  if len(sent) > MAX_LEN:
            if len(sent) < MAX_LEN:
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
                train_data.loc[i, 'document'] = sent + '' + summa_sent

        # #only
        #  for i, data in test_data.iterrows():
        #      sent = data['document']
        #      if (i + 1) % 1000 == 0:
        #          print("test summarizing " + str(i + 1))
        #      if len(sent) > MAX_LEN:
        #          words = sent.split(' ')
        #          if len(words) < 2:
        #              continue
        #          try:
        #              summa_words = summarizer.summarize(words, topk=30)
        #          except:
        #              continue
        #          summa_sent = ''
        #          for word in summa_words:
        #              summa_sent += word[0].split('/')[0] + ' '
        #          test_data.loc[i, 'document'] = sent + summa_sent

    print("tokenizing...")
    log.info("tokenizing...")
    okt = Okt()
    print("tokenizing train_data")
    X_train = tokenize(train_data, okt)
    print("tokenizing test_data")
    X_test = tokenize(test_data, okt)
    print()
    log.info("")

    # encode
    print("encoding and preprocessing...")
    log.info("encoding and preprocessing...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index)
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    # delete rare tokens
    vocab_size = total_cnt - rare_cnt + 1

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [
        index for index, sentence in enumerate(X_train) if len(sentence) < 1
    ]
    drop_test = [
        index for index, sentence in enumerate(X_test) if len(sentence) < 1
    ]

    # delete empty samples
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    X_test = np.delete(X_test, drop_test, axis=0)
    y_test = np.delete(y_test, drop_test, axis=0)

    # 32 / 64 / 128
    max_len = MAX_LEN

    # padding
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    print()

    print("reached checkpoint!")
    log.info("reached checkpoint!")

    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(GRU(128))
    model.add(Dense(1, activation='sigmoid'))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5',
                         monitor='val_acc',
                         mode='max',
                         verbose=1,
                         save_best_only=True)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(X_train,
                        y_train,
                        epochs=15,
                        callbacks=[es, mc],
                        batch_size=60,
                        validation_split=0.1)
    loaded_model = load_model('best_model.h5')

    print("acc : %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
    log.info("acc : %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
