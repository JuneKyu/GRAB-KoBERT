#!/usr/bin/env python
# -*- coding: utf-8 -*-

def kor_summa(summarizer, data, MAX_LEN):
    for i, data_ in data.iterrows():
        sent = data_['document']
        if (i + 1) % 1000 == 0:
            print("summarizing " + str(i + 1))
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
            data.loc[i, 'document'] = sent + '' + summa_sent

    return data

