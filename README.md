# GRAB vector, GRAB-(Ko)BERT

This is a implementation for (kor)\<그래프 순위 결정 메커니즘을 이용한 BERT 기반 감정분석 모델의 개선\> (eng)[\<Improving BERT-based Sentiment Analysis Model using Graph-based Ranking Mechanism\>](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE10528632&mark=0&useDate=&bookmarkCnt=1&ipRange=N&accessgl=Y&language=ko_KR)

This paper is a future work from (kor)\<감정 분석을 위한 그래프 순위화 기바 강인한 한국어 BERT 모델\> (eng)[\<Robust Korean BERT Model for Sentiment Analysis using Graph-based Ranking Mechansim\>](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE09874584&mark=0&useDate=&bookmarkCnt=1&ipRange=N&accessgl=Y&language=ko) (best paper awarded from KCC2020)


## Abstract

> > Due to the need for automated document processing, artificial intelligence research has been actively
> > conducted in the field of natural lanugage processing(NLP). In this paper, we propose the GRAB vector
> > (GRAph-Based vector), which consists of vectorized keyword-based morphemes or summaries extracted
> > from the graph-based ranking mechanism. Next, we applied the GRAB vector to the sentiment analysis task,
> > which is an NLP task, and we proposed a more accurate and robust model, GRAB-BERT(GRAB vector-BERT model).
> > Then, to analyze the effect of the GRAB vector on this model, we compared the performances of recurrent
> > neural network models(RNNs) and BERT models with or without the application of the GRAB vecotr on both
> > English and Korean text samples with different sequence sizes. Our results demonstrate that applying the
> > GRAB vector to models such as BERT to process inputs in parallel improes the robustness of the model and
> > its performance. Furthermore, unlike BERT-based models, RNN models are more effective ehen applying
> > graph-based extracted summaries than when applying morpheme-based summaries.

### Requirements

* Python 3.6
* PyTorch 1.7.1
* Tensorflow 1.5.1

To install all the required elements, run the code:
```
bash requirements.txt
```

### Run scripts

The English textrank module (graph-based keywork extractor) is from [textrank](https://github.com/summanlp/textrank). 
The Korean textrank module (graph-based morpheme extractor) is from [textrank](https://github.com/lovit/textrank).
(Korean module requires jdk installation)

Implemented dataset list : [NSMC](https://github.com/e9t/nsmc), [IMDB](https://www.imdb.com/interfaces/).

To run the experiments, run the scripts:
```
sh run.sh
```

experiments:
examples of outputs of graph-based ranking mechanism.



validation accuracy plot of lstm based models

![lstm_val_acc](imgs/lstm_val_acc.jpg)

validation accuracy plot of bert based models

![bert_val_acc](imgs/bert_val_acc.jpg)

validation accuracy plot of kobert based models

![kobert_val_acc](imgs/kobert_val_acc.jpg)

training loss plot of kobert based models

![kobert_training_loss](imgs/kobert_loss.jpg)
