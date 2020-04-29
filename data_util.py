import urllib.request
import pandas as pd
import numpy as np
import os

def load_data():
    
    # download data
    if(not os.path.exists("data/ratings_train.txt")):
        print("downloading nsmc dataset")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="data/ratings_train.txt")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="data/ratings_test.txt") 

    # read data
    print("reading data")
    train_data = pd.read_table("data/ratings_train.txt")
    test_data = pd.read_table("data/ratings_test.txt")

    # delete duplicates and nulls
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train_data['document'].replace('', np.nan, inplace=True) # replace space to null
    train_data = train_data.dropna(how='any')
    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    test_data['document'].replace('', np.nan, inplace=True) # replace space to null
    test_data = test_data.dropna(how='any')

    return train_data, test_data
