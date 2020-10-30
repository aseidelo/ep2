import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import argparse

def generate_dataset(file_path, max_size=None):
    reviews = []
    ratings = []
    with open(file_path + '.csv', 'r') as in_file:
        head = in_file.readline().split(';')
        i = 0
        for line in in_file:
            try:
                fields = line.split(';')
                if(int(fields[8]) in [1, 2, 3, 4, 5]):
                    ratings.append(int(fields[8]) - 1)
                    reviews.append(fields[10].rstrip())
                    i = i + 1
                if (max_size is not None):
                    if(i >= max_size):
                        break
            except:
                pass
    dataset = pd.DataFrame(data={"review_text" : reviews, "rating" : ratings})
    return dataset

def split_and_save(dataset, val_prop, test_prop, out_path, regularize_train):
    x = dataset["review_text"]
    y = dataset["rating"]
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_prop, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_prop, random_state=42, stratify=None)
    pd.DataFrame(data={"review_text" : x_train, "rating" : y_train}).to_csv(out_path + "_" + str(len(x)) + '_train.csv')
    pd.DataFrame(data={"review_text" : x_val, "rating" : y_val}).to_csv(out_path + "_" + str(len(x)) + '_val.csv')
    pd.DataFrame(data={"review_text" : x_test, "rating" : y_test}).to_csv(out_path + "_" + str(len(x)) + '_test.csv')

def load_dataset(file_path):
    return pd.read_csv(file_path)

def load_train_val_test(file_path):
    train = pd.read_csv(file_path + '_train.csv')
    val = pd.read_csv(file_path + '_val.csv')
    test = pd.read_csv(file_path + '_test.csv')    
    return train, val, test

def create_tokenizer(texts, vocab_size, maxlen):
    tokenizer = layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size, output_sequence_length=maxlen)
    tokenizer.adapt(texts)
    return tokenizer

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Prepare B2W dataset for Sentiment Analisys task in portuguese.')
    parser.add_argument('--inpath', metavar='<dir>', type=str, help='raw data path')
    parser.add_argument('--outpath', metavar='<dir>', type=str, help='path for processed data')
    parser.add_argument('--dataset', metavar='<csv file prefix>', type=str, help='csv dataset file name')
    parser.add_argument('-N', metavar='N', type=int, help='Number of samples for output dataset', default=None)
    parser.add_argument('-strattrain', action='store_true', help='Stratify train and validation labels?')
    args = parser.parse_args()
    in_path = args.inpath
    out_path = args.outpath
    dataset_name = args.dataset
    N = args.N
    dataset = generate_dataset(in_path + '/' + dataset_name, max_size=N)
    print(args.strattrain)
    split_and_save(dataset, 0.2, 0.2, out_path + '/' + dataset_name, args.strattrain)
    print(dataset["rating"].value_counts())


