from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from plot import create_plot
from model import LstmClassifier
import numpy as np
from collections import Counter

MAX_INPUT_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 2000

def fit_input_text(X, max_input_seq_length=None, max_vocab_size=None):
    if max_input_seq_length is None:
        max_input_seq_length = MAX_INPUT_SEQ_LENGTH
    if max_vocab_size is None:
        max_vocab_size = MAX_VOCAB_SIZE

    input_counter = Counter()
    max_seq_length = 0
    for line in X:
        text = [word.lower() for word in line.split(' ')]
        seq_length = len(text)
        if seq_length > max_input_seq_length:
            text = text[0:max_input_seq_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_seq_length = max(max_seq_length, seq_length)

    word2idx = dict()

    for idx, word in enumerate(input_counter.most_common(max_vocab_size)):
        word2idx[word[0]] = idx + 2
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    idx2word = dict([(idx, word) for word, idx in word2idx.items()])
    num_input_tokens = len(word2idx)
    config = dict()
    config['word2idx'] = word2idx
    config['idx2word'] = idx2word
    config['num_input_tokens'] = num_input_tokens
    config['max_input_seq_length'] = max_seq_length

    return config


np.random.seed(888)

print('loading csv file ...')
df = pd.read_csv("/data/home/cs224n/fake_news_data/merged.csv")

# Set `y`
Y = [1 if label == 'REAL' else 0 for label in df.label]

# Drop the `label` column
df.drop("label", axis=1)

print('extract configuration from input texts ...')
X = df['summarized_text']

config = fit_input_text(X)
config['num_target_tokens'] = 2

print('configuration extracted from input texts ...')

classifier = LstmClassifier(config)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

print('training size: ', len(Xtrain))
print('testing size: ', len(Xtest))

print('start fitting ...')
history = classifier.fit(Xtrain, Ytrain, Xtest, Ytest)

create_plot(history, "LSTM RNN")
