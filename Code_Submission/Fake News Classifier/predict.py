from __future__ import print_function

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from plot import plot_confusion_matrix
from model import LstmClassifier
import numpy as np

np.random.seed(888)
data_dir_path = './data'
model_dir_path = './models'
config_file_path = model_dir_path + '/' + LstmClassifier.model_name + '-config.npy'
weight_file_path = model_dir_path + '/' + LstmClassifier.model_name + '-weights.h5'

print('loading csv file ...')

df = pd.read_csv("/data/home/cs224n/fake_news_data/merged.csv")

# Set `y`
Y = [1 if label == 'REAL' else 0 for label in df.label]
# Drop the `label` column
df.drop("label", axis=1)

X = df['summarized_text']

config = np.load(config_file_path).item()

classifier = LstmClassifier(config)
classifier.load_weights(weight_file_path)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

print('testing size: ', len(Xtest))

print('start predicting ...')
pred = classifier.predict(Xtest)
print(pred)
score = metrics.accuracy_score(Ytest, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(Ytest, pred, labels=[0, 1])
plot_confusion_matrix(cm, classes=[0, 1])