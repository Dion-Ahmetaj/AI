
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense,LSTM,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from logisticregression import logisticregression
from sklearn.metrics import accuracy_score

max_features=50000
maxlen=1000
batch_size=32

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

x_train_imdb=np.array(x_train_imdb,dtype=object)
x_test_imdb=np.array(x_test_imdb,dtype=object)

#Concatenate training and testing sets for splitting
x = np.concatenate((x_train_imdb, x_test_imdb), axis=0)
y = np.concatenate((y_train_imdb, y_test_imdb), axis=0)

# Split the data into 80% training and 20% testing
x_train_imdb, x_test_imdb, y_train_imdb, y_test_imdb = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train_imdb.shape)
x_train=pad_sequences(x_train_imdb,maxlen=maxlen)
x_test=pad_sequences(x_test_imdb,maxlen=maxlen)

print(x_train_imdb.shape)
model=Sequential()
model.add(Embedding(max_features,64))

model.add(LSTM(units=64,activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.025),loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train_imdb,epochs=9,batch_size=batch_size)
accuracy,loss=model.evaluate(x_test,y_test_imdb)
print()
print("Simple_RNN Score---> ", model.evaluate(x_test, y_test_imdb))


