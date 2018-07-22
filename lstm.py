import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc

np.random.seed(0)

import database as db

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    #model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    #model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(10, input_shape=(input_length,),activation='relu'))
    model.add(Dense(50, activation='softmax')) 
    model.add(Dense(1,activation='sigmoid'))
    #model.add(Dense(1))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def get_data():
    _xparams, X, y = db.load_data_with_goes_for_lstm_bin_clf(th=4.5, mI=1, isgoes = True, y=2017)
    print _xparams
    y = np.asarray(y[:,0].tolist())
    return pad_sequences(X),y


X,y = get_data()
print(X)
model = create_model(len(X[0]))

print ('Fitting model...')
hist = model.fit(X, y, batch_size=64, nb_epoch=10, validation_split = 0.1, verbose = 1)

score, acc = model.evaluate(X, y, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred_keras = model.predict(X).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()
plt.savefig("out/stat/keras.roc.png")
