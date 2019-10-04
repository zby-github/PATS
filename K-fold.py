
from sklearn.model_selection import KFold
import pandas as pd
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from new_feature_vis import load_data
from sklearn import metrics
from perform import Performance

def mish(x):
    return x*K.tanh(K.softplus(x))

def evaluate(actual, predicted):
  acc = metrics.accuracy_score(actual, predicted)
  rec = metrics.recall_score(actual, predicted, average='macro')
  pp = metrics.precision_score(actual, predicted, average='macro')
  return acc, rec, pp

def sgru_model(input_size=(178, 1), num_classes=1, af=mish):
  inputs = Input(shape=input_size)
  g1 = GRU(32, activation=af, dropout=0.2, return_sequences=True)(inputs)
  g2 = GRU(32, activation=af, dropout=0.2, return_sequences=True)(g1)
  g3 = GRU(32, activation=af, dropout=0.2, return_sequences=True)(g2)
  a1 = add([g1, g3])
  g4 = GRU(32, dropout=0.2, activation=af, return_sequences=True)(a1)
  a2 = add([g2, g4])
  f1 = Flatten()(a2)
  d1 = Dense(128, activation=af)(f1)
  d2 = Dropout(0.2)(d1)
  outputs = Dense(num_classes, activation='sigmoid')(d2)
  model = Model(input=inputs, output=outputs)
  return model

data_path = 'Epileptic Seizure Recognition.csv'
Raw_data, new_data = load_data(data_path)

print(Raw_data.shape)
print(new_data.shape)

New_sam = KFold(n_splits=10)

i = 0
ACC = []
Se = []
Pp = []
F_score = []
for train_index, test_index in New_sam.split(Raw_data):
  i += 1
  Sam_train, Sam_test = Raw_data[train_index], Raw_data[test_index]
  X_train = Sam_train[:, 0:-1]
  Y_train = Sam_train[:, -1].astype(int)
  X_test = Sam_test[:, 0:-1]
  Y_test = Sam_test[:, -1].astype(int)
  Y_train[Y_train > 1] = 0
  Y_test[Y_test > 1] = 0

  print(X_train.shape)
  print(Y_train.shape)
  print(X_test.shape)
  print(Y_test.shape)

  #y_train = to_categorical(Y_train, 2)
  #y_test = to_categorical(Y_test, 2)
  y_train = Y_train
  y_test = Y_test

  x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
  x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  print('X_train:', x_train.shape, 'X_test:', x_test.shape)  # 结果表明每次划分的数量
  print('Y_train:', y_train.shape, 'Y_test:', y_test.shape) #结果表明每次划分的数量

  opt = Adam(lr=0.001)
  model_0 = sgru_model(input_size=(178, 1), num_classes=1, af=mish)
  model_0.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
  model_0.save('model_0.h5')
  history_0 = model_0.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=2)
  pd.DataFrame.from_dict(history_0.history).to_csv("sgru_rawdata_log_"+str(i)+".csv", float_format="%.5f", index=False)

  # fig = plt.figure()
  # ax1 = fig.add_subplot(111)
  # ax1.spines['top'].set_color('none')
  # ax1.spines['right'].set_color('none')
  # plt.plot(history_0.history['loss'], label='Train loss')
  # plt.plot(history_0.history['val_loss'], label='test loss')
  # plt.xlabel('Epochs')
  # plt.ylabel('loss')
  # plt.grid(ls='--')
  # plt.legend()
  # plt.savefig(os.path.join('Raw-Train-loss_'+str(i)+'.png'))

  # fig_1 = plt.figure()
  # ax2 = fig_1.add_subplot(111)
  # ax2.spines['top'].set_color('none')
  # ax2.spines['right'].set_color('none')
  # plt.plot(history_0.history['acc'], label='train acc')
  # plt.plot(history_0.history['val_acc'], label='test acc')
  # plt.xlabel('Epochs')
  # plt.ylabel('Acc')
  # plt.grid(ls='--')
  # plt.legend()
  # plt.savefig(os.path.join('Raw-Test-acc_'+str(i)+'.png'))

  y_hat = model_0.predict(x_test, batch_size=32)
  loss, acc0 = model_0.evaluate(x_test, y_test, batch_size=32)

  labels = y_test
  scores = y_hat[:, 0]
  p = Performance(labels, scores)
  acc = p.accuracy()
  pre = p.presision()
  rec = p.recall()
  f1 = p.F1()

  ACC.append(acc)
  Se.append(rec)
  Pp.append(pre)
  F_score.append(f1)

j = 0
ACC1 = []
Se1 = []
Pp1 = []
F_score1 = []
for train_index, test_index in New_sam.split(new_data):
  j += 1
  Sam_train, Sam_test = new_data[train_index], new_data[test_index]
  X_train = Sam_train[:, 0:-1]
  Y_train = Sam_train[:, -1].astype(int)
  X_test = Sam_test[:, 0:-1]
  Y_test = Sam_test[:, -1].astype(int)
  Y_train[Y_train > 1] = 0
  Y_test[Y_test > 1] = 0

  y_train = Y_train
  y_test = Y_test

  x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
  x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

  print('X_train:', x_train.shape, 'X_test:', x_test.shape)  # 结果表明每次划分的数量
  print('Y_train:', y_train.shape, 'Y_test:', y_test.shape) #结果表明每次划分的数量

  opt = Adam(lr=0.001)
  model_1 = sgru_model(input_size=(179, 1), num_classes=1, af=mish)
  model_1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
  history_1 = model_1.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=2)
  pd.DataFrame.from_dict(history_1.history).to_csv("sgru_newdata_log_"+str(j)+".csv", float_format="%.5f", index=False)

  y_hat = model_1.predict(x_test, batch_size=32)

  labels = y_test
  scores = y_hat[:, 0]
  p = Performance(labels, scores)
  acc = p.accuracy()
  pre = p.presision()
  rec = p.recall()
  f1 = p.F1()

  ACC1.append(acc)
  Se1.append(rec)
  Pp1.append(pre)
  F_score1.append(f1)

print('ACC:', ACC)
print('ACC1:', ACC1)
print('avg_acc:', sum(ACC)/len(ACC))
print('avg_acc1:', sum(ACC1)/len(ACC1))
print('**********')

print('Se:', Se)
print('Se1:', Se1)
print('avg_Se:', sum(Se)/len(Se))
print('avg_Se1:', sum(Se1)/len(Se1))
print('**********')

print('Pp:', Pp)
print('Pp1:', Pp1)
print('avg_PP:', sum(Pp)/len(Pp))
print('avg_PP1:', sum(Pp1)/len(Pp1))
print('**********')

print('F_score:', F_score)
print('F_Score1:', F_score1)
print('avg_Fcore', sum(F_score)/len(F_score))
print('avg_Fcore1', sum(F_score1)/len(F_score1))

