from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.layers.merge import add
from tqdm import tqdm
from keras import backend as K
from load_data import dataset

def mish(x):
    return x*K.tanh(K.softplus(x))

def evalute_mc(model, X_test, Y_test, sample_times=50):
    batch_size = 32
    err = 0.
    for batch_id in tqdm(range(X_test.shape[0] // batch_size)):
        # take batch of data
        x = X_test[batch_id * batch_size: (batch_id + 1) * batch_size]
        # init empty predictions
        y_ = np.zeros((sample_times, batch_size, Y_test[0].shape[0]))
        for sample_id in range(sample_times):
            # save predictions from a sample pass
            y_[sample_id] = model.predict(x, batch_size)
        # average over all passes
        mean_y = y_.mean(axis=0)
        # evaluate against labels
        y = Y_test[batch_size * batch_id: (batch_id + 1) * batch_size]
        # compute error
        err += np.count_nonzero(np.not_equal(mean_y.argmax(axis=1), y.argmax(axis=1)))
    err = err / X_test.shape[0]
    return 1. - err

def Gru_model(input_size=(178, 1), num_classes=2, af=mish):
    inputs = Input(shape=input_size)
    g1 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(inputs)
    g2 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(g1)
    g3 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(g2)
    a1 = add([g1, g3])
    g4 = GRU(64, activation=af, dropout=0.2)(a1)
    a2 = add([g2, g4])
    f1 = Flatten()(a2)
    d1 = Dense(128, activation=af)(f1)
    d2 = Dropout(0.2)(d1)
    outputs = Dense(num_classes, activation='softmax')(d2)
    model = Model(input=inputs, output=outputs)
    return model

def sgru_model(input_size=(178, 1), num_classes=2, af=mish):
    inputs = Input(shape=input_size)
    g1 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(inputs)
    g2 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(g1)
    g3 = GRU(64, activation=af, dropout=0.2, return_sequences=True)(g2)
    g4 = GRU(64, activation=af, dropout=0.2)(g3)
    #f1 = Flatten()(g4)
    d1 = Dense(128, activation=af)(g4)
    d2 = Dropout(0.2)(d1)
    outputs = Dense(num_classes, activation='softmax')(d2)
    model = Model(input=inputs, output=outputs)
    return model

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from keras.utils import to_categorical
    #from keras.utils import plot_model

    # data = pd.read_csv('Epileptic Seizure Recognition.csv')
    # M = data.values
    # #sns.countplot(data['y'])
    # # plt.show()
    #
    # X_data = M[:, 1:-1]
    # y_data = M[:, -1].astype(int)
    #
    # for j in range(len(y_data)):
    #     if y_data[j] > 1:
    #         y_data[j] = 0
    #
    # print(X_data.shape)
    # C0 = np.argwhere(y_data == 0).flatten()
    # C1 = np.argwhere(y_data == 1).flatten()
    # # C3 = np.argwhere(y_data == 3).flatten()
    # # C4 = np.argwhere(y_data == 4).flatten()
    # # C5 = np.argwhere(y_data == 5).flatten()
    # print('number of each class:', len(C0), len(C1))
    # #y_data = y_data.reshape(-1, 1)
    #
    # scaler = MinMaxScaler(feature_range=(0, 1))  # feature_range=(-1, 1)
    # x_data = scaler.fit_transform(X_data)
    # # x_data = ((X_data-X_data.min())/(X_data.max()-X_data.min()))
    # print('all x:', x_data.shape)
    # print('all y:', y_data.shape)
    #
    # y_data = to_categorical(y_data, 2)
    # x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
    #
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=7)
    # rate = x_test.shape[0]/x_train.shape[0]
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=rate, random_state=7)
    x_train, x_test, y_train, y_test, x_val, y_val = dataset('Epileptic Seizure Recognition.csv')
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(x_val.shape)
    print(y_val.shape)

    opt = Adam(lr=0.001, decay=1e-4)
    model_0 = Gru_model(input_size=(178, 1), num_classes=2, af=mish)
    model_0.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    model_1 = sgru_model(input_size=(178, 1), num_classes=2, af=mish)
    model_1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    #plot_model(model_0, to_file="gru-model.png", show_shapes=True)

    history_0 = model_0.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=2)
    history_1 = model_1.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=2)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    plt.plot(history_0.history['loss'], label='Shunt-GRU')
    plt.plot(history_1.history['loss'], label='GRU')
    plt.xlabel('Epochs')
    plt.ylabel('train loss')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join('Train loss.png'))

    fig_1 = plt.figure()
    ax2 = fig_1.add_subplot(111)
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    plt.plot(history_0.history['val_acc'], label='Shunt-GRU')
    plt.plot(history_1.history['val_acc'], label='GRU')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join('Test acc.png'))

    acc = evalute_mc(model_0, x_test, y_test)
    acc1 = evalute_mc(model_1, x_test, y_test)
    print('mc-evaluate:', acc, acc1)
    loss0, ac0 = model_0.evaluate(x_test, y_test, batch_size=32)
    loss, ac = model_1.evaluate(x_test, y_test, batch_size=32)
    print(ac0, ac)
