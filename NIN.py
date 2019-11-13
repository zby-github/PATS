# -*- coding: utf-8 -*-
"""
Test on CIFAR-10 using NIN network model
"""
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator 
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers import Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import keras
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import tensorflow as tf
import pandas as pd
from dataset_10 import load_data

def error(y_true, y_pred):
    Error = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
    return Error

def mish(x):
    return x*K.tanh(K.softplus(x))

def swish(x):
    return x*K.sigmoid(x)

def atan(x):
    return tf.atan(x)

p = np.random.uniform(1/2, 3/4, 1)
def pats(x, k=p[0]):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

x_train, y_train, x_test, y_test = load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

batch_size = 128
epochs = 200
iterations = 391
weight_decay = 0.0001

def nin_model(a, dropout):
    model = Sequential()
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(192, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(96, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (5, 5), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(192, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(192, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(192, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(192, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))
    model.add(Conv2D(10, (1, 1), kernel_regularizer=regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(a))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[error])
    return model

def sch_sgd(epoch):
    if epoch < 80:
        return 0.001
    if epoch < 150:
        return 0.0001
    return 0.00001

cbks_sgd = [LearningRateScheduler(sch_sgd)]

model_1 = nin_model('relu', 0.5)
model_2 = nin_model(mish, 0.5)
model_3 = nin_model(pats, 0.5)

datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, fill_mode='constant',cval=0.)
datagen.fit(x_train)

history_1 = model_1.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks_sgd, validation_data=(x_test, y_test))
#pd.DataFrame.from_dict(history_1.history).to_csv("NIN_relu_log.csv", float_format="%.5f", index=False)

history_2 = model_2.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks_sgd, validation_data=(x_test, y_test))
#pd.DataFrame.from_dict(history_2.history).to_csv("NIN_mish_log.csv", float_format="%.5f", index=False)

history_3 = model_3.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks_sgd, validation_data=(x_test, y_test))
#pd.DataFrame.from_dict(history_3.history).to_csv("NIN_myfun_log.csv", float_format="%.5f", index=False)

# fig = plt.figure()
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])
# ax1.plot(history_1.history['loss'], label='Relu')
# ax1.plot(history_2.history['loss'], label='Mish')
# ax1.plot(history_3.history['loss'], label='Proposed AF')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Train Loss')
# plt.grid(ls='--')
# plt.legend()

# left, bottom, width, height = 0.6, 0.4, 0.25, 0.25
# plt.axes([left, bottom, width, height])
# plt.plot(history_1.history['loss'], label='Relu')
# plt.plot(history_2.history['loss'], label='Mish')
# plt.plot(history_3.history['loss'], label='Proposed AF')
# plt.xlim(150, 200)
# plt.ylim(0, 0.2)
# plt.grid(ls='--')
# plt.xlabel('Epochs')
# plt.ylabel('Train Loss')
# plt.savefig(os.path.join('NIN-Train loss.png'))
#
# fig_1 = plt.figure()
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig_1.add_axes([left, bottom, width, height])
# ax1.plot(history_1.history['val_error'], label='Relu')
# ax1.plot(history_2.history['val_error'], label='Mish')
# ax1.plot(history_3.history['val_error'], label='Proposed AF')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Train Loss')
# plt.grid(ls='--')
# plt.legend()
#
# left, bottom, width, height = 0.6, 0.4, 0.25, 0.25
# plt.axes([left, bottom, width, height])
# plt.plot(history_1.history['val_error'], label='Relu')
# plt.plot(history_2.history['val_error'], label='Mish')
# plt.plot(history_3.history['val_error'], label='Proposed AF')
# plt.xlim(150, 200)
# plt.ylim(0, 0.15)
# plt.grid(ls='--')
# plt.xlabel('Epochs')
# plt.ylabel('Train Loss')
# plt.savefig(os.path.join('NIN-Test error.png'))

_, error_1 = model_1.evaluate(x_test, y_test)
_, error_2 = model_2.evaluate(x_test, y_test)
_, error_3 = model_3.evaluate(x_test, y_test)

print('relu:', error_1)
print('mish:', error_2)
print('my fun:', error_3)