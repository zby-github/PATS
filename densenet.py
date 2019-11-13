# -*- coding: utf-8 -*-
"""
Test on CIFAR-10 using DenseNet-100 network model
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, \
    concatenate
from dataset_10 import load_data
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd

def error(y_true, y_pred):
    Error = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
    return Error

def mish(x):
    return x*K.tanh(K.softplus(x))

def atan(x):
    return tf.atan(x)

p = np.random.uniform(1/2, 3/4, 1)
def pats(x, k=p[0]):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

growth_rate = 12
depth = 100
compression = 0.5

img_rows, img_cols = 32, 32
img_channels = 3
num_classes = 10
batch_size = 32 # 64 or 32 or other
epochs = 200
iterations = 391
weight_decay = 1e-4

if ('tensorflow' == K.backend()):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 120:
        return 0.01
    if epoch < 150:
        return 0.001
    return 0.0001

def densenet_model(img_input, classes_num, af):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)
    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation(af)(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1, 1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def single(x):
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1, 1))
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x, blocks, nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x, concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    nblocks = (depth - 4) // 6
    nchannels = growth_rate * 2

    x = conv(img_input, nchannels, (3, 3))
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output_0 = densenet_model(img_input, num_classes, af='relu')
    output_1 = densenet_model(img_input, num_classes, af=mish)
    output_2 = densenet_model(img_input, num_classes, af=pats)

    model_0 = Model(img_input, output_0)
    model_1 = Model(img_input, output_1)
    model_2 = Model(img_input, output_2)

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

    model_0.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])
    model_1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])
    model_2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])

    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]

    #set data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, fill_mode='constant', cval=0.)
    datagen.fit(X_train)
    # start training
    history_0 = model_0.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, validation_data=(X_test, Y_test))
    pd.DataFrame.from_dict(history_0.history).to_csv("densenet_relu_log.csv", float_format="%.5f", index=False)

    history_1 = model_1.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, validation_data=(X_test, Y_test))
    pd.DataFrame.from_dict(history_1.history).to_csv("densenet_mish_log.csv", float_format="%.5f", index=False)

    history_2 = model_2.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks, validation_data=(X_test, Y_test))
    pd.DataFrame.from_dict(history_2.history).to_csv("densenet_myfun_log.csv", float_format="%.5f", index=False)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    plt.plot(history_0.history['loss'], label='Relu')
    plt.plot(history_1.history['loss'], label='Mish')
    plt.plot(history_2.history['loss'], label='Proposed AF')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join('Densenet-Train loss.png'))

    fig_1 = plt.figure()
    ax2 = fig_1.add_subplot(111)
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    plt.plot(history_0.history['val_error'], label='Relu')
    plt.plot(history_1.history['val_error'], label='Mish')
    plt.plot(history_2.history['val_error'], label='Proposed AF')
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join('Densenet-Test error.png'))

    _, error_0 = model_0.evaluate(X_test, Y_test)
    _, error_1 = model_1.evaluate(X_test, Y_test)
    _, error_2 = model_2.evaluate(X_test, Y_test)
    print('Relu:', error_0)
    print('Mish:', error_1)
    print('My fun:', error_2)
