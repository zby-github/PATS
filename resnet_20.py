# -*- coding: utf-8 -*-
"""
Test on CIFAR-10 using ResNet-20 network model
"""
import matplotlib.pyplot as plt
import keras
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical
import os
import tarfile
import sys
import pickle
import tensorflow as tf
import pandas as pd
from dataset_10 import load_data

# set GPU memory 
if('tensorflow' == K.backend()):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n', '--stack_n', type=int, default=3, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
# parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
#                 help='dataset. (default: cifar10)')

args = parser.parse_args()

def error(y_true, y_pred):
    Error = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
    return Error

stack_n = args.stack_n
layers = 6 * stack_n + 2
num_classes = 10
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 50000 // batch_size + 1
print('iterations:', iterations)
weight_decay = 1e-4

def sch_sgd(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 120:
        return 0.01
    if epoch < 150:
        return 0.001
    return 0.0001

def resnet(img_input, classes_num, stack, a):
    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)
        o1 = Activation(a)(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation(a)(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation(a)(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

def atan(x):
    return tf.atan(x)

def pats_0(x, k=1/4):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

def pats_1(x, k=1/2):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

def pats_2(x, k=5/8):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

def pats_3(x, k=3/4):
    f = x*atan(np.pi*k/(1+K.exp(-x)))
    return f

if __name__ == '__main__':
    print("========================================") 
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2)) 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))

    train_x, train_y, test_x, test_y = load_data()

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))

    output_0 = resnet(img_input, num_classes, stack_n, pats_0)
    output_1 = resnet(img_input, num_classes, stack_n, pats_1)
    output_2 = resnet(img_input, num_classes, stack_n, pats_2)
    output_3 = resnet(img_input, num_classes, stack_n, pats_3)

    resnet_0 = Model(img_input, output_0)
    resnet_1 = Model(img_input, output_1)
    resnet_2 = Model(img_input, output_2)
    resnet_3 = Model(img_input, output_3)

    # set optimizer
    sgd = SGD(lr=0.1, nesterov=True, momentum=0.9)

    resnet_0.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])
    resnet_1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])
    resnet_2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])
    resnet_3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[error])

    cbks_sgd = [LearningRateScheduler(sch_sgd)]

    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 fill_mode='constant', cval=0.)
    datagen.fit(train_x)

    history_0 = resnet_0.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                                       steps_per_epoch=iterations, epochs=epochs, callbacks=cbks_sgd,
                                       validation_data=(test_x, test_y))
    #pd.DataFrame.from_dict(history_0.history).to_csv("af_1_log.csv", float_format="%.5f", index=False)

    history_1 = resnet_1.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), steps_per_epoch=iterations,
                                       epochs=epochs, callbacks=cbks_sgd, validation_data=(test_x, test_y))
    #pd.DataFrame.from_dict(history_1.history).to_csv("af_2_log.csv", float_format="%.5f", index=False)

    history_2 = resnet_2.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), steps_per_epoch=iterations,
                                       epochs=epochs, callbacks=cbks_sgd, validation_data=(test_x, test_y))
    #pd.DataFrame.from_dict(history_2.history).to_csv("af_3_log.csv", float_format="%.5f", index=False)

    history_3 = resnet_3.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs,
                                       callbacks=cbks_sgd, validation_data=(test_x, test_y))
    #pd.DataFrame.from_dict(history_3.history).to_csv("af_4_log.csv", float_format="%.5f", index=False)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.spines['top'].set_color('none')
    # ax1.spines['right'].set_color('none')
    # plt.plot(history_0.history['loss'], label='Proposed AF(k=1/4)')
    # plt.plot(history_1.history['loss'], label='Proposed AF(k=1/2)')
    # plt.plot(history_2.history['loss'], label='Proposed AF(k=5/8)')
    # plt.plot(history_3.history['loss'], label='Proposed AF(k=3/4)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Train loss')
    # plt.grid(ls='--')
    # plt.legend()
    # plt.savefig(os.path.join('Resnet-N-Train loss.png'))
    #
    # fig_1 = plt.figure()
    # ax2 = fig_1.add_subplot(111)
    # ax2.spines['top'].set_color('none')
    # ax2.spines['right'].set_color('none')
    # plt.plot(history_0.history['val_error'], label='Proposed AF(k=1/4)')
    # plt.plot(history_1.history['val_error'], label='Proposed AF(k=1/2)')
    # plt.plot(history_2.history['val_error'], label='Proposed AF(k=5/8)')
    # plt.plot(history_3.history['val_error'], label='Proposed AF(k=3/4)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Test Error')
    # plt.grid(ls='--')
    # plt.legend()
    # plt.savefig(os.path.join('Resnet-N-Test error.png'))

    loss_0, error_0 = resnet_0.evaluate(test_x, test_y)
    loss_1, error_1 = resnet_1.evaluate(test_x, test_y)
    loss_2, error_2 = resnet_2.evaluate(test_x, test_y)
    loss_3, error_3 = resnet_3.evaluate(test_x, test_y)

    print('K=1/4:', error_0)
    print('K=1/2:', error_1)
    print('K=5/8:', error_2)
    print('K=3/4:', error_3)