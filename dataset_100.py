"Load CIFAR-100 dataset"
import pickle
import sys
import numpy as np
from keras import backend as K
from keras.utils import to_categorical

def load_batch(fpath, label_key='fine_labels'):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

path = '../cifar100/cifar-100-python/train'
path1 = '../cifar100/cifar-100-python/test'
def load_data():
    train_num = 50000
    train_x = np.zeros(shape=(train_num, 3, 32, 32))
    train_y = np.zeros(shape=(train_num))
    test_num = 10000
    test_x = np.zeros(shape=(test_num, 3, 32, 32))
    test_y = np.zeros(shape=(test_num))

    train_x[:, :, :, :], train_y[::] = load_batch(path)
    test_x[:], test_y[:] = load_batch(path1)

    if K.image_data_format() == 'channels_last':
        x_test = test_x.transpose(0, 2, 3, 1)
        x_train = train_x.transpose(0, 2, 3, 1)
    else:
        print("channels_first")

    X_train = x_train / 255.0
    X_test = x_test / 255.0

    Y_train = to_categorical(train_y, 100)
    Y_test = to_categorical(test_y, 100)
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)


