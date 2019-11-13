"Load CIFAR-10 dataset"
from keras.utils import to_categorical
import sys
import pickle
import numpy as np
from keras import backend as K

def read_data():
    train_num = 50000
    train_x = np.zeros(shape=(train_num, 3, 32, 32))
    train_y = np.zeros(shape=(train_num))
    test_num = 10000
    test_x = np.zeros(shape=(test_num, 3, 32, 32))
    test_y = np.zeros(shape=(test_num))

    for i in range(1, 6):
        begin = (i - 1) * 10000
        end = i * 10000
        train_x[begin:end, :, :, :], train_y[begin:end] = load_batch('../cifar-10-batches-py/data_batch_' + str(i))
    test_x[:], test_y[:] = load_batch('../cifar-10-batches-py/test_batch')
    return train_x, train_y, test_x, test_y

def load_batch(fpath, label_key='labels'):
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

def load_data():
    train_x, train_y, test_x, test_y = read_data()

    if K.image_data_format() == 'channels_last':
        test_x = test_x.transpose(0, 2, 3, 1)
        train_x = train_x.transpose(0, 2, 3, 1)
    else:
        print("channels_first")

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)