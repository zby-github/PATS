"""
Training loss and test error on CIFAR-10 using DenseNet network model
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

data1 = pd.read_csv('../densenet_relu_log.csv')
densenet100_relu_los = data1['loss'].values
densenet100_relu_err = data1['val_error'].values*100
print(densenet100_relu_err[-1])

data2 = pd.read_csv('../densenet_mish_log.csv')
densenet100_mish_err = data2['val_error'].values*100
densenet100_mish_los = data2['loss'].values
print(densenet100_mish_err[-1])

data3 = pd.read_csv('../densenet_myfun_log.csv')
densenet100_myfun_err = data3['val_error'].values*100
densenet100_myfun_los = data3['loss'].values
print(densenet100_myfun_err[-1])

#Test Error
fig = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(densenet100_relu_err, label='Relu')
ax1.plot(densenet100_mish_err, label='Mish')
ax1.plot(densenet100_myfun_err, label='PATS')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Test Error (%)')
plt.grid(ls='--')
plt.legend()

left, bottom, width, height = 0.6, 0.3, 0.25, 0.25
plt.axes([left, bottom, width, height])
plt.plot(densenet100_relu_err)
plt.plot(densenet100_mish_err)
plt.plot(densenet100_myfun_err)
plt.xlim(150, 200)
plt.ylim(5.5, 7.0)
plt.grid(ls='--')
plt.xlabel('Epochs')
plt.ylabel('Test Error (%)')
plt.savefig(os.path.join('Densenet-Test Error.png'), dpi=650)
plt.show()

#Train Loss
fig_1 = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig_1.add_axes([left, bottom, width, height])
ax1.plot(densenet100_relu_los, label='Relu')
ax1.plot(densenet100_mish_los, label='Mish')
ax1.plot(densenet100_myfun_los, label='PATS')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss')
plt.grid(ls='--')
plt.legend()

left, bottom, width, height = 0.6, 0.3, 0.25, 0.25
plt.axes([left, bottom, width, height])
plt.plot(densenet100_relu_los, label='Relu')
plt.plot(densenet100_mish_los, label='Mish')
plt.plot(densenet100_myfun_los, label='PATS')
plt.xlim(150, 200)
plt.ylim(0.12, 0.18)
plt.grid(ls='--')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.savefig(os.path.join('DenseNet-Train Loss.png'), dpi=650)
plt.show()

