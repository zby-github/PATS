"""
Training loss and test error on CIFAR-10 using ResNet-20 network model
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

data1 = pd.read_csv('../af_1_log.csv')
resnet20_K1_los = data1['loss'].values
resnet20_K1_err = data1['val_error'].values*100
print(resnet20_K1_err[-1])

data2 = pd.read_csv('../af_2_log.csv')
resnet20_K2_los = data2['loss'].values
resnet20_K2_err = data2['val_error'].values*100
print(resnet20_K2_err[-1])

data3 = pd.read_csv('../af_3_log.csv')
resnet20_K3_los = data3['loss'].values
resnet20_K3_err = data3['val_error'].values*100
print(resnet20_K3_err[-1])

data4 = pd.read_csv('../af_4_log.csv')
resnet20_K4_los = data4['loss'].values
resnet20_K4_err = data4['val_error'].values*100
print(resnet20_K4_err[-1])

#Test Error
fig = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(resnet20_K1_err, label='PATS (k=1/4)')
ax1.plot(resnet20_K2_err, label='PATS (k=1/2)')
ax1.plot(resnet20_K3_err, label='PATS (k=5/8)')
ax1.plot(resnet20_K4_err, label='PATS (k=3/4)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Test Error (%)')
plt.grid(ls='--')
plt.legend()

left, bottom, width, height = 0.6, 0.3, 0.25, 0.25
plt.axes([left, bottom, width, height])
plt.plot(resnet20_K1_err)
plt.plot(resnet20_K2_err)
plt.plot(resnet20_K3_err)
plt.plot(resnet20_K4_err)
plt.xlim(150, 200)
plt.ylim(8.0, 9.0)
plt.grid(ls='--')
plt.xlabel('Epochs')
plt.ylabel('Test Error (%)')
plt.savefig(os.path.join('Resnet-20-Test Error.png'), dpi=650)
plt.show()

#Train Loss
fig_1 = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig_1.add_axes([left, bottom, width, height])
ax1.plot(resnet20_K1_los, label='PATS (k=1/4)')
ax1.plot(resnet20_K2_los, label='PATS (k=1/2)')
ax1.plot(resnet20_K3_los, label='PATS (k=5/8)')
ax1.plot(resnet20_K4_los, label='PATS (k=3/4)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss')
plt.grid(ls='--')
plt.legend()

left, bottom, width, height = 0.6, 0.3, 0.25, 0.25
plt.axes([left, bottom, width, height])
plt.plot(resnet20_K1_los)
plt.plot(resnet20_K2_los)
plt.plot(resnet20_K3_los)
plt.plot(resnet20_K4_los)
plt.xlim(150, 200)
plt.ylim(0.1, 0.16)
plt.grid(ls='--')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.savefig(os.path.join('Resnet-20-Train Loss.png'), dpi=650)
plt.show()

