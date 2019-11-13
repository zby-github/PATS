"Graphical description of common activation functions and PATS function"
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def mish(x):
    f = x*np.tanh(np.log(np.exp(x) + 1))
    return f

def relu(x):
    return np.where(x < 0, 0, x)

def pats(x, k=5/8):
    f = x*np.arctan(np.pi*k/(1+np.exp(-x)))
    return f

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def swish(x):
    s = x/(1+np.exp(-x))
    return s

x = np.linspace(-10, 10)
ouraf = pats(x, k=5/8)
Mish = mish(x)
Relu = relu(x)
sig = sigmoid(x)
tanh = tanh(x)
swish = swish(x)

fig = plt.figure(figsize=(6, 6))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
ax.axis[:].set_visible(False)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["x"].set_axisline_style("->", size=1.5)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("->", size=1.5)

plt.xlim(-10, 10)
plt.ylim(-5, 5)
plt.plot(x, ouraf, label='PATS(k=5/8)')
plt.plot(x, Mish, label='Mish')
plt.plot(x, Relu, label='Relu')
plt.plot(x, tanh, label='Tanh')
plt.plot(x, sig, label='Sigmoid')
plt.plot(x, swish, label='Swish')
plt.grid(linestyle='--')
plt.legend()
plt.savefig('AF-3.png', dpi=650)
#plt.show()