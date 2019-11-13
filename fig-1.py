"The graphical depiction of PATS"
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def pats(x, k=1/4):
    f = x*np.arctan(np.pi*k/(1+np.exp(-x)))
    return f

x = np.linspace(-10, 10)
arc1 = pats(x, k=1/4)
arc2 = pats(x, k=1/2)
arc3 = pats(x, k=5/8)
arc4 = pats(x, k=3/4)

fig = plt.figure(figsize=(5, 6))
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
plt.plot(x, arc1, 'y', label='k=1/4')
plt.plot(x, arc2, 'r', label='k=1/2')
plt.plot(x, arc3, 'b', label='k=5/8')
plt.plot(x, arc4, 'g', label='k=3/4')
plt.grid(linestyle='--')
plt.title('PATS Activation Function')
plt.legend(loc="upper left")
plt.savefig('k-4-af.png', dpi=650)
#plt.show()

del x

#Diff
def diff_pats(x, k=1/4):
    f = np.arctan(np.pi*k/(1+np.exp(-x))) + k*np.pi*x*np.exp(-x)/((1+np.exp(-x))**2+(k*np.pi)**2)
    return f

x = np.linspace(-10, 10)
de1 = diff_pats(x, k=1/4)
de2 = diff_pats(x, k=1/2)
de3 = diff_pats(x, k=5/8)
de4 = diff_pats(x, k=3/4)

fig1 = plt.figure(figsize=(5, 6))
ax1 = axisartist.Subplot(fig1, 111)
fig1.add_axes(ax1)
ax1.axis[:].set_visible(False)

ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.axis["x"] = ax1.new_floating_axis(0, 0)
ax1.axis["x"].set_axisline_style("->", size=1.5)
ax1.axis["y"] = ax1.new_floating_axis(1, 0)
ax1.axis["y"].set_axisline_style("->", size=1.5)

plt.xlim(-10, 10)
plt.ylim(-2, 2)
plt.plot(x, de1, 'y--', label='k=1/4')
plt.plot(x, de2, 'r--', label='k=1/2')
plt.plot(x, de3, 'b--', label='k=5/8')
plt.plot(x, de4, 'g--', label='k=3/4')
plt.grid(linestyle='--')
plt.title('Derivative of PATS')
plt.legend(loc="upper left")
plt.savefig('k-4-diff.png', dpi=650)
#plt.show()