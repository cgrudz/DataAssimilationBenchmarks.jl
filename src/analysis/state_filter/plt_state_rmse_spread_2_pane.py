import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
import h5py as h5

method = 'etkf'
stat = 'anal'
tanl = 0.05


f = h5.File('./processed_filter_state_diffusion_0.00_tanl_0.05_nanl_40000_burn_05000.h5', 'r')

rmse = np.array(f[method +'_' + stat + '_rmse'])
rmse = np.transpose(rmse)
spread = np.array(f[method + '_' + stat + '_spread'])
spread = np.transpose(spread)


fig = plt.figure()
ax3 = fig.add_axes([.460, .13, .02, .70])
ax2 = fig.add_axes([.940, .13, .02, .70])
ax1 = fig.add_axes([.530, .13, .390, .70])
ax0 = fig.add_axes([.060, .13, .390, .70])


color_map = sns.color_palette("husl", 101)


sns.heatmap(rmse, linewidth=0.5, ax=ax0, cbar_ax=ax3, vmin=0.01, vmax=1.0, cmap=color_map)
sns.heatmap(spread, linewidth=0.5, ax=ax1, cbar_ax=ax2, vmin=0.01, vmax=1.0, cmap=color_map)


ax2.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False)

ax0.tick_params(
        labelsize=20)

ax3.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)
ax2.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)


x_labs = []
counter = 0
for i in range(14,42):
    if counter % 3 == 0:
        x_labs.append(str(i))
        counter += 1
    else:
        x_labs.append('')
        counter += 1


y_labs = []
y_vals = np.linspace(1.0, 1.2, 21)
for i in range(len(y_vals)):
    if i % 3 == 0:
        y_labs.append(str(np.around(y_vals[i],2)))
    else:
        y_labs.append('')


y_labs = y_labs[::-1]

ax1.set_xlim([0,28])
ax0.set_xlim([0,28])
ax1.set_xticks(range(0,28))
ax0.set_xticks(range(0,28))
ax1.set_xticklabels(x_labs)
ax0.set_xticklabels(x_labs)
ax1.set_ylim([21,0])
ax0.set_ylim([21,0])
ax0.set_yticks(range(0,21))
ax0.set_yticklabels(y_labs, va='top')
ax1.set_yticks(range(0,21))

if stat == 'anal':
    stat = 'Analysis'

elif stat == 'fore':
    stat = 'Forecast'

plt.figtext(.2525, .87, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .87, stat + ' spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.015, .52, r'Inflation level', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .04, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .95, method + ' ' + stat, horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
