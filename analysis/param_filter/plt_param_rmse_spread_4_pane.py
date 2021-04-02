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
from matplotlib.colors import LogNorm

method = 'enkf'
wlk = 0.0100
stat = 'anal'

f = h5.File('./processed_filter_param_diffusion_0.00_wlk_' + str(wlk).ljust(6,"0") + '_tanl_0.05_nanl_40000_burn_05000.h5', 'r')

state_rmse = np.array(f[method +'_' + stat + '_rmse']).transpose()
state_spread = np.array(f[method + '_' + stat + '_spread']).transpose()
param_rmse = np.array(f[method +'_para_rmse']).transpose()
param_spread = np.array(f[method + '_para_spread']).transpose()

fig = plt.figure()
ax2 = fig.add_axes([.935, .520, .01, .427])
ax5 = fig.add_axes([.935, .080, .01, .427])

ax1 = fig.add_axes([.504, .08, .425, .427])
ax0 = fig.add_axes([.070, .08, .425, .427])
ax4 = fig.add_axes([.504, .520, .425, .427])
ax3 = fig.add_axes([.070, .520, .425, .427])

color_map_state = sns.color_palette("husl", 101)
color_map_params = sns.color_palette("cubehelix", 100)


log_norm = LogNorm(vmin=0.00001, vmax=1.0)

sns.heatmap(state_rmse, linewidth=0.5, ax=ax3, cbar_ax=ax2, vmin=0.01, vmax=1.0, cmap=color_map_state)
sns.heatmap(state_spread, linewidth=0.5, ax=ax4, vmin=0.01, vmax=1.0, cmap=color_map_state, cbar=False)
sns.heatmap(param_rmse, linewidth=0.5, ax=ax0, cbar_ax=ax5, cmap=color_map_params, norm=log_norm)
sns.heatmap(param_spread, linewidth=0.5, ax=ax1, cmap=color_map_params, cbar=False, norm=log_norm)


ax2.tick_params(
        labelsize=20)

ax5.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False,
        right=True)

ax0.tick_params(
        labelsize=20)

ax2.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)

ax3.tick_params(
        labelsize=20,
        labelbottom=False)

ax4.tick_params(
        labelleft=False,
        labelbottom=False)

x_labs = []
for i in range(14,44,3):
    x_labs.append(str(i))


y_labs = []
y_vals = np.linspace(1.0, 1.2, 21)
for i in range(len(y_vals)):
    if i % 6 == 0:
        y_labs.append(str(np.around(y_vals[i],2)))
    else:
        y_labs.append('')

y_labs = y_labs[::-1]

if stat == 'anal':
    stat = 'Analysis'

elif stat == 'fore':
    stat = 'Forecast'


ax1.set_xticks(np.arange(0,29,3) + 0.5)
ax0.set_xticks(np.arange(0,29,3) + 0.5)
ax3.set_xticks(np.arange(0,29,3) + 0.5)
ax4.set_xticks(np.arange(0,29,3) + 0.5)
ax1.set_xticklabels(x_labs, ha='center')
ax0.set_xticklabels(x_labs, ha='center')

ax1.set_ylim([21,0])
ax0.set_ylim([21,0])
ax0.set_yticks(range(21))
ax0.set_yticklabels(y_labs, va='center')
ax3.set_yticks(range(21))
ax3.set_yticklabels(y_labs, va='center')
ax1.set_yticks(range(21))
#ax1.set_yticklabels(y_labs, va='top', rotation='90')
plt.figtext(.2, .96, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .96, stat + ' Spread', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.03, .7335, r'State vector', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.03, .2935, r'F parameter', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.02, .52, r'Inflation level', horizontalalignment='right', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.5, .965, method + ' ' + stat + ' Param wlk ' + str(wlk), horizontalalignment='center', verticalalignment='bottom', fontsize=24)

plt.show()
