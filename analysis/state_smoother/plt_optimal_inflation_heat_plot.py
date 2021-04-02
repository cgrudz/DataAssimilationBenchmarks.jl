import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
from matplotlib.colors import LogNorm

method = 'etks'
version = 'classic'
tanl = 0
nanl = 45000
burn = 5000
wlks = [0.0000, 0.0001, 0.0010, 0.0100]
diff = 0.1
stat = 'smooth'

inflation = np.zeros([len(wlks), len(range(1, 52, 5)), len(range(14,42))])
w = 0

for wlk in wlks:
    f = open('./processed_'+ version + '_smoother_param_rmse_spread_nanl_40000_tanl_0.05_burn_5000_wlk_' + str(wlk).ljust(6, '0') + '_diff_' + str(diff) + '.txt', 'rb')
    tmp = pickle.load(f)
    f.close()

    inflation[w, :, :] = tmp[method + '_optimal_inflation']
    w += 1

fig = plt.figure()
ax2 = fig.add_axes([.935, .080, .01, .867])

ax1 = fig.add_axes([.504, .08, .425, .427])
ax0 = fig.add_axes([.070, .08, .425, .427])
ax4 = fig.add_axes([.504, .520, .425, .427])
ax3 = fig.add_axes([.070, .520, .425, .427])


color_map_state = sns.color_palette("viridis", 101)

sns.heatmap(inflation[0, :, :], linewidth=0.5, ax=ax3, cbar_ax=ax2, vmin=1.0, vmax=1.2, cmap=color_map_state)
sns.heatmap(inflation[1, :, :], linewidth=0.5, ax=ax4, vmin=1.0, vmax=1.2, cmap=color_map_state, cbar=False)
sns.heatmap(inflation[2, :, :], linewidth=0.5, ax=ax0, vmin=1.0, vmax=1.2, cmap=color_map_state, cbar=False)
sns.heatmap(inflation[3, :, :], linewidth=0.5, ax=ax1, vmin=1.0, vmax=1.2, cmap=color_map_state, cbar=False)


ax2.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False)

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
        labelbottom=False,
        right=False)

ax4.tick_params(
        labelleft=False,
        labelbottom=False,
        right=False)

x_labs = []
for i in range(14,44,3):
    x_labs.append(str(i))

y_labs = []
y_vals = range(1, 52, 5)
j = 0
for i in range(len(y_vals)):
    if j % 2 == 0:
        y_labs.append(str(np.around(y_vals[i],2)))
    else:
        y_labs.append('')
    
    j+=1


y_labs = y_labs[::-1]

ax1.set_xticks(np.arange(0,29,3) + 0.5)
ax0.set_xticks(np.arange(0,29,3) + 0.5)
ax3.set_xticks(np.arange(0,29,3) + 0.5)
ax4.set_xticks(np.arange(0,29,3) + 0.5)
ax1.set_xticklabels(x_labs, ha='center')
ax0.set_xticklabels(x_labs, ha='center')
ax0.set_ylim([11,0])
ax1.set_ylim([11,0])
ax3.set_ylim([11,0])
ax4.set_ylim([11,0])
ax0.set_yticks(np.arange(0,11) + 0.5)
ax0.set_yticklabels(y_labs, va='center')
ax3.set_yticks(np.arange(0,11) + 0.5)
ax3.set_yticklabels(y_labs, va='center')
ax1.set_yticks(np.arange(0,11) + 0.5)
if stat == 'smooth':
    stat = 'Smoother'

elif stat == 'filter':
    stat = 'Filter'

elif stat == 'fore':
    stat = 'Forecast'

if method == 'enks':
    method = 'EnKS'

elif method == 'etks':
    method = 'ETKS'



plt.figtext(.2, .96, 'Parameter walk std ' + str(wlks[0]), horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .96, 'Parameter walk std ' + str(wlks[1]), horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.04, .52, r'Lag length', horizontalalignment='right', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.2, .02, r'Parameter walk std ' + str(wlks[2]), horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .02, r'Parameter walk std ' + str(wlks[3]), horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.5, .965, method + ' ' + version + ' - optimal inflation values', horizontalalignment='center', verticalalignment='bottom', fontsize=24)

plt.show()
