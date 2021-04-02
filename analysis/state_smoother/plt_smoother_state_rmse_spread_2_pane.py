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
import h5py as h5

method = "etks_hybrid"
stat = 'fore'
tanl = 0.10
mda = 'false'

f = h5.File('./processed_all_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0") +  '_nanl_20000_burn_05000_mda_' + mda + '.h5', 'r')


def find_optimal_values(method, stat, data):
    tuning_stat = 'anal'
    tuned_rmse = np.array(f[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
    lag, ens = np.shape(tuned_rmse_min_vals)
    
    stat_rmse = np.array(f[method +'_' + stat + '_rmse'])
    stat_spread = np.array(f[method + '_' + stat + '_spread'])

    rmse_vals = np.zeros([lag, ens])
    spread_vals = np.zeros([lag, ens])

    for i in range(lag):
        for j in range(ens):
            min_val = tuned_rmse_min_vals[i,j]
            indx = tuned_rmse[i,:,j] == min_val

            rmse_vals[i,j] = stat_rmse[i, indx, j]
            spread_vals[i,j] = stat_spread[i, indx, j]
   
    rmse_vals = np.transpose(rmse_vals)
    spread_vals = np.transpose(spread_vals)

    return [rmse_vals, spread_vals]

rmse, spread = find_optimal_values(method, stat, f)


fig = plt.figure()
ax3 = fig.add_axes([.460, .13, .02, .80])
ax2 = fig.add_axes([.940, .13, .02, .80])
ax1 = fig.add_axes([.530, .13, .390, .80])
ax0 = fig.add_axes([.060, .13, .390, .80])



color_map = sns.color_palette("husl", 101)
max_scale = 0.50
min_scale = 0.00


sns.heatmap(rmse, linewidth=0.5, ax=ax0, cbar_ax=ax3, vmin=min_scale, vmax=max_scale, cmap=color_map)
sns.heatmap(spread, linewidth=0.5, ax=ax1, cbar_ax=ax2, vmin=min_scale, vmax=max_scale, cmap=color_map)


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
for i in range(15,44,2):
    x_labs.append(str(i))

y_labs = []
y_vals = np.arange(1,53, 3)
for i in range(len(y_vals)):
    if i % 1 == 0:
        y_labs.append(str(y_vals[i]))
    else:
        y_labs.append('')


y_labs = y_labs[::-1]

ax1.set_xticks(range(0,15))
ax0.set_xticks(range(0,15))
ax1.set_xticklabels(x_labs)
ax0.set_xticklabels(x_labs)
#ax1.set_ylim([9,1])
#ax0.set_ylim([9,1])
ax0.set_yticks(range(18))
ax0.set_yticklabels(y_labs, va='bottom')
ax1.set_yticks(range(18))

if stat == 'anal':
    stat = 'smoother'

elif stat == 'filt':
    stat = 'filter'

elif stat == 'fore':
    stat = 'forecast'

plt.figtext(.2525, .05, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .05, stat + ' spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.015, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .04, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .97, method + ', ' + ' mda ' + mda + ', ' + stat +  ' statistics, optimally tuned inflation, shift 1',
        horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
