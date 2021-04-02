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

methods = ["ienks-bundle", "etks_hybrid", "etks_classic"]
stat = 'anal'
tanl = 0.10
mda = 'false'
total_lag = 53
total_ens = 44

f = h5.File('./processed_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0")+ '_nanl_20000_burn_05000_mda_' + mda + '.h5', 'r')


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



fig = plt.figure()
ax0 = fig.add_axes([.935, .100, .02, .830])
ax1a = fig.add_axes([.070, .100, .410, .25])
ax1b = fig.add_axes([.070, .390, .410, .25])
ax1c = fig.add_axes([.070, .680, .410, .25])
ax2a = fig.add_axes([.520, .100, .410, .25])
ax2b = fig.add_axes([.520, .390, .410, .25])
ax2c = fig.add_axes([.520, .680, .410, .25])



color_map = sns.color_palette("husl", 101)
max_scale = 0.30
min_scale = 0.00

rmse_ax_list = [ax1a, ax1b, ax1c]
spread_ax_list = [ax2a, ax2b, ax2c]

for i in range(1,3):
    rmse, spread = find_optimal_values(methods[i], stat, f)
    sns.heatmap(rmse, linewidth=0.5, ax=rmse_ax_list[i], cbar_ax=ax0, vmin=min_scale, vmax=max_scale, cmap=color_map)
    sns.heatmap(spread, linewidth=0.5, ax=spread_ax_list[i], cbar_ax=ax0, vmin=min_scale, vmax=max_scale, cmap=color_map)



x_labs = []
for i in range(15,total_ens,2):
    x_labs.append(str(i))

y_labs = []
y_tics = []

y_vals = np.arange(1,total_lag, 3)
tic_vals = range(len(y_vals), 0, -1)
for i in range(len(y_vals)):
    if i % 3 == 0:
        y_labs.append(str(y_vals[i]))
        y_tics.append(tic_vals[i])
y_labs.append(str(y_vals[-1]))
y_tics.append(tic_vals[-1])

ax0.tick_params(
        labelsize=20)
ax1a.tick_params(
        labelsize=20)
ax1b.tick_params(
        labelsize=20,
        labelbottom=False)
ax1c.tick_params(
        labelsize=20,
        labelbottom=False)

ax2a.tick_params(
        labelsize=20,
        labelleft=False)
ax2b.tick_params(
        labelleft=False,
        labelbottom=False)
ax2c.tick_params(
        labelleft=False,
        labelbottom=False)

ax1a.set_yticks(y_tics)
ax1b.set_yticks(y_tics)
ax1c.set_yticks(y_tics)
ax2a.set_yticks(y_tics)
ax2b.set_yticks(y_tics)
ax2c.set_yticks(y_tics)
ax1a.set_yticklabels(y_labs, va="bottom", rotation=0)
ax1b.set_yticklabels(y_labs, va="bottom", rotation=0)
ax1c.set_yticklabels(y_labs, va="bottom", rotation=0)

ax2a.set_xticklabels(x_labs, rotation=0)
ax1a.set_xticklabels(x_labs, rotation=0)

if stat == 'anal':
    stat = 'smoother'

elif stat == 'filt':
    stat = 'filter'

elif stat == 'fore':
    stat = 'forecast'

if mda=="true":
    fig_title = "Tuned inflation, MDA, shift 1, $\Delta$t="+ str(tanl)

else:
    fig_title = "Tuned inflation, shift 1, $\Delta$t="+ str(tanl)


plt.figtext(.080, .931, 'ETKS classic ' + stat + ' RMSE', horizontalalignment='left', verticalalignment='bottom', fontsize=20)
plt.figtext(.920, .931, 'ETKS classic ' + stat + ' spread', horizontalalignment='right', verticalalignment='bottom', fontsize=20)
plt.figtext(.080, .641, 'ETKS hybrid ' + stat + ' RMSE', horizontalalignment='left', verticalalignment='bottom', fontsize=20)
plt.figtext(.920, .641, 'ETKS hybrid ' + stat + ' spread', horizontalalignment='right', verticalalignment='bottom', fontsize=20)
plt.figtext(.080, .351, 'IEnKS bundle ' + stat + ' RMSE', horizontalalignment='left', verticalalignment='bottom', fontsize=20)
plt.figtext(.920, .351, 'IEnKS bundle ' + stat + ' spread', horizontalalignment='right', verticalalignment='bottom', fontsize=20)
plt.figtext(.025, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .97, fig_title,
        horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
