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

method = "etks_classic"
if method == "etks_classic":
    scheme = "ETKS Classic"

elif method == "etks_single_iteration":
    scheme = "SIETKS"

elif method == "enks-n-dual_classic":
    scheme = "EnKS-N-dual Classic"

elif method == "enks-n-dual_single_iteration":
    scheme = "SIETKS-N-dual"

elif method == "ienks-bundle":
    scheme = "IEnKS bundle"

elif method == "ienks-n-bundle":
    scheme = "IEnKS-N bundle"

elif method == "ienks-transform":
    scheme = "IEnKS transform"

elif method == "ienks-n-transform":
    scheme = "IEnKS-N transform"

elif method == "etks_adaptive_single_iteration":
    scheme = "SIETKS-Q"

stats = ["post", "filt", "fore"]
tanl = 0.05
mda = "false"
total_lag = 53
total_ens = 44
shift = 1

f = h5.File('./processed_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0")+ '_nanl_20000_burn_05000_mda_' + mda + '_shift_' + str(shift).rjust(3, "0") + '.h5', 'r')


def find_optimal_values(method, stat, data):
    tuning_stat = 'post'
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
            tmp_rmse = stat_rmse[i, indx, j]
            tmp_spread = stat_spread[i, indx, j]
            if len(tmp_rmse) > 1:
                tmp_rmse = tmp_rmse[0]
                tmp_spread = tmp_spread[0]

            rmse_vals[i,j] = tmp_rmse
            spread_vals[i,j] = tmp_spread
   
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

label_h_positions = [0.08, 0.932]
label_v_positions = [0.351, 0.641, 0.931 ]


color_map = sns.color_palette("husl", 101)
max_scale = 0.30
min_scale = 0.00

rmse_ax_list = [ax1a, ax1b, ax1c]
spread_ax_list = [ax2a, ax2b, ax2c]

for i in range(3):
    stat = stats[i]
    if method == "etks_adaptive_single_iteration" or \
       method == "enks-n-dual_classic" or \
       method == "enks-n-dual_single_iteration" or \
       method == "ienks-n-bundle" or \
       method == "ienks-n-transform":
        rmse = np.transpose(np.array(f[method +'_' + stat + '_rmse']))
        spread = np.transpose(np.array(f[method +'_' + stat + '_spread']))
    else:
        rmse, spread = find_optimal_values(method, stat, f)

    sns.heatmap(rmse, linewidth=0.5, ax=rmse_ax_list[i], cbar_ax=ax0, vmin=min_scale, vmax=max_scale, cmap=color_map)
    sns.heatmap(spread, linewidth=0.5, ax=spread_ax_list[i], cbar_ax=ax0, vmin=min_scale, vmax=max_scale, cmap=color_map)

    if stat == 'postl':
        stat = 'smoother'

    elif stat == 'filt':
        stat = 'filter'

    elif stat == 'fore':
        stat = 'forecast'

    
    plt.figtext(label_h_positions[0], label_v_positions[i], scheme + " " + stat + ' RMSE', 
            horizontalalignment='left', verticalalignment='bottom', fontsize=20)
    plt.figtext(label_h_positions[1], label_v_positions[i], scheme + " " + stat + ' spread', 
            horizontalalignment='right', verticalalignment='bottom', fontsize=20)



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


if mda=="true":
    fig_title = "MDA, shift 1, $\Delta$t="+ str(tanl)

else:
    fig_title = "Shift 1, $\Delta$t="+ str(tanl)


plt.figtext(.025, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .97, fig_title, horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
