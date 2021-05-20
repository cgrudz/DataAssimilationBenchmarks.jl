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

method = "mlef-transform"
if method == "mlef-transform":
    scheme = "MLEF"

elif method == "mlef-n-transform":
    scheme = "MLEF-N"

elif method == "mlef-ls-transform":
    scheme = "MLEF-ls"

elif method == "mlef-ls-n-transform":
    scheme = "MLEF-ls-N"

stats = ["filt", "fore"]
tanl = 0.05
total_ens = 44
total_gammas = 9

f = h5.File('processed_filter_nonlinear_obs_state_diffusion_0.00_tanl_0.05_nanl_20000_burn_05000.h5', 'r')


def find_optimal_values(method, stat, data):
    tuning_stat = 'filt'
    tuned_rmse = np.array(f[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
    ens, gammas = np.shape(tuned_rmse_min_vals)
    
    stat_rmse = np.array(f[method +'_' + stat + '_rmse'])
    stat_spread = np.array(f[method + '_' + stat + '_spread'])

    rmse_vals = np.zeros([ens, gammas])
    spread_vals = np.zeros([ens, gammas])

    for i in range(ens):
        for j in range(gammas):
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
ax0 = fig.add_axes([.935, .075, .02, .875])
ax1a = fig.add_axes([.070, .075, .410, .40])
ax1b = fig.add_axes([.070, .550, .410, .40])
ax2a = fig.add_axes([.520, .075, .410, .40])
ax2b = fig.add_axes([.520, .550, .410, .40])

label_h_positions = [0.08, 0.932]
label_v_positions = [0.351, 0.641, 0.931 ]


color_map = sns.color_palette("icefire", as_cmap=True) 
max_scale = 0.01
min_scale = -0.01

rmse_ax_list = [ax1a, ax1b]
spread_ax_list = [ax2a, ax2b]

for i in range(2):
    stat = stats[i]
    if method == "mlef-n-transform":
        rmse1 = np.transpose(np.array(f[method +'_' + stat + '_rmse']))
        spread1 = np.transpose(np.array(f[method +'_' + stat + '_spread']))
        rmse2 = np.transpose(np.array(f['mlef-ls-n-transform_' + stat + '_rmse']))
        spread2 = np.transpose(np.array(f['mlef-ls-n-transform_' + stat + '_spread']))
        rmse = rmse1 - rmse2
        spread = spread1 - spread2
    elif method == "mlef-transform":
        rmse1, spread1 = find_optimal_values(method, stat, f)
        rmse2, spread2 = find_optimal_values("mlef-ls-transform", stat, f)
        #ipdb.set_trace()
        rmse = rmse1 - rmse2
        spread = spread1 - spread2


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

y_vals = np.arange(1,total_gammas)
tic_vals = range(len(y_vals), 0, -1)
for i in range(len(y_vals)):
    if i % 2 == 0:
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
ax2a.tick_params(
        labelsize=20,
        labelleft=False)
ax2b.tick_params(
        labelleft=False,
        labelbottom=False)
ax1a.set_yticks(y_tics)
ax1b.set_yticks(y_tics)
ax2a.set_yticks(y_tics)
ax2b.set_yticks(y_tics)
ax1a.set_yticklabels(y_labs, va="bottom", rotation=0)
ax1b.set_yticklabels(y_labs, va="bottom", rotation=0)

ax2a.set_xticklabels(x_labs, rotation=0)
ax1a.set_xticklabels(x_labs, rotation=0)


plt.figtext(.025, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
#plt.figtext(.5, .97, fig_title, horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
