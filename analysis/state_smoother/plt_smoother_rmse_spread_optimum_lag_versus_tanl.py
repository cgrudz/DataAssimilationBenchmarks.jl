import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
import math
import h5py as h5

obs_un = 1.0
method_list = ["enks-n-primal_classic", "enks-n-primal_single_iteration", "lin-ienks-n-transform", "ienks-n-transform"]
#method_list = ["etks_classic", "etks_single_iteration", "lin-ienks-transform", "ienks-transform"]
stats = ["post", "filt", "fore"]
mda = "false"
#mda = "true"
markerlist = ['+', 'x', "d", "o", '^']
markersizes = [24, 24, 16, 16, 16]
color_list = ['#d95f02', '#7570b3', '#1b9e77']
total_lag = 53
shift = 1
tanls = [0.05, 0.10, 0.15, 0.20, 0.25]
total_tanls = len(tanls)

fig = plt.figure()
ax1 = fig.add_axes([.520, .10, .43, .72])
ax0 = fig.add_axes([.050, .10, .43, .72])

f = h5.File('processed_smoother_state_versus_tanl_diffusion_0.00_nanl_20000_burn_05000_mda_' +\
        mda + '_shift_' + str(shift).rjust(3, "0") + '.h5', 'r')

def find_optimal_values(method, stat, data):
    tuning_stat = 'post' 
    tuned_rmse = np.array(f[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_nan = np.isnan(tuned_rmse)
    tuned_rmse[tuned_rmse_nan] = np.inf
    stat_rmse = np.array(f[method +'_' + stat + '_rmse'])
    stat_spread = np.array(f[method + '_' + stat + '_spread'])

    dims = np.shape(tuned_rmse)

    if len(dims) == 2:
        tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
        num_tanls = len(tuned_rmse_min_vals)
        rmse_vals = np.zeros([num_tanls])
        spread_vals = np.zeros([num_tanls])

        for j in range(num_tanls):
            min_val = tuned_rmse_min_vals[j]
            indx = tuned_rmse[j,:]
            indx = indx == min_val
            tmp_rmse = stat_rmse[j,:]
            tmp_rmse = tmp_rmse[indx]
            tmp_spread = stat_spread[j,:]
            tmp_spread = tmp_spread[indx]
            if len(tmp_rmse) > 1:
                tmp_rmse = tmp_rmse[0]
                tmp_spread = tmp_spread[0]
            rmse_vals[j] = tmp_rmse
            spread_vals[j] = tmp_spread

    else:
        tuned_rmse_min_vals = np.min(tuned_rmse, axis=(1,2))
        num_tanls = len(tuned_rmse_min_vals)
        rmse_vals = np.zeros([num_tanls])
        spread_vals = np.zeros([num_tanls])

        for j in range(num_tanls):
            min_val = tuned_rmse_min_vals[j]
            indx = tuned_rmse[j,:,:]
            indx = indx == min_val
            tmp_rmse = stat_rmse[j, :, :]
            tmp_rmse = tmp_rmse[indx]
            tmp_spread = stat_spread[j, :, :]
            tmp_spread = tmp_spread[indx]
            if len(tmp_rmse) > 1:
                tmp_rmse = tmp_rmse[0]
                tmp_spread = tmp_spread[0]

            rmse_vals[j] = tmp_rmse
            spread_vals[j] = tmp_spread
   
    return [rmse_vals, spread_vals]


line_list = []
line_labs = []
j = 0
k = 0
for meth in method_list:
    for stat in stats:
        rmse, spread = find_optimal_values(meth, stat, f)
        l, = ax0.plot(tanls, rmse[:total_tanls], marker=markerlist[j], linewidth=2, markersize=markersizes[j], color=color_list[k])
        ax1.plot(tanls, spread[:total_tanls], marker=markerlist[j], linewidth=2, markersize=markersizes[j], color=color_list[k])
        line_list.append(l)

        if stat == 'post':
            stat_name = 'smoother'

        elif stat == 'filt':
            stat_name = 'filter'

        elif stat == 'fore':
            stat_name = 'forecast'

        if meth == "etks_classic":
            meth_name = "ETKS"
        elif meth == "etks_single_iteration":
            meth_name = "SIETKS"
        elif meth == "enks-n-primal_classic":
            meth_name = "EnKS-N"
        elif meth == "enks-n-primal_single_iteration":
            meth_name = "SIETKS-N"
        elif meth == "ienks-transform":
            meth_name = "IEnKS"
        elif meth == "ienks-n-transform":
            meth_name = "IEnKS-N"
        elif meth == "lin-ienks-transform":
            meth_name = "LIEnKS"
        elif meth == "lin-ienks-n-transform":
            meth_name = "LIEnKS-N"
        
        line_labs.append(meth_name + ' ' + stat_name)
        k+=1 
        k = k % 3

    j+=1


ax1.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True)

ax0.tick_params(
        labelsize=20,
        right=True)

ax1.set_ylim([0.00,1.00])
ax0.set_ylim([0.00,1.00])
ax1.set_yticks(np.arange(1,11,2)*.10)
ax0.set_yticks(np.arange(1,11,2)*.10)

ax1.set_xlim([tanls[0] - 0.05, tanls[-1] + 0.05])
ax0.set_xlim([tanls[0] - 0.05, tanls[-1] + 0.05])
#ax0.set_yscale('log')
#ax1.set_yscale('log')

if mda == "true":
    title = 'MDA, ensemble size=21, shift=' + str(shift)  
else:
    title = 'SDA, ensemble size=21, shift=' + str(shift) 

fig.legend(line_list, line_labs, fontsize=18, ncol=4, loc='upper center')
plt.figtext(.05, .04, r'RMSE versus $\Delta t$', horizontalalignment='left', verticalalignment='top', fontsize=24)
plt.figtext(.95, .04, r'Spread versus $\Delta t$', horizontalalignment='right', verticalalignment='top', fontsize=24)
plt.figtext(.50, .02, title, horizontalalignment="center", verticalalignment='center', fontsize=24)

plt.show()
