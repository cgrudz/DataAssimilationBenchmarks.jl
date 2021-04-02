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

tanl = 0.05
obs_un = 1.0
method_list = ["etks_classic", "etks_single_iteration", "ienks-bundle", "ienks-transform"]
#method_list = ["enks-n-dual_classic", "enks-n-dual_single_iteration", "ienks-n-bundle", "ienks-n-transform"]
stats = ['post', 'filt', 'fore']
markerlist = ['+', 'x', "d", "o", '^']
markersizes = [24, 24, 16, 16, 16]
color_list = ['#1b9e77', '#d95f02', '#7570b3']
ensemble = 5
total_lag = 53
plot_range = len(range(1,total_lag,3))
mda = 'false'


fig = plt.figure()
ax1 = fig.add_axes([.520, .10, .43, .72])
ax0 = fig.add_axes([.050, .10, .43, .72])

f = h5.File('./processed_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0") +'_nanl_20000_burn_05000_mda_' + mda + '.h5', 'r')

def find_optimal_values(method, stat, data):
    tuning_stat = 'post'
    tuned_rmse = np.array(data[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
    lag, ens = np.shape(tuned_rmse_min_vals)
    
    stat_rmse = np.array(data[method +'_' + stat + '_rmse'])
    stat_spread = np.array(data[method + '_' + stat + '_spread'])

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


line_list = []
line_labs = []
j = 0
k = 0
for meth in method_list:
    for stat in stats:
        if meth[:6] == "enks-n" or \
           meth[:7] == "ienks-n":
            rmse = np.transpose(np.array(f[meth +"_" + stat + "_rmse"]))
            spread = np.transpose(np.array(f[meth +"_" + stat + "_spread"]))
            rmse = rmse[::-1,ensemble]
            spread = spread[::-1,ensemble]

        else:
            rmse, spread = find_optimal_values(meth, stat, f)
            rmse = rmse[::-1,ensemble]
            spread = spread[::-1,ensemble]
        
        l, = ax0.plot(range(1, total_lag, 3), rmse[:plot_range], marker=markerlist[j], linewidth=2, markersize=markersizes[j], color=color_list[k])
        ax1.plot(range(1, total_lag, 3), spread[:plot_range], marker=markerlist[j], linewidth=2, markersize=markersizes[j], color=color_list[k])
        line_list.append(l)

        if stat == 'post':
            stat_name = 'smoother'

        elif stat == 'filt':
            stat_name = 'filter'

        elif stat == 'fore':
            stat_name = 'forecast'

        if meth == "etks_classic":
            meth_name = "ETKS classic"
        elif meth == "etks_single_iteration":
            meth_name = "SIETKS"
        elif meth == "enks-n-dual_classic":
            meth_name = "EnKS-N-dual classic"
        elif meth == "enks-n-dual_single_iteration":
            meth_name = "SIETKS-N-dual"
        elif meth == "ienks-bundle":
            meth_name = "IEnKS bundle"
        elif meth == "ienks-n-bundle":
            meth_name = "IEnKS-N bundle"
        elif meth == "ienks-transform":
            meth_name = "IEnKS transform"
        elif meth == "ienks-n-transform":
            meth_name = "IEnKS-N transform"
        
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

ax1.set_ylim([0.00,0.30])
ax0.set_ylim([0.00,0.30])
ax1.set_yticks(np.arange(0,31,2)*.01)
ax0.set_yticks(np.arange(0,31,2)*.01)

ax1.set_xlim([0.5, total_lag])
ax0.set_xlim([0.5, total_lag])
ax1.set_xticks(range(1, total_lag,3))
ax0.set_xticks(range(1, total_lag ,3))
#ax0.set_yscale('log')
#ax1.set_yscale('log')



fig.legend(line_list, line_labs, fontsize=18, ncol=4, loc='upper center')
plt.figtext(.05, .04, 'RMSE versus lag', horizontalalignment='left', verticalalignment='top', fontsize=24)
plt.figtext(.95, .04, 'Spread versus lag', horizontalalignment='right', verticalalignment='top', fontsize=24)
plt.figtext(.50, .02, r'$\Delta$t=' + str(tanl).ljust(4,"0")+', shift=1, mda=' + mda + ', ensemble size='+ str(range(15,44,2)[ensemble]), horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
