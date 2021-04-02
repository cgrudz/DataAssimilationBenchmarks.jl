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
method_list = ["etkf", "enkf-n"]
stats = ['anal', 'fore']
markerlist = ['+', 'x']
color_list = ['#1b9e77', '#d95f02']
total_ens = 42
plot_range = len(range(15,total_ens))

fig = plt.figure()
ax1 = fig.add_axes([.520, .10, .43, .72])
ax0 = fig.add_axes([.050, .10, .43, .72])

f = h5.File('./processed_filter_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0") +'_nanl_40000_burn_05000.h5', 'r')

def find_optimal_values(method, stat, data):
    tuning_stat = 'anal'
    tuned_rmse = np.array(data[method + '_' + tuning_stat + '_rmse'])
    tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
    ens = len(tuned_rmse_min_vals)
    
    stat_rmse = np.array(data[method +'_' + stat + '_rmse'])
    stat_spread = np.array(data[method + '_' + stat + '_spread'])

    rmse_vals = np.zeros([ens])
    spread_vals = np.zeros([ens])

    for j in range(ens):
        min_val = tuned_rmse_min_vals[j]
        indx = tuned_rmse[j,:] == min_val

        rmse_vals[j] = stat_rmse[j, indx]
        spread_vals[j] = stat_spread[j, indx]

    return [rmse_vals, spread_vals]


line_list = []
line_labs = []
j = 0
k = 0
for meth in method_list:
    for stat in stats:
        if meth == "enkf-n":
            rmse = f[meth + '_' + stat + '_rmse']
            spread = f[meth + '_' + stat + '_spread']

        else:
            rmse, spread = find_optimal_values(meth, stat, f)

        l, = ax0.plot(range(15, total_ens), rmse[:plot_range], marker=markerlist[j], linewidth=2, markersize=20, color=color_list[k])
        ax1.plot(range(15, total_ens), spread[:plot_range], marker=markerlist[j], linewidth=2, markersize=20, color=color_list[k])
        line_list.append(l)

        if stat == 'anal':
            stat_name = 'Analysis'

        elif stat == 'fore':
            stat_name = 'Forecast'

        if meth == "enkf-n":
            meth_name = "EnKF-n"
        elif meth == "etkf":
            meth_name = "ETKF"
        elif meth == "enkf":
            meth_name = "EnKF"
        
        line_labs.append(meth_name + ' ' + stat_name)
        k+=1 
        k = k % 2

    j+=1


ax1.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True)

ax0.tick_params(
        labelsize=20,
        right=True)

ax1.set_ylim([0.00,0.40])
ax0.set_ylim([0.00,0.40])
ax1.set_yticks(np.arange(0,41,2)*.01)
ax0.set_yticks(np.arange(0,41,2)*.01)

ax1.set_xlim([14.5, total_ens])
ax0.set_xlim([14.5, total_ens])
ax1.set_xticks(range(15, total_ens, 2))
ax0.set_xticks(range(15, total_ens, 2))
#ax0.set_yscale('log')
#ax1.set_yscale('log')



fig.legend(line_list, line_labs, fontsize=24, ncol=2, loc='upper center')
plt.figtext(.05, .04, 'RMSE versus ensemble size', horizontalalignment='left', verticalalignment='top', fontsize=24)
plt.figtext(.95, .04, 'Spread versus ensemble size', horizontalalignment='right', verticalalignment='top', fontsize=24)
plt.figtext(.50, .02, r'$\Delta$t=' + str(tanl).ljust(4,"0"), horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
