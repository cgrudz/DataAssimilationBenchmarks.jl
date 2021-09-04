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

methods = ["ienks-transform", "ienks-n-transform", "ienks-transform"]
tanl = 0.05
total_lag = 92
total_ens = 42
shift = 1

fsda = h5.File('./processed_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0") +'_nanl_20000_burn_05000_mda_false' + \
        '_shift_' + str(shift).rjust(3, "0") + '.h5', 'r')

fmda = h5.File('./processed_smoother_state_diffusion_0.00_tanl_' + str(tanl).ljust(4,"0") +'_nanl_20000_burn_05000_mda_true' + \
        '_shift_' + str(shift).rjust(3, "0") + '.h5', 'r')

def find_optimal_values(method, data):
    tuning_stat = 'fore'
    tuned_rmse = np.array(f[method + '_' + tuning_stat + '_rmse'])
    method_iteration_means = np.array(f[method + '_iteration_mean'])
    method_iteration_stds = np.array(f[method + '_iteration_std'])
    method_iteration_medians = np.array(f[method + '_iteration_median'])
    
    tuned_rmse_nan = np.isnan(tuned_rmse)
    method_iteration_means[tuned_rmse_nan] = np.inf
    method_iteration_stds[tuned_rmse_nan] = np.inf
    method_iteration_medians[tuned_rmse_nan] = np.inf
    tuned_rmse[tuned_rmse_nan] = np.inf
    
    dims = np.shape(tuned_rmse)

    if len(dims)==2:
        iteration_mean = np.transpose(method_iteration_means)
        iteration_std = np.transpose(method_iteration_stds)
        iteration_median = np.transpose(method_iteration_medians)


    else:
        tuned_rmse_min_vals = np.min(tuned_rmse, axis=1)
        lag, ens = np.shape(tuned_rmse_min_vals)
        
        iteration_median = np.zeros([lag, ens])
        iteration_mean = np.zeros([lag, ens])
        iteration_std = np.zeros([lag, ens])

        for i in range(lag):
            for j in range(ens):
                min_val = tuned_rmse_min_vals[i,j]
                indx = tuned_rmse[i,:,j] == min_val
                tmp_rmse = tuned_rmse[i, indx, j]
                tmp_iteration_mean = method_iteration_means[i, indx, j] 
                tmp_iteration_std = method_iteration_stds[i, indx, j] 
                tmp_iteration_median = method_iteration_medians[i, indx, j] 

                if len(tmp_rmse) > 1:
                    tmp_iteration_mean = tmp_iteration_mean[0]
                    tmp_iteration_std = tmp_iteration_std[0]
                    tmp_iteration_median = tmp_iteration_median[0]

                iteration_mean[i,j] = tmp_iteration_mean
                iteration_std[i,j] = tmp_iteration_std
                iteration_median[i,j] = tmp_iteration_median

   
        iteration_mean = np.transpose(iteration_mean)
        iteration_std = np.transpose(iteration_std)
        iteration_median = np.transpose(iteration_median)

    return [iteration_mean, iteration_std]


fig = plt.figure()
axa = fig.add_axes([.935, .085, .020, .395])
ax0 = fig.add_axes([.935, .510, .020, .395])
ax1 = fig.add_axes([.070, .085, .280, .395])
ax2 = fig.add_axes([.360, .085, .280, .395])
ax3 = fig.add_axes([.650, .085, .280, .395])
ax4 = fig.add_axes([.070, .510, .280, .395])
ax5 = fig.add_axes([.360, .510, .280, .395])
ax6 = fig.add_axes([.650, .510, .280, .395])

color_map1 = sns.cubehelix_palette(n_colors=100, rot=1.75)
color_map2 = sns.cubehelix_palette(n_colors=100, rot=1.75, start=1.5)
color_maps = [color_map1, color_map2]
max_scale2 = 5.0
min_scale2 = 0.0

max_scale1 = 10.0
min_scale1 = 1.0

max_scales = [max_scale1, max_scale2]
min_scales = [min_scale1, min_scale2]

ax_list = [ax4, ax1, ax5, ax2, ax6, ax3]
scale_list = [ax0,axa]
j = 0

for method in methods:
    if j == 0 or j == 2:
        f = fsda
    else:
        f = fmda

    iterations = find_optimal_values(method, f)
    
    for i in range(0,2):
        sns.heatmap(iterations[i], linewidth=0.5, ax=ax_list[i+j], cbar_ax=scale_list[i], vmin=min_scales[i], vmax=max_scales[i], cmap=color_maps[i])
    j += 2


x_labs = []
x_tics = []
x_vals = range(15, total_ens, 2)

for i in range(len(x_vals)):
    if i % 2 == 0:
        x_labs.append(str(x_vals[i]))
        x_tics.append(i)

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


ax1.set_yticks(y_tics)
ax2.set_yticks(y_tics)
ax3.set_yticks(y_tics)
ax4.set_yticks(y_tics)
ax5.set_yticks(y_tics)
ax6.set_yticks(y_tics)
ax1.set_yticklabels(y_labs, va="bottom", rotation=0)
ax4.set_yticklabels(y_labs, va="bottom", rotation=0)

ax1.set_xticks(x_tics)
ax2.set_xticks(x_tics)
ax3.set_xticks(x_tics)
ax1.set_xticklabels(x_labs, rotation=0, ha="left")
ax2.set_xticklabels(x_labs, rotation=0, ha="left")
ax3.set_xticklabels(x_labs, rotation=0, ha="left")

ax0.tick_params(
        labelsize=20)

axa.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20)

ax2.tick_params(
        labelleft=False,
        labelsize=20)

ax3.tick_params(
        labelleft=False,
        labelsize=20)

ax4.tick_params(
        labelsize=20,
        labelbottom=False)

ax5.tick_params(
        labelleft=False,
        labelbottom=False)

ax6.tick_params(
        labelleft=False,
        labelbottom=False)

fig_title = r"Iteration statistics, $S=$1, $\Delta$t="+ str(tanl)


plt.figtext(.210, .905, "IEnKS SDA", horizontalalignment='center', verticalalignment='bottom', fontsize=20)
plt.figtext(.500, .905, "IEnKS-N", horizontalalignment='center', verticalalignment='bottom', fontsize=20)
plt.figtext(.790, .905, "IEnKS MDA", horizontalalignment='center', verticalalignment='bottom', fontsize=20)

plt.figtext(.025, .2925, r'Std', horizontalalignment='center', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.025, .7075, r'Mean', horizontalalignment='center', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.015, .52, r'$L$', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .02, r'$N_e$', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .97, fig_title,
        horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
